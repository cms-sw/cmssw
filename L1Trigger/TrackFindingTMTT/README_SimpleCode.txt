-------------------------------------------------------------------------------------
* How to setup the framework and how to run our analysis code:

The code is in GITHUB. It can be browsed in in
https://github.com/CMS-TMTT/cmssw/blob/TMTT_1060/L1Trigger/TrackFindingTMTT/ . 
(An older version of the code can be found in SVN in https://svnweb.cern.ch/cern/wsvn/UK-TrackTrig/software/cmssw/trunkSimpleCode9).

- Setup a CMSSW environment. 
cmsrel CMSSW_10_6_0
cd CMSSW_10_6_0/src
cmsenv

- Software checkout

Check out from GIT following instructions in https://github.com/CMS-TMTT/cmssw/blob/TMTT_1060/L1Trigger/TrackFindingTMTT/README.md .

- MC samples 
  You can use the RelVal samples in L1Trigger/TrackFindingTMTT//test/MCsamples/1040/RelVal/ or 
  L1Trigger/TrackFindingTMTT/test/MCsamples/937/RelVal/ .
  These are based on CMS geometry D21 or D17 respectively. 

  It is strongly recommended to copy a sample of events to your local scratch disk, since this will run much
  faster than reading them from dcache or XROOTD.

- Optionally, change the configuration parameters. (See section "Changing configuration options" below).

- Compile:
    scram b -j8

- Run the code
    cd L1Trigger/TrackFindingTMTT/test/
    cmsRun   tmtt_tf_analysis_cfg.py 
    
    or with options:

    cmsRun   tmtt_tf_analysis_cfg.py Events=50 inputMC=../../../Samples91X/930pre3/TTbar/PU200.txt histFile=outputHistFile.root makeStubs=1

- Look at the printout from the job. At the end, it prints the number of track candidates reconstructed
  and the algorithmic tracking efficiency.

- Look at the analysis histograms (which are explained in '(11) Class "Histos"' below).
    root Hist.root
    TBrowser b

-------------

=== Producing stubs ===:

If you want to remake stubs on the fly (e.g. with different window sizes), you can do:

cmsRun tmtt_tf_analysis_cfg.py makeStubs=1

This will run the CMSSW modules that produce the stubs (and truth association) before running our tracking algorithms.
Note this will take a long time to run (~100 events per hour for ttbar+200PU), so is only suitable for
debugging on a small number of events.

To produce the stubs, add them to the event content, and output the new collections to file
(along with the existing event content), you can run make_stubs.py:

cmsRun make_stubs.py inputMC=../../../Samples91X/930pre3/TTbar/PU200.txt Events=50

This will produce an output file output.root, which can then be read by tmtt_tf_analysis_cfg without having to remake the stubs.
i.e. you can then run with (or omit the makeStubs argument as it's False by default):

cmsRun tmtt_tf_analysis_cfg.py makeStubs=0

=== Changing configuration options ===:

a) The full set of config parameters, which comments explaining what each is, can be found in
specifiied in L1Trigger/TrackFindingTMTT/python/TMTrackProducer_Defaults_cfi.py. 

b) The file mentioned on (a) is imported by
L1Trigger/TrackFindingTMTT/python/TMTrackProducer_cff.py ,
which can optionally override the values of some parameters. This file lists the subset of the cfg
parameters that are most useful.

e.g. Enable tracking down to 2 GeV, enabled displaced tracking etc.

c) Alternatively, you can use L1Trigger/TrackFindingTMTT/python/TMTrackProducer_Ultimate_cff.py ,
which is like (b) but includes improvements not yet available in the firmware, such as reducing the
Pt threshold to 2 GeV. It is suitable for L1 trigger studies.

d) You can also override cfg parameters in tmtt_tf_analysis_cfg.py . A line there illustrates how.
This file imports TMTrackProducer_cff.py.

-------------

=== Software structure ===:

1) Class "TMTrackProducer" -- This is the main routine, which uses classes: "InputData" to unpack the useful
data from the MC dataset, and "Sector" & "HTphi" to do the L1 Hough transform track-finding,
and "Get3Dtracks" to estimate the helix params in 3D, optionally by running an r-z track filter.
It creates matrices of "Sector", "HTphi" & "Get3Dtracks", where the matrix elements each correspond to 
a different (phi,eta) sector. It then uses "TrackFitGeneric" to do the track fitting and optionally
"KillDupFitTrks" to remove duplicate tracks after the fit. It employs "Histos" to create the analysis
 histograms. 
   To allow comparison of our tracks with those of the AM & Tracklet groups, it also converts our tracks
to the agree common TTTrack format, with this conversion done by the "ConverterToTTTrack" class.

2) Class "InputData" -- This unpacks the most useful information from the Stubs and Tracking Particle 
(truth particles) collections in the MC dataset, and it for convenient access in the "Stub" and "TP"
classes. The "Stub" class uses a class called "DigitalStub" to digitize and then undigitize again
the stub data. This process degrades slightly the resolution, as would happen in the real firmware.
The digitisation is optional. It is called from TMTrackProducer after the stubs have been assigned to
sectors.

3) Class "Sector" -- This knows about the division of the Tracker into (phi,eta) sectors that we use
for the L1 tracking, and decides which of these sectors each stub belongs to.

4) Class "HTrphi" implements the Hough transforms in the r-phi plane. It inherits from
a base class "HTbase". The Hough transform array is implemented as a matrix of "HTcell" 
objects. The HTrphi class stores tracks it finds using the "L1track2D" class. It optionally 
uses class "KillDupTrks" to attempt to eliminate duplicate tracks. And optionally, class "MuxHToutputs"
can be used to multiplex the tracks found in different HT arrays (sectors) onto a single output
optical link pair.

5) Class "HTcell" -- This represents a single cell in an HT array. It provides functions allowing stubs
to be added to this cell, to check if the stubs in a cell give a good track candidate, and to check
if this matches a tracking particle (truth).

6) Class "Get3Dtracks" makes an estimate of the track parameters in 3D, stored in the "L1track3D" 
class, by taking the r-phi track found by the HT, assuming z0 = 0 and that eta is given by the centre 
of the eta sector that track is in. Optionally it can also create a second collection of L1track3D,
by running an r-z track filter (from class "TrkRZfilter") on the tracks found by the HT, which gives
cleaner tracks with more precise r-z helix param info.

7) Class "L1track2D" represents a 2D track, reconstructed in the r-phi or r-z plane by a Hough transform.
Class "L1track3D" represents a 3D tracks, obtained by combining the information in the 2D tracks
from r-phi and r-z Hough transforms. These classes give access to the stubs associated to each track,
to the reconstructed helix parameters, and to the associated truth particle (if any). They represent
the result of the track finding. Both inherit from a pure virtual class L1trackBase, which contains
no functionality but imposes common function names.

8) Class "KillDupTrks" contains algorithms for killing duplicate tracks found within a single
HT array. Class "KillDupFitTrks" contains algorithms for killing duplicate fitted tracks.

9) Class "TrkRZfilter" contains r-z track filters, such as the Seed Filter, that check if the stubs
on a track are consistent with a straight line in the r-z plane.

10) Class "TrackFitGeneric" does one (or more) helix fit(s) to the track candidates, using various
other classes that implement linearized chi2, linear regression or Kalman filter fits. These are:

   - ChiSquared4ParamsApprox (chi2 linear fit, with maths simplified for easier use in FPGA)
   - SimpleLR (linear regression fit, which is similar to chi2 fit, but assumes all hits have same uncertainty).
   - KF4ParamsComb & KF5ParamsComb: Kalman Filter fits to a 4 or 5 parameter helix.

The fit also uses a couple of dedicated utility classes (Matrix & kalmanState & StubCluster).

11) Class "L1fittedTrack" contains the result of running a track fitter (the algorithm for which is 
implemented in class "TrackFitAlgo") on the L1track3D track candidate found by the Hough transform. 
It gives access to the fitted track parameters and chi2, and via a link to the L1track3D candidate 
that produced it, also to the stubs on the track and the associated truth particle (if any). 
It inherits from the pure virutal L1trackBase, ensuring it has some common classes with L1track3D and 
L1track2D.

12) Class "L1fittedTrk4and5" contains a pair of L1fittedTrack objects, containg the result of doing
either a 4 or 5 parameter helix fit to the track, where the former assumes d0 = 0.

13) Class "DegradeBend" -- This is used by class "Stub" to degrade the resolution on the stub
bend information to that expected in the electronics, as opposed to that currently in CMSSW.

14) "Utility" -- contains a few useful functions, that are not part of a class.

15) Class "Settings" -- Reads in the configuration parameters.

16) Class "Histos" -- Books and fills all the histograms. There are several categories of histograms,
with each category being booked/filled by its own function inside "Histos", and being placed inside its
own ROOT directory in the output histogram file. The categories are "InputData" = plots made with the 
Stubs & Tracking Particles; "CheckEtaPhiSectors" = plots checking assignment of stubs to (eta,phi) 
sectors; "HT" = plots checking how stubs are stored in the Hough Transform arrays; "TrackCands" = plots 
showing number of track candidates found & investigating why tracking sometimes failed, 
"EffiAndFakeRate" = plots of tracking efficiency. 

Each user of the code will probably want to book their own set of histograms inside "Histos". So 
just consider the version of this class in GIT as a set of examples of how to do things. Don't feel
obliged to understand what every histogram does.

17) Class "DeadModuleDB" is used both to emulate dead modules by killing stubs in certain tracker
regions, and to recover efficiency caused by dead modules indicating in which sectors looser
track-finding cuts are required.

Class "StubKiller" also emulates dead modules. It was written in collaboration with Tracklet to
model the scenarios requested by the Stress Test committee. If it is used, then the emulation in
DeadModuleDB should not be.

18) SimTracker/TrackTriggerAssociation/ contains a modification to the official L1 track to TrackingParticle
matching software used by Louise Skinnari's official L1 track analysis code. This modification (made by 
Seb Viret) to TTTrackAssociator.h allows one incorrect hit on L1 tracks, whereas the original matching code
allowed none.

------

To update the dOxygen documentation, just type "doxygen". This creates the web page inside html/
A recent version of this documentation is in http://tomalini.web.cern.ch/tomalini/IanSimpleCode/hierarchy.html .

=== To output TTTrack dataset to EDM output_dataset.root file (e.g. for L1 trigger studies) ===

Before doing "cmsRun tmtt_tf_analysis_cfg.py", edit this script to set "outputDataset = 1". 
You may also need to edit the python associating the TMTT L1 tracks to the MC truth particles,
to specify which track fitter you are using.

If you only want to make TTTracks, and don't care about the TMTT histograms or job summary, you 
can save CPU by setting cfg params:

TMTrackProducer.EnableMCtruth = cms.bool(False)
TMTrackProducer.EnableHistos  = cms.bool(False)

=== To run Louise Skinnari's official CMS L1 track performance analysis code ===

i) This runs on the TTTrack objects produced by our TMTrackProducer. As the Tracklet group also 
produce TTTracks, this performance code can be used by all groups. (N.B. When not comparing our
results with another group, our own tmtt_tf_analysis_cfg.py analysis software described
above is usually more convenient).

ii) To include the official analysis code in your setup, follow this recipe:

cmsrel CMSSW_9_3_8
cd CMSSW_9_3_8/src 
cmsenv 

# Checkout the directory containing the ntuple maker analysis code from git.
git cms-merge-topic skinnari:Tracklet_932

# Checkout the analysis and plotting scripts from gitlab
git clone https://gitlab.cern.ch/cms-tracker-phase2-backend-development/BE_software/L1TrackTools.git

# Checkout our private analysis code from git as explained previously.
(see above)

# Compile
scramv1 b -j8

iii) To run it, do:

    cd L1Trigger/TrackFindingTMTT/test/
    cmsRun L1TrackNtupleMaker_cfg.py trkFitAlgo=All 

This runs our TMTrackProducer code, exactly as before, and configured with the same python files.
This will also run the tracklet track finding code.

The argument trkFitAlgo specifies which track finding algorithms to run.
Possible options are KF4ParamsComb, SimpleLR, Tracklet, All

KF4ParamsComb : TMTT chain with KF4ParamsComb track fitter (HT+KF+DR)
SimpleLR : TMTT chain with seed filter and SimpleLR track fitter (HT+SF+LR+DR)
Tracklet : Runs the default tracklet tracking from skinnari:Tracklet_93X
All : Runs all three

Note that you can run KF4ParamsComb chain without the SF and SimpleLR with the SF in the same job.
You can also specify a comma separated list e.g. if you just wanted to run two of the fitters, you 
can specify trkFitAlgo=SimpleLR,Tracklet

It then runs CMS-agreed code to produce an ntuple from these tracks:

  L1Trigger/TrackFindingTracklet/test/L1TrackNtupleMaker.cc

iv) When the job has finished, you will see a Hist.root file containing all the histograms produced
by our standard analysis TMTrackProducer, plus in addition the ntuples.
There will be one ntuple corresponding to the tracks produced by each track fitter you ran with, 
and one ntuple for the tracklet tracks.

v) To make .png files containing histograms of tracking efficiency, resolution etc., start root & type
the two commands:
       .L ../../../L1TrackTools/L1TrackNtuplePlot.C++
       L1TrackNtuplePlot("Hist","_TMTT_KF4ParamsComb")

Altenartively, you can copy/move your output Hist.root file to the L1TrackTools directory:
        mv Hist.root ../../../L1TrackTools/
        cd ../../../L1TrackTools/
        .L L1TrackNtuplePlot.C++
        L1TrackNtuplePlot("Hist","_TMTT_KF4ParamsComb")

After running, the histograms will be in a root file called "output_Hist_TMTT_KF4ParamsComb.root",
and also saved as pdf/png in the TrkPlots directory (if this directory doesn't exist, you will need
to create it and rerun).

The arguments of the macro are explained around line 45, and here are some usage examples:

To produce plots for muons (from the primary interaction), for the KF4ParamsComb fitter:
      .L L1TrackNtuplePlot.C++
      L1TrackNtuplePlot("Hist", "_TMTT_KF4ParamsComb", 0, 13 )

To produce plots for all TP in jets with pt>100 GeV:
      .L L1TrackNtuplePlot.C++
      L1TrackNtuplePlot("Hist", "_TMTT_KF4ParamsComb", 2, 0 )

Note that by default the efficiency is defined for TPs with pt>2GeV.  To consider pt>3GeV, you need to specify a few more arguments.
e.g. to produce plots for TP with pt>3GeV in jets with pt>100 GeV:
      .L L1TrackNtuplePlot.C++
      L1TrackNtuplePlot("Hist", "_TMTT_KF4ParamsComb", 2, 0, 0, false, false, 3 )
