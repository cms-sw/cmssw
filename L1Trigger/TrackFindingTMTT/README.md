Two options:

1) Run the TMTT L1 tracking algorithm (with detailed internal histos)

    cd L1Trigger/TrackFindingTMTT/test/
    cmsRun -n 4 tmtt_tf_analysis_cfg.py 

editing if necessary variables: GEOMETRY (CMS geometry version), inputMCtxt (MC file), makeStubs (regenerate stubs instead of using those from input MC), outputDataSet (write TTTrack collection to file).

- Look at the printout from the job. At the end, it prints the number of track candidates reconstructed
  and the algorithmic tracking efficiency.

- Look at the performance histograms Hist.root (explained in class "Histos" below)

2) cmsRun -n 4 L1TrackNtupleMaker_cfg.py 
after editing it to change L1TRACKALGO = 'TMTT'. This writes a TTree of the fitted L1 tracks to .root file, from which tracking performance can be studied with ROOT macro L1Trigger/TrackFindingTracklet/test/L1TrackNtuplePlot.C. Other values of L1TRACKALGO permit to run the Hybrid or Tracklet emulation, or floating point emulation.

Both (1) & (2) are able to write a dataset containing the TTTrack collection of the fitted tracks.

N.B. .txt files listing available MC samples can be found in https://github.com/cms-data/L1Trigger-TrackFindingTMTT .

-------------

=== Reproducing stubs ===:

The makeStubs option to remake the stubs is very slow. If you need to do this, it may be better to "cmsRun make_stubs.py" to write the new stubs to disk.

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

e) The python files in python/ all disable production of histograms & tracking performance summaries to save CPU. However, tmtt_tf_analysis_cfg overrides this default and switches them on via parameters EnableMCtruth & EnableHistos. If you do not care about this analysis information, and only care about producing TTTracks, then keep them switched off.

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
classes. Info about tracker silicon modules is stored in "ModuleInfo". The "Stub" class uses a class called "DigitalStub" to digitize and then undigitize again the stub data. This process degrades slightly the resolution, as would happen in the real firmware. The digitisation is optional. It is called from TMTrackProducer after the stubs have been assigned to
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
   - KFParamsComb: Kalman Filter fit to a 4 or 5 parameter helix.

The fit also uses a classe to represent the helix state + stubs: KalmanState.

11) Class "L1fittedTrack" contains the result of running a track fitter (the algorithm for which is 
implemented in class "TrackFitAlgo") on the L1track3D track candidate found by the Hough transform. 
It gives access to the fitted track parameters and chi2, and via a link to the L1track3D candidate 
that produced it, also to the stubs on the track and the associated truth particle (if any). 
It inherits from the pure virutal L1trackBase, ensuring it has some common classes with L1track3D and 
L1track2D.

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

17) Class "StubKiller" emulates dead modules. It was written to
model the scenarios requested by the Stress Test committee. 

18) Class "GlobalCacheTMTT" contains data shared by all threads. It includes configuration data and histograms.
