To run the L1 tracking & create a TTree of tracking performance: 

cmsRun L1TrackNtupleMaker_cfg.py

By setting variable L1TRKALGO inside this script, you can change which L1 tracking algo is used. It defaults to HYBRID, which runs Tracklet pattern reco followed by old Kalman track fit.

The version of the hybrid algorithm that corresponds to the current firmware, and includes the new Kalman track fit, can be run by changing L1TRKALGO=HYBRID_NEWKF. It is not yet the default for MC production, as it's tracking performance is not quite has good as HYBRID. e.g. Only a basic duplicate track removal is available for it.

Displaced Hybrid tracking can be run by setting L1TRKALGO=HYBRID_DISPLACED.

The ROOT macros L1TrackNtuplePlot.C & L1TrackQualityPlot.C make track performance & BDT track quality performance plots from the TTree. Both can be run via makeHists.csh .

If you need to modify the cfg params of the algorithm, then TrackFindingTracklet/interface/Settings.h configures the pattern reco stage, (although some parameters there are overridden by l1tTTTracksFromTrackletEmulation_cfi.py). The old KF fit is configured by the constructor of TrackFindingTMTT/src/Settings.cc. The DTC and new KF fit are configured via TrackTrigger/python/ProducerSetup_cfi.py.

For experts
============

1) To make plots to monitor data rates assicuated to truncation after each step in the tracklet pattern reco algo, set writeMonitorData_ = true in Settings.h . This creates txt files, which the ROOT macros in https://github.com/cms-L1TK/TrackPerf/tree/master/PatternReco can then use to study truncation of individual algo steps within tracklet chain.



Firmware emulation
============

=== Run Instructions ===

    cmsRun L1Trigger/TrackFindingTracklet/test/HybridTracksNewKF_cfg.py Events=<n>

runs the clock and bit accurate emulation of the Hybrid chain. In the run script one may want to change the used event files or tracker geometry.

Apart from producing TTTrack collection as the f/w will, HybridTracksNewKF_cfg.py analyses the results. It provides a end-of-job summary, which reports data rates and tracking efficiencies at the end of each processing step. The definition of which Tracking Particles are taken into account for this efficiency measurements are described here: SimTracker/TrackTriggerAssociation/python/StubAssociator_cfi.py in the PSet StubAssociator_params.TrackingParticle. The "maximal possible tracking efficiency" reported for tracking steps part way through the chain is derived assuming zero efficiency loss in subsequent steps. This method allows to assess which processing steps cause most inefficiency. Beside this end job summary HybridTracksNewKF_cfg.py produces Hist.root which contains histograms with more details like efficiencies over certain tracking particle parameter.

    cmsRun L1Trigger/TrackFindingTracklet/test/demonstrator_cfg.py Events=<n>

runs the clock-and-bit-accurate emulation of the Hybrid chain, calls questasim to simulate the f/w and compares the results of both. A single bit error interrupts the run. This script is used to validate the clock and bit accurate emulation code. In L1Trigger/TrackFindingTracklet/python/Demonstrator_cfi.py one has to specify the input and output stage of the chain one wants to test, the location where the IPBB (https://ipbus.web.cern.ch/doc/user/html/firmware/ipbb-primer.html) project is located and the runtime of the f/w simulator. Additionally one needs to ensure that the system variable PATH contains the questasim executables.

=== Configuration ===

All configuration params to manipulate the algorithms one may want to play with can be found in L1Trigger/TrackTrigger/python/Setup_cfi.py which is also be used by L1Trigger/TrackerDTC/, which emulates the DTC, (which is bit-accurate but not clock-accurate (since one would need to interfere events which is not possible with EDProducer)).

=== Code structure ===

There are 6 Hybrid algorithm steps: Tracklet (L1FPGATrackProducer), TM (Track Multiplexer), DR (Duplicate Removal), KF (Kalman Filter), TQ (Track Quality), TFP (Track Finding Processor). We call the last 5 the Track Processing chain. Each step of this chain comes with one EDProducer, one EDAnalyzer and one class which contains the actual emulation of this step for one nonant (1/9 phi slice of outer tracker). Their EDProducts combine the connection to MCTruth (and does not conatain MCTruth) via edm::Refs of either TTStubs or TTTracks with the actual bits used in h/w via std::bitset<64> using a std::pair of those objects.
The track-finding firmware is described in a highly parallel fashion. On the one hand, one has multiple worker nodes for each step and on the other hand is the process pipelined in a way to receive potentially one product per clock tick. This parallelism is reflected by a two dimensional vector of the earlier mentioned pairs. The inner vector describes the output (also called Frame) of one working node per clock tick. If the f/w produces no valid product in a given clock tick, then the bitset will be all '0' and the edm:ref will be null. Valid products do not necessarily form a contiguous block of valid products. The outer vector will have one entry per worker node (also called channel) where all nodes for the whole tracker are counted. Finally ProducerTFP takes the h/w liked structured output from the TQ and produces one collection of TTTracks.

There are 4 additional classes in L1Trigger/TrackFindingTracklet of interest. DataFormats describes Stubs and Tracks for each process step and automates the conversion from floating points to bits as used in h/w and vice versa. Demonstrator allows one to compare s/w with f/w. KalmanFilterFormats describes the used precisions in the Kalman Filter. State is a helper class to simplify the KalmanFilter code. LayerEncoding in L1Trigger/TrackerTFP allows one to transform the layer encoding used before the Track Processing chain the encoding used in the chain.

In order to simplify the conversion of floating point values into arbitrary long (within 64 bit) binary or twos-complement number, the class DataFormats/L1TrackTrigger/interface/TTBV.h has been created.

In order to simplify the tracking efficiency measurement the class StubAssociator in SimTracker/TrackTriggerAssociation/ has been created.

=== Details to commonly used Classes ===

Frame           is a typedef for std::bitset<64> representing the h/w words which level-1 track finder or level-1 trigger boards will receive and transmit per optical link and internal clock tick.

TTBV            Class representing a BitVector used by TrackTrigger emulators. Based on Frame. The class is mainly used to convert h/w-like structured bits into integers and vice versa. A typical constructors receive an integer values, a bit width (number of bits used to represent this value, an error will be thrown when not enough bits are provided.) and a boolean to distinguish between binary and two's complement representation. Multiple operators are provided, e.g bit wise or, or concatenation with another TTBV.

FrameStub       is a typedef for std::pair<TTStubRef, Frame>. This object is used to represent a Stub in TrackTrigger emulators. On the one hand side it connects to the original TTStub and on the other hand it has the bit string used in h/w to represent this stub.

FrameTrack      same as FrameStub but for Tracks

StreamStub      h/w-like structured collection of stubs. Clock ticks where no Stub can be provided are represented by default constructed FrameStubs (edm:Ref recognising being a null ref and bit set being zero'd). This enables to store stubs bit and clock accurately.

StreamTrack     same as StreamStub but for Tracks

DataFormat      Base class to represent formats of a specific variable at a specific processing step. A format is given by a bit width, an boolean to distinguish between signed and unsigned cover as well as an conversion factor to transform between floating point and biased integer representation. These formats are used to transform h/w words (TTBVs) into variables (supporting conversion to int, double, bool or TTBV).

DataFormats     ESProduct which provides access to all DataFormats used by Track Trigger emulators

Setup           ESProduct providing run time constants configuring Track Trigger emulators


