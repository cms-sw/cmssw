This directory contains L1 tracking code used by the TMTT & Hybrid algorithms.

=== Run Instructions ===

    cmsRun L1Trigger/TrackerTFP/test/test_cfg.py Events=<n>

runs the clock and bit accurate emulation of the TMTT chain. In the run script one may want to change the used event files or tracker geometry.

Apart from producing TTTrack collection as the f/w will, test_cfg.py analyses the results. It provides a end-of-job summary, which reports data rates and tracking efficiencies at the end of each processing step. The definition of which Tracking Particles are taken into account for this efficiency measurements are described here: L1Trigger/TrackTrigger/python/ProducerSetup_cfi.py in the PSet TrackTrigger_params.TrackingParticle. The "maximal possible tracking efficiency" reported for tracking steps part way through the chain is derived assuming zero efficiency loss in subsequent steps. This method allows to assess which processing steps cause most inefficiency. Beside this end job summary test_cfg.py produces Hist.root which contains histograms with more details like efficiencies over certain tracking particle parameter.

    cmsRun L1Trigger/TrackerTFP/test/demonstrator_cfg.py Events=<n>

runs the clock-and-bit-accurate emulation of the TMTT chain, calls questasim to simulate the f/w and compares the results of both. A single bit error interrupts the run. This script is used to validate the clock and bit accurate emulation code. In L1Trigger/TrackerTFP/python/Demonstrator_cfi.py one has to specify the input and output stage of the chain one wants to test, the location where the IPBB (https://ipbus.web.cern.ch/doc/user/html/firmware/ipbb-primer.html) project is located and the runtime of the f/w simulator. Additionally one needs to ensure that the system variable PATH contains the questasim executables.

=== Configuration ===

All configuration params to manipulate the algorithms one may want to play with can be found in L1Trigger/TrackTrigger/python/ProducerSetup_cfi.py which is also be used by L1Trigger/TrackerDTC/, which emulates the DTC, (which is bit-accurate but not clock-accurate (since one would need to interfere events which is not possible with EDProducer)).

=== Code structure ===

There are 7 TMTT algorithm steps: GP (Geometric Process), HT (Hough Transform), CTB (Clean Track Builder), KF (Kalman Filter), DR (Duplicate Removal), TQ (Track Quality), TFP (Track Finding Processor). Each comes with one EDProducer, one EDAnalyzer and one class which contains the actual emulation of this step for one nonant (1/9 phi slice of outer tracker). Their EDProducts combine the connection to MCTruth (and does not conatain MCTruth) via edm::Refs of either TTStubs or TTTracks with the actual bits used in h/w via std::bitset<64> using a std::pair of those objects.
The track-finding firmware is described in a highly parallel fashion. On the one hand, one has multiple worker nodes for each step and on the other hand is the process pipelined in a way to receive potentially one product per clock tick. This parallelism is reflected by a two dimensional vector of the earlier mentioned pairs. The inner vector describes the output (also called Frame) of one working node per clock tick. If the f/w produces no valid product in a given clock tick, then the bitset will be all '0' and the edm:ref will be null. Valid products do not necessarily form a contiguous block of valid products. The outer vector will have one entry per worker node (also called channel) where all nodes for the whole tracker are counted. Since the KF uses Tracks and Stubs as input the EDProducer ProducerCTB is used to form TTTracks after the HT. Finally ProducerTFP takes the h/w liked structured output from the TQ and produces one collection of TTTracks.

There are 5 additional classes in L1Trigger/TrackerTFP. DataFormats describes Stubs and Tracks for each process step and automates the conversion from floating points to bits as used in h/w and vice versa. Demonstrator allows one to compare s/w with f/w. KalmanFilterFormats describes the used precisions in the Kalman Filter. LayerEncoding allows one to transform the layer encoding used before the KF into the encoding after KF. State is a helper class to simplify the KalmanFilter code.

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