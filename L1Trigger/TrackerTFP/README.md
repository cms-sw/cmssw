This directory contains L1 tracking code used by the TMTT & Hybrid algorithms.

=== Run Instructions ===

    cmsRun L1Trigger/TrackerTFP/test/test_cfg.py Events=<n>

runs the clock and bit accurate emulation of the TMTT chain. In the run script one may want to change the used event files or tracker geometry. The option CheckHistory in L1Trigger/TrackerTFP/python/Producer_cfi.py is set to false by default but is highly recommended to be set to true if one runs the emulator. This will automatically test if the configuration of the emulation run is consistent with the configuration of the input evevnt production and points out for example if one chooses a different tracker geometry.

Apart from producing TTTrack collection as the f/w will, test_cfg.py analyses the results. It provides a end-of-job summary, which reports data rates and tracking efficiencies at the end of each processing step. The definition of which Tracking Particles are taken into account for this efficiency measurements are described here: L1Trigger/TrackTrigger/python/ProducerSetup_cfi.py in the PSet TrackTrigger_params.TrackingParticle. The "maximal possible tracking efficiency" reported for tracking steps part way through the chain is derived assuming zero efficiency loss in subsequent steps. This method allows to assess which processing steps cause most inefficiency. Additionally a "lost tracking efficiency" is reported, which is the loss due to truncation as a result of bottlenecks in the data throughput of the implemented design. Beside this end job summary test_cfg.py produces Hist.root which contains histograms with more details like efficiencies over certain tracking particle parameter.

    cmsRun L1Trigger/TrackerTFP/test/demonstrator_cfg.py Events=<n>

runs the clock-and-bit-accurate emulation of the TMTT chain, calls questasim to simulate the f/w and compares the results of both. A single bit error interrupts the run. This script is used to validate the clock and bit accurate emulation code. In L1Trigger/TrackerTFP/python/Demonstrator_cfi.py one has to specify the input and output stage of the chain one wants to test, the location where the IPBB (https://ipbus.web.cern.ch/doc/user/html/firmware/ipbb-primer.html) project is located and the runtime of the f/w simulator. Additionally one needs to ensure that the system variable PATH contains the questasim executables.

=== Configuration ===

All configuration params to manipulate the algorithms one may want to play with can be found in L1Trigger/TrackTrigger/python/ProducerSetup_cfi.py which is also be used by L1Trigger/TrackerDTC/, which emulates the DTC, (which is bit-accurate but not clock-accurate (since one would need to interfere events which is not possible with EDProducer)).

=== Code structure ===

There are 6 TMTT algorithm steps: GP (Geometric Process), HT (Hough Transform), MHT (Mini Hough Transform), ZHT (r-z Hough Transform), KF (Kalman Filter), DR (Duplicate Removal). Each comes with one EDProducer, one EDAnalyzer and one class which contains the actual emulation of this step for one nonant (1/9 phi slice of outer tracker). Their EDProducts combine the connection to MCTruth (and does not conatain MCTruth) via edm::Refs of either TTStubs or TTTracks with the actual bits used in h/w via std::bitset<64> using a std::pair of those objects.
The track-finding firmware is described in a highly parallel fashion. On the one hand, one has multiple worker nodes for each step and on the other hand is the process pipelined in a way to receive potentially one product per clock tick. This parallelism is reflected by a two dimensional vector of the earlier mentioned pairs. The inner vector describes the output (also called Frame) of one working node per clock tick. If the f/w produces no valid product in a given clock tick, then the bitset will be all '0' and the edm:ref will be null. Valid products do not necessarily form a contiguous block of valid products. The outer vector will have one entry per worker node (also called channel) where all nodes for the whole tracker are counted. Each EDProducer will produce two branches: Accepted Track or Stub and Lost Track or Stub. The Lost branch contains the Tracks or Stubs which are lost since they got not processed in time. Since the KF uses Tracks and Stubs as input the EDProducer ProducerZHTout is used to form TTTracks after the ZHT and ProducerKFin to create the edm::ref to this TTTracks. Finally ProducerTT takes the h/w liked structured output from the KF and produces one collection of TTTracks and ProducerAS creates a map between KF input TTracks to KF output TTTracks.

There are 5 additional classes in L1Trigger/TrackerTFP. DataFormats describes Stubs and Tracks for each process step and automates the conversion from floating points to bits as used in h/w and vice versa. Demonstrator allows one to compare s/w with f/w. KalmanFilterFormats describes the used precisions in the Kalman Filter. LayerEncoding allows one to transform the layer encoding used before the KF into the encoding after KF. State is a helper class to simplify the KalmanFilter code.

In order to simplify the conversion of floating point values into arbitrary long (within 64 bit) binary or twos-complement number, the class DataFormats/L1TrackTrigger/interface/TTBV.h has been created.

In order to simplify the tracking efficiency measurement the class StubAssociator in SimTracker/TrackTriggerAssociation/ has been created.
