To run Outer Tracker DTC stub processing simulation & analysing it's performance: 

cmsRun L1Trigger/TrackerDTC/test/test_cfg.py Events=<n>

If you need to modify the cfg params of the algorithm, then find them in L1Trigger/TrackerDTC/python/Setup_cfi.py

Firmware emulation
============

=== Run Instructions ===

    cmsRun L1Trigger/TrackerDTC/test/Demonstrator_cfg.py Events=<n>

runs the clock-and-bit-accurate emulation of the Hybrid chain, calls questasim to simulate the f/w and compares the results of both. A single bit error interrupts the run. This script is used to validate the clock and bit accurate emulation code. In L1Trigger/TrackerDTC/python/Demonstrator_cfi.py one has to specify the DTC Ids one wants to test, the location where the IPBB (https://ipbus.web.cern.ch/doc/user/html/firmware/ipbb-primer.html) project is located and the runtime of the f/w simulator. Additionally one needs to ensure that the system variable PATH contains the questasim executables. The questasim call as well as the comparison may be disabled via an enable flag in the config file. In L1Trigger/TrackerDTC/python/Setup_cfi.py one can enable the printout of compile time constants and bend encoding in case these needs updating. In the run script one may want to change the used event files or tracker geometry.

Demonstrator_cfg.py also analyses the results. It provides a end-of-job summary, which reports data rates and tracking efficiencies at the end of stub processing step. The definition of which Tracking Particles are taken into account for this efficiency measurements are described here: SimTracker/TrackTriggerAssociation/python/StubAssociator_cfi.py in the PSet StubAssociator_params.TrackingParticle. The "maximal possible tracking efficiency" reported is derived assuming zero efficiency loss in subsequent steps. This method allows to assess which processing steps cause most inefficiency. Beside this end job summary Demonstrator_cfg.py produces Hist.root which contains histograms with more details like Stub position resolutions.


=== Configuration ===

All configuration params to manipulate the algorithms one may want to play with can be found in L1Trigger/TrackerDTC/python/Setup_cfi.py

=== Details to commonly used Classes ===

StubFE          stub with local coordinates before its processing in DTC

StubGL          stub transformed to global coordinates by DTC emulator

StubDTC         stub with final format send from DTC

Frame           is a typedef for std::bitset<64> representing the h/w words which level-1 track finder or level-1 trigger boards will receive and transmit per optical link and internal clock tick.

TTBV            Class representing a BitVector used by TrackTrigger emulators. Based on Frame. The class is mainly used to convert h/w-like structured bits into integers and vice versa. A typical constructors receive an integer values, a bit width (number of bits used to represent this value, an error will be thrown when not enough bits are provided.) and a boolean to distinguish between binary and two's complement representation. Multiple operators are provided, e.g bit wise or, or concatenation with another TTBV.

FrameStub       is a typedef for std::pair<TTStubRef, Frame>. This object is used to represent a Stub in TrackTrigger emulators. On the one hand side it connects to the original TTStub and on the other hand it has the bit string used in h/w to represent this stub.


StreamStub      h/w-like structured collection of stubs. Clock ticks where no Stub can be provided are represented by default constructed FrameStubs (edm:Ref recognising being a null ref and bit set being zero'd). This enables to store stubs bit and clock accurately.

Setup           ESProduct providing run time constants configuring Track Trigger emulators

SensorModule    represents an outer tracker sensormodule conataining it's parameters as well as bend and layer encoding
