# Test directory

## SiStripClusterizerConditionsSoA
SiStripClusterizerConditionsSoA is a portable multi collection (i..e, made of 4 SoA collections) storing the necessary data for the `RecoLocalTracker/SiStripClusterizer` alpaka-module.

The multi-collection can be inspected in `CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h`. It is made of a mapping between detector ID and FEDs (DetToFeds), a table with the inverse thickness and APV pair number (Data_fedch), a table with strip-indexed noise (Data_strip) and finally an apv-indexed table for gain (Data_apv). 

The unit-test populates the `SiStripClusterizerConditionsSoA` multicollection on host with a number of entries which are typical of the physics case, then copy on device and finally checks back on host there are no mismatches. The tests can be compiled and run with the commands:
```bash
scram b -j 16 runtests_SiStripClusterizerConditionsSoA_alpakaSerialSync
scram b -j 16 runtests_SiStripClusterizerConditionsSoA_alpakaCudaAsync
scram b -j 16 runtests_SiStripClusterizerConditionsSoA_alpakaROCmAsync
```

## SiStripMappingSoA
SiStripMappingSoA is the SoA generating the host/device `SiStripMappingHost`/`SiStripMappingDevice` portable collection.

This collection is an auxiliary data structure, introduced first in the development of the `RecoLocalTracker/SiStripClusterizer`. It is used to store a map of the conditions-passing pointers of FED raw data, to be processed by an host/device kernel for the actual FEDraw->Digi unpacking.

The unit-test creates an host collection, fills it, copies on device and back on host to compare it. Tests can be compiled and run with:

```bash
scram b -j 16 runtests_SiStripMappingSoA_alpakaSerialSync
scram b -j 16 runtests_SiStripMappingSoA_alpakaCudaAsync
scram b -j 16 runtests_SiStripMappingSoA_alpakaROCmAsync
```