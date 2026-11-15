=== Compile & Run Instructions ===

```
cmsrel CMSSW_20_1_0_pre4
cd CMSSW_20_1_0_pre4/src
cmsenv
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_20_1_0_pre4
# Need these as they depend on the changed TrackerDetToDTCELinkCablingMap.h class
git cms-addpkg CondFormats/DataRecord
git cms-addpkg CondCore/SiPhase2TrackerPlugins
scram b -j
cd EventFilter/Phase2PixelRawToDigi/test/
# Run IT Qcore --> RAW for Inner Tracker.
cmsRun PixelDigiToRawProducer_cfg.py
```

