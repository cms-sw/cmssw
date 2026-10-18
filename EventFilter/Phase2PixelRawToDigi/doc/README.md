=== Compile & Run Instructions ===

NOTE : There is no compatible cabling map that can be used in CMSSW_20_1_0_pre4 for now. Cabling map in EventFilter/Phase2PixelRawToDigi/test/cablingmpa/OTandITDTCCablingMap.db is modified to make it manually work in CMSSW_20_1_0_pre4.

```
cmsrel CMSSW_20_1_0_pre4
cd CMSSW_20_1_0_pre4/src
cmsenv
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_CMSSW_20_1_0_pre4
scram b -j
cd EventFilter/Phase2PixelRawToDigi/test/
# Run IT Qcore --> RAW for Inner Tracker.
cmsRun PixelDigiToRawProducer_cfg.py
```

