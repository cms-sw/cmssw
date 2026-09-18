=== Compile & Run Instructions ===

NOTE : There is no compatible cabling map that can be used in CMSSW_16_0_0_pre1 for now. Cabling map in EventFilter/Phase2PixelRawToDigi/test/cablingmpa/OTandITDTCCablingMap.db is modified to make it manually work in CMSSW_16_0_0_pre1.

```
cmsrel CMSSW_16_0_0_pre1
cd CMSSW_16_0_0_pre1/src
cmsenv
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_CMSSW_16_0_0_pre1
scram b -j
cd EventFilter/Phase2PixelRawToDigi/test/
# Run IT Qcore --> RAW for Inner Tracker.
cmsRun PixelDigiToRawProducer_cfg.py
```

