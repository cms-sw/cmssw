The Python configuration `testCSCDigi2Raw_cfg.py` contains options to include the CSC packer, unpacker, reconstruction and/or validation modules on real or simulated data. Presently it is configured for Run-3.

Other options:
pack - enable packing
unpack - enable unpacking
view - enable digi view dumping

usePreTriggers - enable/disable use of preTriggers from sim for packing (enabled by default). Disable for real data repacking (will set packEverything option)
useGEMs - enable/disable Run3 GEM data packing (disabled by default)
useCSCShowers - enable/disable Run3 CSC Shower HMT objects packing

Example how to run it:
1) run WF 11650.0
2) `cmsRun EventFilter/CSCRawToDigi/test/testCSCDigi2Raw_cfg.py mc=True reconstruct=True validate=True inputFiles=file:step2.root`
