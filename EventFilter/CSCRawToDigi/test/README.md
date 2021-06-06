The Python configuration `testCSCDigi2Raw_cfg.py` contains options to include the CSC packer, unpacker, reconstruction and/or validation modules on real or simulated data. Presently it is configured for Run-3.

Example how to run it:
1) run WF 11650.0
2) `cmsRun EventFilter/CSCRawToDigi/test/testCSCDigi2Raw_cfg.py mc=True reconstruct=True validate=True inputFiles=file:step2.root`
