CSC Trigger Primitives: Test Modules
===================================

runCSCTriggerPrimitiveProducer
------------------------------

Configuration to run the CSC trigger primitive producer. Option available to unpack RAW data - useful for data vs emulator comparisons.


runCSCL1TDQM
------------

Configuration to run the module `l1tdeCSCTPG` on a data file. Will produce a DQM file onto which the runCSCTriggerPrimitiveAnalyzer can be run


runCSCTriggerPrimitiveAnalyzer
------------------------------

Configuration to run analysis on CSC trigger primitives. Choose from options to analyze data vs emulator comparison (`l1tdeCSCTPGClient`), MC resolution or MC efficiency.

For data vs emulator comparison, first run `DQM/Integration/python/clients/l1tstage2emulator_dqm_sourceclient-live_cfg.py` to obtain the DQM file, then use it as input for the comparison. Alternatively, run runCSCL1TDQM.


runL1CSCTPEmulatorConfigAnalyzer
--------------------------------

Compare configuration from DB with Python for CSC trigger primitives. Typically not necessary to do; all configuration is by default loaded from Python, not the configuration DB.


runGEMCSCLUTAnalyzer
--------------------

Makes the lookup tables for the GEM-CSC integrated local trigger in simulation and firmware. Current lookup tables can be found in https://github.com/cms-data/L1Trigger-CSCTriggerPrimitives/GEMCSC
