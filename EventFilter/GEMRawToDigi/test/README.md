CSC Trigger Primitives: Test Modules
====================================

AMC13 Spy for GEM-CSC test-stand at B904
----------------------------------------

<PRE>
cmsRun AMC13SpyReadout.py inputFiles=file:run000000_Testing_CERN904_2021-06-08_chunk_0.dat
</PRE>

GEM unpacker
------------

This module can be run on regular CMS data - it is integrated in the standard CMS sequence. It can also be run on data from the AMC13 Spy at the B904 GEM-CSC test-stand, using

<PRE>
cmsRun runGEMUnpacker_cfg.py inputFiles=file:output_raw.root useB904Data=True feds=1478
</PRE>
