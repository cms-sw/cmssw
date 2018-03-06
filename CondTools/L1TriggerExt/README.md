L1T O2O code organization
-------------------------

This README briefly covers some "bolts and nuts" of the L1T O2O system.
For more general information on the system organization check [this link](https://github.com/kkotov/l1o2o).

The L1T O2O system is partitioned into the core framework (this package)
and a set of system-specific [online producers](https://github.com/cms-sw/cmssw/blob/master/L1TriggerConfig/L1TConfigProducers)
invoked by means of [data writers](https://github.com/cms-sw/cmssw/blob/master/CondTools/L1TriggerExt/src/DataWriterExt.cc)
from the core framework and fetching the information from the online DB. The only component of the core framework that
explicitly queries online DB is
[L1SubsystemKeysOnlineProdExt](https://github.com/cms-sw/cmssw/blob/master/CondTools/L1TriggerExt/plugins/L1SubsystemKeysOnlineProdExt.cc)
generating a L1TriggerKeyExt object with TSC and RS keys for all of the systems. This object is distributed to
system-specific ObjectKey\_online\_producers that in turn generate system-specific L1TriggerKeyExt objects forwarded
to the payload online\_producers. The CondDB tag names are specified in
[this config](https://github.com/cms-sw/cmssw/blob/master/CondTools/L1TriggerExt/python/L1SubsystemParamsExt_cfi.py),
while the current set of versions (suffixes) is given
[here](https://github.com/cms-sw/cmssw/blob/master/CondTools/L1TriggerExt/python/L1O2OTagsExt_cfi.py).
The versions can also be overridden from the top-level scripts as described in the bottom of
[this slide](http://kkotov.github.io/l1o2o/talks/2017.03.01/#4).
The core framework's general design is outlined in [this talk](http://kkotov.github.io/l1o2o/talks/2016.04.19).

Currently, the RS specific code is still available in the core framework,
but is not used or intended to be used. So if you browse the code, you can
ignore files containing the RS in the name. Same applies to all but
[runL1-O2O-iov.sh](https://github.com/cms-sw/cmssw/blob/master/CondTools/L1TriggerExt/scripts/runL1-O2O-iov.sh)
scripts.

