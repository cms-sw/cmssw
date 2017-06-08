## Dumping XML configuration from the Online DB to local files

The whole trigger system's online configuration is aggregated by two top-level keys: *Trigger System Configuration*
(TSC) key and *Run Settings* (RS) key. These keys are prepared by the Level-1 Detector On Call (L1 DOC) shifter and
utilized every time when a new data taking is started. You can check the XML configuration for the specific TSC
and RS keys using the [L1 Configuration Editor](https://l1ce.cms) (assuming you are within .cms network or use a
tunnel) or alternatively, using following python scripts:
[ugtDump.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/ugtDump.py),
[ugmtDump.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/ugmtDump.py),
[caloDump.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/caloDump.py),
[emtfDump.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/emtfDump.py),
[omtfDump.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/omtfDump.py),
[bmtfDump.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/bmtfDump.py).
These scripts can be ran from my afs public area on lxplus as well as within the private .cms network from
~l1emulator/o2o/. For example:

lxplus> python ~kkotov/public/bmtfDump.py l1\_trg\_cosmics2017/v75 l1\_trg\_rs\_cosmics2017/v57

dumps the Barrel Muon Track-Finder trigger configuration for TSC\_KEY=l1\_trg\_cosmics2017/v75 and
RS\_KEY=l1\_trg\_rs\_cosmics2017/v57 into several local XML files. For arguments you can use both: top-level
TSC and RS keys (as in the example above) and system-specific TSC and RS keys
(it could have been bmtf\_cosmics\_2017/v4 bmtf\_rs\_base\_2017/v1 in the example above).

The L1T O2O framework manages a set of XML parsers (referred to as [Online Producers](https://github.com/cms-sw/cmssw/tree/master/L1TriggerConfig/L1TConfigProducers/src))
that can be run individually as, for example, shown in [runOneByOne.sh](https://github.com/cms-sw/cmssw/blob/master/L1TriggerConfig/Utilities/test/runOneByOne.sh)
script as well as in one go using the framework. In the first case you can run the script from lxplus or .cms:

lxplus> ~kkotov/python/runOneByOne.sh l1\_trg\_cosmics2017/v75 l1\_trg\_rs\_cosmics2017/v57

ssh cms-conddb-1.cms '/data/O2O/L1T/runOneByOne.sh l1\_trg\_cosmics2017/v75 l1\_trg\_rs\_cosmics2017/v57'

The result of running the script is a comprehensive printout the last two lines of which will summarize if
any problems were encountered parsing the configuration XMLs. In addition, an l1config.db sqlite file will
contain all of the successfully produced payloads ready to be used with the L1 trigger emulators in CMSSW.

## Dumping conditions from the Offline (Conditions) DB

Another set of scripts allows you to print fields of the payloads in Cond DB (production and development), local
sqlite file, or static configuration python in the release (if applies). Diffing the results is a key use case.
These CMSSW scripts are:
[viewMenu.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewMenu.py),
[viewCaloParams.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewCaloParams.py),
[viewOverPar.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewOverPar.py),
[viewECpar.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewECpar.py),
[viewL1TGlobalPrescalesVetos.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewL1TGlobalPrescalesVetos.py),
[viewTKE.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewTKE.py),
[viewTKLE.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/viewTKLE.py)

You can run them from lxplus (but not from .cms):

lxplus> cmsRun viewCaloParams.py db=prod run=1000000

## For experts: uploading prototype payloads in Cond DB

The following set of script allows to update the prototype (starting point that L1T O2O take and updates with parameters
extracted from the online XMLs):
[uploadBmtfParams.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/uploadBmtfParams.py),
[uploadEmtfParams.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/uploadEmtfParams.py),
[uploadCaloParams.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/uploadCaloParams.py),
[uploadGmtParams.py](https://github.com/kkotov/cmssw/blob/o2oUtilities92X/L1TriggerConfig/Utilities/test/uploadGmtParams.py)

## For experts: standalone XML parsers

The read*.cc standalones (compilation instructions inside) are able to read the online xml config files. Although I do
not commit to support them up to the date.
