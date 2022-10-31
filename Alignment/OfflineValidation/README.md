# Validation

We use the Boost library (Program Options, Filesystem & Property Trees) to deal with the treatment of the config file.
Basic idea:
 - a generic config file is "projected" for each validation (*e.g.* the geometry is changed, together with the plotting style);
 - for each config file, a new condor config file is produced;
 - a DAGMAN file is also produced in order to submit the whole validation at once.

In principle, the `validateAlignments.py` command is enough to submit everything.
However, for local testing, one may want to make a dry run: all files will be produced, but the condor jobs will not be submitted;
then one can just test locally any step, or modify any parameter before simply submitting the DAGMAN.

## HOWTO use

The main script is `validateAlignments.py`. One can check the options with:
```
validateAlignments.py -h
usage: validateAlignments.py [-h] [-d] [-v] [-e] [-f]
                             [-j {espresso,microcentury,longlunch,workday,tomorrow,testmatch,nextweek}]
                             config

AllInOneTool for validation of the tracker alignment

positional arguments:
  config                Global AllInOneTool config (json/yaml format)

optional arguments:
  -h, --help            show this help message and exit
  -d, --dry             Set up everything, but don't run anything
  -v, --verbose         Enable standard output stream
  -e, --example         Print example of config in JSON format
  -f, --force           Force creation of enviroment, possible overwritten old configuration
  -j {espresso,microcentury,longlunch,workday,tomorrow,testmatch,nextweek}, --job-flavour {espresso,microcentury,longlunch,workday,tomorrow,testmatch,nextweek}
                        Job flavours for HTCondor at CERN, default is 'longlunch'
```

As input the AllInOneTool config in `yaml` or `json` file format has to be provided. One proper example can be find here: `Alignment/OfflineValidation/test/test.yaml`. To create the set up and submit everything to the HTCondor batch system, one can call

```
validateAlignments.py $CMSSW_BASE/src/Alignment/OfflineValidation/test/test.yaml 

-----------------------------------------------------------------------
File for submitting this DAG to HTCondor           : /afs/cern.ch/user/d/dbrunner/ToolDev/CMSSW_10_6_0/src/MyTest/DAG/dagFile.condor.sub
Log of DAGMan debugging messages                 : /afs/cern.ch/user/d/dbrunner/ToolDev/CMSSW_10_6_0/src/MyTest/DAG/dagFile.dagman.out
Log of HTCondor library output                     : /afs/cern.ch/user/d/dbrunner/ToolDev/CMSSW_10_6_0/src/MyTest/DAG/dagFile.lib.out
Log of HTCondor library error messages             : /afs/cern.ch/user/d/dbrunner/ToolDev/CMSSW_10_6_0/src/MyTest/DAG/dagFile.lib.err
Log of the life of condor_dagman itself          : /afs/cern.ch/user/d/dbrunner/ToolDev/CMSSW_10_6_0/src/MyTest/DAG/dagFile.dagman.log

Submitting job(s).
1 job(s) submitted to cluster 5140155.
-----------------------------------------------------------------------
```

To create the set up without submitting jobs to HTCondor one can use the dry run option:

```
validateAlignments.py $CMSSW_BASE/src/Alignment/OfflineValidation/test/test.yaml -d
Enviroment is set up. If you want to submit everything, call 'condor_submit_dag /afs/cern.ch/user/d/dbrunner/ToolDev/CMSSW_10_6_0/src/MyTest/DAG/dagFile'
```

## TODO list 

 - improve exceptions handling (filesystem + own)
   - check inconsistencies in config file?
 - from DMR toy to real application
   - GCP (get "n-tuples" + grid, 3D, TkMaps)
   - DMRs (single + merge + trend)
   - PV (single + merge + trend)
   - Zµµ (single + merge)
   - MTS (single + merge)
   - overlap (single + merge + trend)
   - ...
 - documentation (this README)
   - tutorial
   - instructions for developers
 - details
   - copy condor config like the executable (or similar) and use soft links instead of hard copy
   - make dry and local options (i.e. just don't run any condor command)
(list from mid-January)

