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

## HOWTO implement

To implement a new/or porting an existing validation to the new frame work, two things needs to be provided: executables and a python file providing the information for each job.

#### Executables

In the new frame work standalone executables do the job of the validations. They are designed to run indenpendently from the set up of validateAlignments.py, the executables only need a configuration file with information needed for the validation/plotting. One can implement a C++ or a python executable. 

If a C++ executable is implemented, the source file of the executable needs to be placed in the` Alignment/OfflineValidation/bin` directory and the BuildFile.xml in this directory needs to be modified. For the readout of the configuration file, which is in JSON format, the property tree class from the boost library is used. See `bin/DMRmerge.cc as` an example of a proper C++ implementation.

If a python executable is implemented, the source file needs to be placed in the `Alignment/OfflineValidation/scripts` directory. In the first line of the python script a shebang like `#!/usr/bin/env python` must be written and the script itself must be changed to be executable. In the case of python the configuration file can be both in JSON/YAML, because in python both after read in are just python dictionaries. See `Example of Senne when he finished it` as an example of a proper python implementation.

For the special case of a cmsRun job, one needs to provide only the CMS python configuration. Because it is python again, both JSON/YAML for the configuration file are fine to use. Also for this case the execution via cmsRun is independent from the set up provided by validateAligments.py and only need the proper configuration file. See `python/TkAlAllInOneTool/DMR_cfg.py` as an example of a proper implementation.

#### Python file for configuration

For each validation several jobs can be executed, because there are several steps like nTupling, fitting, plotting or there is categorization like alignments, IOVs. The information will be encoded in a global config provided by the aligner, see `Alignment/OfflineValidation/test/test.yaml` as an example. To figure out from the global config which/how many jobs should be prepared, a python file needs to be implemented which reads the global config, extract the relevant information of the global config and yields smaller config designed to be read from the respective executable. As an example see `python/TkAlAllInOneTool/DMR.py`.

There is a logic which needed to be followed. Each job needs to be directionary with a structure like this:

```
job = {
       "name": Job name ##Needs to be unique!
       "dir": workingDirectory  ##Also needs to be unique!
       "exe": Name of executable/or cmsRun
       "cms-config": path to CMS config if exe = cmsRun, else leave this out
       "dependencies": [name of jobs this jobs needs to wait for] ##Empty list [] if no depedencies
       "config": Slimmed config from global config only with information needed for this job
}
```

The python file returns a list of jobs to the `validateAligments.py` which finally creates the directory structure/configuration files/DAG file. To let` validateAligments.py` know one validation implementation exist, import the respective python file and extend the if statements which starts at line 69. This is the only time one needs to touch `validateAligments.py`!
 

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

