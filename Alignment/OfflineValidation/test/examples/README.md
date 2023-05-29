# Examples

## Full JetHT analysis example using semi-automatic CRAB implementation

There is a possibility to run the analysis using CRAB. Currently the implementation for this is semi-automatic, meaning that the All-In-One tool provides you all the necessary configuration, but you will need to manually run the jobs and make sure all the jobs are finished successfully.

Before starting, make sure your have voms proxy available:

```
voms-proxy-init --voms cms
```

To begin, create the configuration using the All-In-One tool. It is important that you will do a dry-run:

```
validateAlignments.py -d jetHtAnalysis_fullExampleConfiguration.json
```

Move to the created directory with the configuration files:

```
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/example_json_jetHT/JetHT/single/fullExample/prompt
```

Check that you have write access to the default output directory used by the CRAB. By default the shared EOS space of the tracker alignment group at CERN is used.

```
crab checkwrite --site=T2_CH_CERN --lfn=/store/group/alca_trackeralign/`whoami`
```

At this point, you should also check the running time and memory usage. Running over one file can take up to 2 hours and you will need about 1000 MB of RAM for that. So you should set the corresponding variables to values slightly above these.

```
vim crabConfiguration.py
...
config.Data.outLFNDirBase = '/store/group/alca_trackeralign/username/' + config.General.requestName
...
config.JobType.maxMemoryMB = 1200
config.JobType.maxJobRuntimeMin = 200
...
config.Data.unitsPerJob = 1
...
config.Site.storageSite = 'T2_CH_CERN'
```

After checking the configuration, submit the jobs.

```
crab submit -c crabConfiguration.py
```

Do the same for ReReco and UltraLegacy folders:

```
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/example_json_jetHT/JetHT/single/fullExample/rereco
crab submit -c crabConfiguration.py
...
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/example_json_jetHT/JetHT/single/fullExample/ultralegacy
crab submit -c crabConfiguration.py
```

Now you need to wait for the crab jobs to finish. It should take around two hours times the value you set for unitsPerJob variable for these example runs. After the jobs are finished, you will need to merge the output files and transfer the merged files to the correct output folders. One way to do this is as follows:

```
cd $CMSSW_BASE/src/JetHtExample/example_json_jetHT/JetHT/merge/fullExample/prompt
hadd -ff JetHTAnalysis_merged.root `xrdfs root://eoscms.cern.ch ls -u /store/group/alca_trackeralign/username/path/to/prompt/files | grep '\.root'`
cd $CMSSW_BASE/src/JetHtExample/example_json_jetHT/JetHT/merge/fullExample/rereco
hadd -ff JetHTAnalysis_merged.root `xrdfs root://eoscms.cern.ch ls -u /store/group/alca_trackeralign/username/path/to/rereco/files | grep '\.root'`
cd $CMSSW_BASE/src/JetHtExample/example_json_jetHT/JetHT/merge/fullExample/ultralegacy
hadd -ff JetHTAnalysis_merged.root `xrdfs root://eoscms.cern.ch ls -u /store/group/alca_trackeralign/username/path/to/ultralegacy/files | grep '\.root'`
```

For 100 files, the merging should be done between one and two minutes. Now that all the files are merged, the only thing that remains is to plot the results. To do this, navigate to the folder where the plotting configuration is located and run it:

```
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/example_json_jetHT/JetHT/plot/fullExample
jetHtPlotter validation.json
```

The final validation plots appear to the output folder. If you want to change the style of the plots or the histograms plotted, you can edit the validation.json file here and rerun the plotter. No need to redo the time-consuming analysis part.

## Full example using condor

The CRAB running is recommended for large datasets, but smaller tests can also be readily done with condor. There are two different modes for running with condor, and the mode is selected automatically based on the input file list. If using the same file list as for CRAB, file based job splitting method is used. However, there are some dangers in using file based splitting with maxevents parameter, namely that if not selected carefully, some of the files might be skipped altogether. This might lead to certain runs not being analyzed. Thus run number based job splitting is recommended to ensure that each run has the statistics defined in the maxevents parameter. To use run number based splitting, we need to include the run numbers that can be found from the input files in the file list.

There is an automatic procedure to do this. You can generate a file list with run numbers from a regular file list using the tool makeListRunsInFiles.py. To do this for the file list used in the CRAB example, you need to run the command

```
makeListRunsInFiles.py --input=jetHtFilesForRun2018A_first100files.txt --output=jetHtFilesForRun2018A_first100files_withRuns.txt
```

Since the information about runs needs to be read from a DAS database, it takes a while to execute this command. For this example with 100 files, the script should finish within two minutes. If it takes significantly longer than that, there might be some network issues. After generating the file list with run information included, change the dataset name in the json configuration file

```
vim jetHtAnalysis_fullExampleConfiguration.json
...
                    "dataset": "$CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/jetHtFilesForRun2018A_first100files_withRuns.txt",
```

Now that the run numbers are included with the files, validateAlignments.py script will automatically setup run number based splitting. The configuration has been setup to run 1000 events from each run. Notice that if you use the original setup, the validation will still work, but 1000 events from each file will be analyzed instead. You can run everything with the command:

```
validateAlignments.py -j espresso jetHtAnalysis_fullExampleConfiguration.json
```

Then you just wait for your jobs to get submitted, and soon afterwards the plots will appear in the folder

```
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/example_json_jetHT/JetHT/plot/fullExample/output
```

## jetHt_multiYearTrendPlot.json

This configuration shows how to plot multi-year trend plots using previously merged jetHT validation files. It uses the jetHT plotting macro standalone. You can run this example using

```
jetHtPlotter jetHt_multiYearTrendPlot.json
```

For example purposes, this configuration has a lot of variables redefined to their default values. It shows you available configuration options, even though they can be omitted if you do not want to change the default values.

## jetHt_ptHatWeightForMCPlot.json

This configuration shows how to apply ptHat weight for MC files produced with different ptHat cuts. What you need to do is to collect the file names and lower boundaries of the ptHat bins into a file, which is this case is ptHatFiles_MC2018_PFJet320.txt. For a file list like this, the ptHat weight is automatically applied by the code. The weights are correct for run2. The plotting can be done using the jetHT plotter standalone:

```
jetHtPlotter jetHt_ptHatWeightForMCPlot.json
```
