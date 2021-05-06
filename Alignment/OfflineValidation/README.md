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

## JetHT validation

### Validation analysis - JetHT_cfg.py

The vast majority of the time in the JetHT validation goes to running the actual validation analysis. This is done with cmsRun using the JetHT_cfg.py configuration file. To configure the analysis in the all-in-one framework, the following parameters can be set in the json/yaml configuration file:

```
All-in-one:

validations
    JetHT
        single
            myDataset
```

Variable | Default value | Explanation
-------- | ------------- | -----------
dataset | Single test file | A file list containing all analysis files.
alignments | None | An array of alignment sets for which the validation is run.
trackCollection     | "ALCARECOTkAlMinBias" | Track collection used for the analysis.
maxevents     | 1 | Maximum number of events before cmsRun terminates.
iovListFile  | "nothing" | File containing a list of IOVs one run boundary each line.
iovListList     | [0,500000] | If iovListFile is not defined, the IOV run boundaries are read from this array.
triggerFilter    | "nothing" | HLT trigger filter used to filter the events selected for the analysis.
printTriggers    | false | Print all available triggers to console. Set true only for local tests.
mc | false | Flag for Monte Carlo. Use true for MC and false for data.
profilePtBorders | [3,5,10,20,50,100] | List for pT borders used in wide pT bin profiles. Also the trend plots are calculated using these pT borders. If changed from default, the variable widePtBinBorders for the jetHtPlotter needs to be changed accordingly to get legends for trend plots correctly and avoid segmentation violations. This is done automatically by the all-in-one configuration.
TrackerAlignmentRcdFile | "nothing" | Local database file from which the TrackerAlignmentRcd is read. Notice that usual method to set this is reading from the database for each alignment.
TrackerAlignmentErrorFile | "nothing" | Local database file from which the TrackerAlignmentExtendedErrorRcd is read. Notice that usual method to set this is reading from the database for each alignment.

### File merging - addHistograms.sh

The addHistograms.sh script is used to merge the root files for jetHT validation. Merging is fast and can easily be done locally in seconds, but the tool is fully integrated to the all-in-one configuration for automated processing.

The instructions for standalone usage are:

```
addHistograms.sh [doLocal] [inputFolderName] [outputFolderName] [baseName]
doLocal = True: Merge local files. False: Merge files on CERN EOS
inputFolderName = Name of the folder where the root files are
outputFolderName = Name of the folder to which the output is transferred
baseName = Name given for the output file without .root extension
```

In use with all-in-one configuration, the following parameters can be set in the json/yaml configuration file:

```
All-in-one:

validations
    JetHT
        merge
            myDataset
```

Any number of datasets can be defined.

Variable | Default value | Explanation
-------- | ------------- | -----------
singles    | None | An array of single job names that must be finished before plotting can be run.
alignments | None | An array of alignment names for which the files are merged within those alignments. Different alignments are kept separate.


### Plotting - jetHtPlotter

The tool is originally designed to be used standalone, since the plotting the histograms locally does not take more that tens of second at maximum. But the plotter works also together with the all-in-one configuration. The only difference for user is the structure of the configuration file, that changes a bit between standalone and all-in-one usage.

Below are listed all the variables that can be configured in the json file for the jetHtPlotter macro. If the value is not given in the configuration, the default value is used instead.

```
Standalone:              All-in-one:

jethtplot                alignments
    alignments               myAlignment
        myAlignment
```

Up to four different alignments can be added for plotting. If more than four alignments are added, only first four are plotted.

Variable | Default value | Explanation
-------- | ------------- | -----------
inputFile   |  None | File containing the jetHT validation histograms for myAlignment. All-in-one config automatically uses default file name for merge job. Must be given if using plotter standalone.
legendText  | "AlignmentN" | Name with which the alignment is referred to in the legend of the drawn figures. For all-in-one configuration, this variable is called "title" instead of "legendText".
color       | Varies | Marker color used with this alignment
style       | 20 | Marker style used with this alignment

```
Standalone:               All-in-one:

jethtplot                 validations
    drawHistograms            JetHT
                                  plot
                                      myDataset
                                          jethtplot
                                              drawHistograms      
```

Select histograms to be drawn for each alignment.

Variable | Default value | Explanation
-------- | ------------- | -----------
drawDz       | false |  Draw probe track dz distributions
drawDzError  | false |  Draw probe track dz error distributions
drawDxy      | false |  Draw probe track dxy distributions
drawDxyError | false |  Draw probe track dxy error distributions

```
Standalone:               All-in-one:

jethtplot                 validations
    drawProfiles              JetHT
                                  plot
                                      myDataset
                                          jethtplot
                                              drawProfiles      
```

Select profile histograms to be drawn for each alignment.

Variable | Default value | Explanation
-------- | ------------- | -----------
drawDzErrorVsPt      | false | Draw dz error profiles as a function of pT
drawDzErrorVsPhi     | false | Draw dz error profiles as a function of phi
drawDzErrorVsEta     | false | Draw dz error profiles as a function of eta
drawDzErrorVsPtWide  | false | Draw dz error profiles as a function of pT in wide pT bins
drawDxyErrorVsPt     | false | Draw dxy error profiles as a function of pT
drawDxyErrorVsPhi    | false | Draw dxy error profiles as a function of phi
drawDxyErrorVsEta    | false | Draw dxy error profiles as a function of eta
drawDxyErrorVsPtWide | false | Draw dxy error profiles as a function of pT in wide pT bins

```
Standalone:               All-in-one:

jethtplot                 validations
    profileZoom               JetHT
                                  plot
                                      myDataset
                                          jethtplot
                                              profileZoom     
```

Axis zooms for profile histograms.

Variable | Default value | Explanation
-------- | ------------- | -----------
minZoomPtProfileDz       | 28 | Minimum y-axis zoom value for dz error profiles as a function of pT
maxZoomPtProfileDz       | 60 | Maximum y-axis zoom value for dz error profiles as a function of pT
minZoomPhiProfileDz      | 45 | Minimum y-axis zoom value for dz error profiles as a function of phi
maxZoomPhiProfileDz      | 80 | Maximum y-axis zoom value for dz error profiles as a function of phi
minZoomEtaProfileDz      | 30 | Minimum y-axis zoom value for dz error profiles as a function of eta
maxZoomEtaProfileDz      | 95 | Maximum y-axis zoom value for dz error profiles as a function of eta
minZoomPtWideProfileDz   | 25 | Minimum y-axis zoom value for dz error profiles as a function of pT in wide pT bins
maxZoomPtWideProfileDz   | 90 | Maximum y-axis zoom value for dz error profiles as a function of pT in wide oT bins
minZoomPtProfileDxy      | 7 | Minimum y-axis zoom value for dxy error profiles as a function of pT
maxZoomPtProfileDxy      | 40 | Maximum y-axis zoom value for dxy error profiles as a function of pT
minZoomPhiProfileDxy     | 40 | Minimum y-axis zoom value for dxy error profiles as a function of phi
maxZoomPhiProfileDxy     | 70 | Maximum y-axis zoom value for dxy error profiles as a function of phi
minZoomEtaProfileDxy     | 20 | Minimum y-axis zoom value for dxy error profiles as a function of eta
maxZoomEtaProfileDxy     | 90 | Maximum y-axis zoom value for dxy error profiles as a function of eta
minZoomPtWideProfileDxy  | 20 | Minimum y-axis zoom value for dxy error profiles as a function of pT in wide pT bins
maxZoomPtWideProfileDxy  | 80 | Maximum y-axis zoom value for dxy error profiles as a function of pT in wide pT bins

```
Standalone:               All-in-one:

jethtplot                 validations
    drawTrends                JetHT
                                  plot
                                      myDataset
                                          jethtplot
                                              drawTrends     
```

Select trend histograms to be drawn for each alignment.

Variable | Default value | Explanation
-------- | ------------- | -----------
drawDzError  | false | Draw the trend plots for dz errors
drawDxyError | false | Draw the trend plots for dxy errors

```
Standalone:               All-in-one:

jethtplot                 validations
    trendZoom                 JetHT
                                  plot
                                      myDataset
                                          jethtplot
                                              trendZoom     
```

Axis zooms for trend histograms.

Variable | Default value | Explanation
-------- | ------------- | -----------
minZoomDzTrend   | 20 | Minimum y-axis zoom value for dz error trends
maxZoomDzTrend   | 95 | Maximum y-axis zoom value for dz error trends
minZoomDxyTrend  | 10 | Minimum y-axis zoom value for dxy error trends
maxZoomDxyTrend  | 90 | Maximum y-axis zoom value for dxy error trends

```
Standalone:               All-in-one:

jethtplot                 validations
                              JetHT
                                  plot
                                      myDataset
                                          jethtplot    
```

All the remaining parameters

Variable | Default value | Explanation
-------- | ------------- | -----------
drawTrackQA            | false | Draw the track QA plots (number of vertices, tracks per vertex, track pT, track phi, and track eta).
drawYearLines          | false | Draw vertical lines to trend plot marking for example different years of data taking.
runsForLines           | [0] | List of run numbers to which the vertical draws in the trend plots are drawn.
lumiPerIovFile         | "lumiPerRun_Run2.txt" | File containing integrated luminosity for each IOV.
drawPlotsForEachIOV    | false | true: For profile plots, draw the profiles separately for each IOV defined in the lumiPerIovFile. false: Only draw plots integrated over all IOVs.
nIovInOnePlot          | 1 | Number of successive IOV:s drawn in a single plot is profile plots are drawn for each IOV.
useLuminosityForTrends | true | true: For trend plots, make the width of the x-axis bin for each IOV proportional to the integrated luminosity within that IOV. false: Each IOV has the same bin width in x-axis.
skipRunsWithNoData     | false | true: If an IOV defined in lumiPerIovFile does not have any data, do not draw empty space to the x-axis for this IOV. false: Draw empty space for IOV:s with no data.
widePtBinBorders       | [3,5,10,20,50,100] | List for pT borders used in wide pT bin profiles. Also the trend plots are calculated using these pT borders. This needs to be set to same value as the profilePtBorders variable used in the corresponding validation analysis. The all-in-one config does this automatically for you.
normalizeQAplots       | true | true: For track QA plots, normalize each distribution with its integral. false: No normalization for QA plots, show directly the counts.
makeIovlistForSlides   | false | true: Create a text file to be used as input for prepareSlides.sh script for making latex presentation template with profile plots from each IOV. false: Do not do this.
iovListForSlides       | "iovListForSlides.txt" | Name given to the above list.
saveComment            | "" | String of text added to all saved figures.

```
All-in-one:

validations
    JetHT
        plot
            myDataset
```

Parameters only used in all-in-one tool. Any number of datasets can be defined.

Variable | Default value | Explanation
-------- | ------------- | -----------
merges     | None | An array of merge job names that must be finished before plotting can be run.
alignments | None | An array of alignment names that will be plotted.

