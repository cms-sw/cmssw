# JetHT validation

## Validation analysis - JetHT_cfg.py

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
dataset | Single test file | A file list containing all analysis files. For CRAB running, you can also specify a CMS dataset in the form: "/JetHT/Run2018A-TkAlMinBias-12Nov2019_UL2018-v2/ALCARECO". This will not work for condor jobs to encourage CRAB usage for large datasets.
filesPerJob | 5 | Number of files per job when running via CRAB or condor
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

## File merging - addHistograms.sh

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

## Plotting - jetHtPlotter

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
markerSize  | 1  | Marker size used with this alignment
copyErrorColor | false | true: Use marker color for statistical error bars. false: Draw statistical error bars black.

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
drawReferenceProfile | false | Draw the profile from all runs as a reference to single IOV profile plots
drawCentralEtaSummaryProfile | true | Draw the central eta curves to the all runs summary profile plots
legendShiftTotalX    | 0 | Shift the legend from the default position in x-direction for profile plots. Units are in fractions of the canvas size.
legendShiftTotalY    | 0 | Shift the legend from the default position in y-direction for profile plots. Units are in fractions of the canvas size.
legendShiftX0    | 0 | Shift only the leftmost part of multicolumn legends in x-direction for profile plots. Units are in fractions of the canvas size.
legendShiftX1    | 0 | Shift only the second column from the left in multicolumn legends in x-direction for profile plots. Units are in fractions of the canvas size.
legendShiftX2    | 0 | Shift only the third column from the left in multicolumn legends in x-direction for profile plots. Units are in fractions of the canvas size.
legendShiftY0    | 0 | Shift only the leftmost part of multicolumn legends in y-direction for profile plots. Units are in fractions of the canvas size.
legendShiftY1    | 0 | Shift only the second column from the left in multicolumn legends in y-direction for profile plots. Units are in fractions of the canvas size.
legendShiftY2    | 0 | Shift only the third column from the left in multicolumn legends in y-direction for profile plots. Units are in fractions of the canvas size.
legendTextSize   | 0.05 | Text size used in the legend.
legendTextFont   | 62 | Text font used in the legend.
nIovInOnePlot          | 1 | Number of successive IOV:s drawn in a single plot if profile plots are drawn for each IOV. If there are more then one aligment that are plotted in single canvas, this is automatically set to 1 regardless of user input.

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
legendShiftTotalX    | 0 | Shift the legend from the default position in x-direction for trend plots. Units are in fractions of the canvas size.
legendShiftTotalY    | 0 | Shift the legend from the default position in y-direction for trend plots. Units are in fractions of the canvas size.
legendTextSize   | 0.05 | Text size used in the legend.
legendTextFont   | 62 | Text font used in the legend.
drawYearLines    | false | Draw vertical lines to trend plot marking for example different years of data taking.
runsForLines     | [290543,314881] | List of run numbers to which the vertical lines in the trend plots are drawn.
yearLineColor | 1 | Color of the drawn vertical lines
yearLineWidth | 1 | Width of the drawn vertical lines
yearLineStyle | 1 | Style of the drawn vertical lines
useLuminosityAxis | true | true: For trend plots, make the width of the x-axis bin for each IOV proportional to the integrated luminosity within that IOV. false: Each IOV has the same bin width in x-axis.
skipRunsWithNoData | false | true: If an IOV defined in lumiPerIovFile does not have any data, do not draw empty space to the x-axis for this IOV. false: Draw empty space for IOV:s with no data.
drawTags | false | Draw manually defined tags to trend plots.
tagLabels | [["2016",0.105,0.855],["2017",0.305,0.855],["2018",0.563,0.855]] | Tags to draw to the trend plots. This is an array of arrays, where the inner arrays must be in format ["tagText", tagPositionX, tagPositionY].
tagTextSize | 0.05 | Text size for the drawn tags
tagTextFont | 42 | Text font for the drawn tags
canvasHeight | 400 | Height of the canvas in trend plots
canvasWidth | 1000 | Width of the canvas in trend plots
marginLeft | 0.08 | Left margin for the trend canvas
marginRight | 0.03 | Right margin for the trend canvas
marginTop | 0.06 | Top margin for the trend canvas
marginBottom | 0.15 | Bottom margin for the trend canvas
titleOffsetX | 1.1 | Offset of the x-axis title in trend plots
titleOffsetY | 0.55 | Offset of the y-axis title in trend plots
titleSizeX | 0.06 | Size of the x-axis title in trend plots
titleSizeY | 0.06 | Size of the y-axis title in trend plots
labelOffsetX | 0.01 | Offset of the x-axis label in trend plots
labelOffsetY | 0.007 | Offset of the y-axis label in trend plots
labelSizeX | 0.05 | Size of the x-axis label in trend plots
labelSizeY | 0.05 | Size of the y-axis label in trend plots

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
lumiPerIovFile         | "lumiPerRun_Run2.txt" | File containing integrated luminosity for each IOV.
iovListMode            | "run" | If "IOV", the plot legends are drawn as IOV NNN-NNN. Otherwise they are Run NNN.
legendTextForAllRuns   | "All" | String referring to all runs included in the file. If you want better description than "all", you can set this to for example "Run2018A" or similar.
drawPlotsForEachIOV    | false | true: Draw the dxy and dz histograms and profiles separately for each IOV defined in the lumiPerIovFile. false: Only draw plots integrated over all IOVs.
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
