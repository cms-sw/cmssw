## DMR validation

#General info
```
validations:
    DMR:
        <step_type>:
            <step_name>: 
                <options>
```

DMR validation runs in consequent steps of 4 possible types:
 - single (validation analysis by DMR_cfg.py)
 - merge (DMRmerge macro)
 - (optional) trends (DMRtrends macro) 
 - (optional) averaged (mkLumiAveragedPlots.py script)
Step name is arbitrary string which will be used as a reference for consequent steps.
Merge job will only start if all corresponding single jobs are done.
Trends/Averaged job will start if all corresponding merge jobs are done.
Trends and averaged jobs will run in parallel.
Averaged job consists of 3 types of sub-jobs (submission is automatized internally). 

#Single DMR jobs:
Single jobs can be specified per run (IoV as well). In case of MC, IoV is specified to arbitrary 1.  

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
IOV | None | List of IOVs/runs defined by integer value. IOV 1 is reserved for MC.
Alignments | None | List of alignments. Will create separate directory for each.
dataset | See defaultInputFiles_cff.py | Path to txt file containing list of datasets to be used. If file is missing at EOS or is corrupted - job will eventually fail (most common issue).
goodlumi | cms.untracked.VLuminosityBlockRange() | Path to json file containing lumi information about selected IoV - must contain list of runs under particular IoV with lumiblock info. Format: `IOV_Vali_{}.json`
magneticfield | true | Is magnetic field ON? Not really needed for validation...
maxevents | 1 | Maximum number of events before cmsRun terminates.
maxtracks | 1 | Maximum number of tracks per event before next event is processed.
trackcollection | "generalTracks" | Track collection to be specified here, e.g. "ALCARECOTkAlMuonIsolated" or "ALCARECOTkAlMinBias" ... 

#Merge DMR job
Its name do not need to match single job name but option `singles` must list all single jobs to be merged.
Needs to be specified in order to run averaged/trends jobs.

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
singles | None | List of strings matching single job names to be merged in one plot.
methods | ["median","rmsNorm"] | List of types of plots to be produced. Available: median,mean,rms,meanNorm,rmsNorm + X/Y suffix optionally
curves  | ["plain"] | List of additional plot type otions. Available: plain,split,layers,layersSeparate,layersSplit,layersSplitSeparate
customrighttitle | "" | Top right title. (To be re-implemented)
legendheader | "" | Legend title.
usefit | false | Use gaussian function to fit distribution otherwise extract mean and rms directly from histogram. 
legendoptions | ["mean","rms"] | Distribution features to be displayed in stat box: mean,meanerror,rms,rmserror,modules,all 
minimum | 15 | Minimum number of hits requested.
bigtext | false | Legend text size should be enlarged.
moduleid | None | Plot residuals for selected list of module IDs. (debugging)

#Trends DMR job
Its name do not need to match merge neither single job name but option `merges` must list all merge jobs to be put in trend plot.
Trend plot style is defined globally for all trend plots (see `Alignment/OfflineValidation/test/example_DMR_full.yaml`)

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
merges | None | List of merge job names to be processed for trends. 
Variables | ["median"] | Trend plot type to be plotted: DrmsNR, median

#Averaged DMR job
Its name do not need to match merge neither single job name but option `merges` must list all merge jobs to be put in averaged distribution.
Each merge job to be passed to averager must consist of data OR MC single jobs exclusively (no mix of Data and MC). 
Some style options are accessible from global style config (see `Alignment/OfflineValidation/test/example_DMR_full.yaml`).
DISCLAIMER: this tool is not to be used blindly. Averaged distributions will only make sense if the same number of events and tracks is considered for each IOV.

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
merges | None | List of merge job names to be processed for averaged distributions.
lumiPerRun | None | List of lumi-per-run files. 
lumiPerIoV | None | List of lumi-per-iov files.
maxfiles | 700 | Maximum number of files to be merged per sub-job. 
lumiMC | None | Define scale factors to be used for normalisation of MC from the list of merge jobs. Format: `["(<igroup>::)<merge_name>::<scale_factor>"]`. `<igroup>` is optional integer in case of multiple MC groups to be merged.
