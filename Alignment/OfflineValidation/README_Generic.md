# Generic Validation

## General info
```
validations:
    Generic:
        <step_type>:
            <step_name>: 
                <options>
```

Generic validation runs in 2 possible types of steps:
 - single (validation analysis by GenericV_cfg.py)
 - (optional) merge (GenericVmerge macro)
Step name is and arbitrary string which will be used as a reference for consequent steps.
Merge job awill only start if all corresponding single jobs are done.
Merge jobs can run in parallel.

## Single Generic jobs
Single jobs can be specified per run (IoV as well). In case of MC, IoV is specified to arbitrary 1.  

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
IOV | None | List of IOVs/runs defined by integer value. IOV 1 is reserved for MC.
Alignments | None | List of alignments. Will create separate directory for each.
dataset | See defaultInputFiles_cff.py | Path to txt file containing list of datasets to be used. If file is missing at EOS or is corrupted - job will eventually fail (most common issue).
goodlumi | cms.untracked.VLuminosityBlockRange() | Path to json file containing lumi information about selected IoV - must contain list of runs under particular IoV with lumiblock info. Format: `IOV_Vali_{}.json`
maxevents | 1 | Maximum number of events before cmsRun terminates.
trackcollection | "generalTracks" | Track collection to be specified here, e.g. "ALCARECOTkAlMuonIsolated" or "ALCARECOTkAlMinBias" ... 
tthrbuilder | "WithAngleAndTemplate" | Specify TTRH Builder

## Merge Generic job
Its name do not need to match single job name but option `singles` must list all single jobs to be merged.
Generic merged plot style can be adjusted from global plotting style.

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
singles | None | List of strings matching single job names to be merged in one plot.
customrighttitle | "" | Top right title. Reserved word "IOV" will be replaced for given IOV/run in the list.
