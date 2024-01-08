## MTS (Muon Track Splitting) validation

### General info

```
validations:
    MTS:
        <step_type>:
            <step_name>:
                <options>
```

MTS validation runs in 1 possible type of steps:
 - single (validation analysis by MTS_cfg.py)
Step name is arbitrary string which will be used as a reference for consequent steps.
Merge and trend jobs are not yet implemented.

### Single MTS jobs

Single jobs can be specified per run (IoV as well).

**Parameters below to be updated**
Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
IOV | None | List of IOVs/runs defined by integer value. IOV 1 is reserved for MC.
Alignments | None | List of alignments. Will create separate directory for each.
dataset | See defaultInputFiles_cff.py | Path to txt file containing list of datasets to be used. If file is missing at EOS or is corrupted - job will eventually fail (most common issue).
goodlumi | cms.untracked.VLuminosityBlockRange() | Path to json file containing lumi information about selected IoV - must contain list of runs under particular IoV with lumiblock info. Format: `IOV_Vali_{}.json`
maxevents | 1 | Maximum number of events before cmsRun terminates.
trackcollection | "generalTracks" | Track collection to be specified here, e.g. "ALCARECOTkAlMuonIsolated" or "ALCARECOTkAlMinBias" ...
tthrbuilder | "WithAngleAndTemplate" | Specify TTRH Builder
usePixelQualityFlag | True | Use pixel quality flag?
cosmicsZeroTesla | False | Is this validation for cosmics with zero magnetic field?
