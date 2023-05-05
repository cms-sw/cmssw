## PV validation

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

### Single PV jobs

Single jobs can be specified per run (IoV as well). In case of MC, IoV is specified to arbitrary 1.

**Parameters below to be updated**
Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
IOV | None | List of IOVs/runs defined by integer value. IOV 1 is reserved for MC.
Alignments | None | List of alignments. Will create separate directory for each.
dataset | See defaultInputFiles_cff.py | Path to txt file containing list of datasets to be used. If file is missing at EOS or is corrupted - job will eventually fail (most common issue).
goodlumi | cms.untracked.VLuminosityBlockRange() | Path to json file containing lumi information about selected IoV - must contain list of runs under particular IoV with lumiblock info. Format: `IOV_Vali_{}.json`
maxevents | 1 | Maximum number of events before cmsRun terminates.
maxtracks | 1 | Maximum number of tracks per event before next event is processed.
trackcollection | "generalTracks" | Track collection to be specified here, e.g. "ALCARECOTkAlMuonIsolated" or "ALCARECOTkAlMinBias" ...
tthrbuilder | "WithAngleAndTemplate" | Specify TTRH Builder
usePixelQualityFlag | True | Use pixel quality flag?
cosmicsZeroTesla | False | Is this validation for cosmics with zero magnetic field?
vertexcollection | "offlinePrimaryVertices" | Specify vertex collection to be used.
isda | True | Use DA algorithm (True) or GAP algorithm (False)
ismc | True | Is validation for MC (True) or Data (False)?
runboundary | 1 | Specify starting run number (can be also list of starting numbers in multirun approach).
runControl | False | Enable run control
ptCut | 3. | Probe tracks with pT > 3GeV
etaCut | 2.5 | Probe tracks in abs(eta) < 2.5 region
minPt | 1. | Define minimal track pT
maxPt | 30. | Define maximum track pT
doBPix | True | Do not run validation for BPix if needed
doFPix | True | Do not run validation for FPix if needed
forceBeamSpot | False | Force beam spot
numberOfBins | 48 | Define histogram granularity
