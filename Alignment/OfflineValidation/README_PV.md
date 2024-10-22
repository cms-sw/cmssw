# Primary Vertex (PV) Validation

## General info
```
validations:
    PV:
        <step_type>:
            <step_name>: 
                <options>
```

PV validation runs in 3 possible types of steps:
 - single (validation analysis by PV_cfg.py)
 - (optional) merge (PVmerge macro)
 - (optional) trends (PVtrends macro) 
Step name is arbitrary string which will be used as a reference for consequent steps.
Merge job and trend jobs will only start if all corresponding single jobs are done.
Trend and merge jobs can run in parallel.
Averaged jobs are not yet implemented. 

## Single PV jobs
Single jobs can be specified per run (IoV as well). In case of MC, IoV is specified to arbitrary 1.  

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

## Merge PV job
Its name do not need to match single job name but option `singles` must list all single jobs to be merged.
PV merged plot style can be adjusted from global plotting style (see `Alignment/OfflineValidation/test/example_PV_full.yaml`)

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
singles | None | List of strings matching single job names to be merged in one plot.
doMaps | false | Also plot 2D maps.
stdResiduals | true | Plot std. residual plots.
autoLimits | false | Take default values for limits in plotting?
m_dxyPhiMax | 40 | If not take this...
m_dzPhiMax | 40 | If not take this...
m_dxyEtaMax | 40 | If not take this...
m_dzEtaMax | 40 | If not take this...
m_dxyPhiNormMax | 0.5 | If not take this...
m_dzPhiNormMax | 0.5 | If not take this...
m_dxyEtaNormMax | 0.5 | If not take this...
m_dzEtaNormMax | 0.5 | If not take this...
w_dxyPhiMax | 150 | If not take this...
w_dzPhiMax | 150 | If not take this... 
w_dxyEtaMax | 150 | If not take this...
w_dzEtaMax | 1000 | If not take this...
w_dxyPhiNormMax | 1.8 | If not take this... 
w_dzPhiNormMax |  1.8 | If not take this...
w_dxyEtaNormMax | 1.8 | If not take this...
w_dzEtaNormMax | 1.8 | If not take this...
customrighttitle | "" | Top right title. Reserved word "IOV" will be replaced for given IOV/run in the list.

## Trends PV job
Its name do not need to match single job name but option `singles` must list all single jobs to be put in trend plot.
Trend plot style is defined globally for all trend plots (see `Alignment/OfflineValidation/test/example_PV_full.yaml`)

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
singles | None | List of single job names to be processed for trends. 
doRMS | true | Plot RMS trends.
labels | None | List of string tags to be added in output rootfile.
firstRun | 272930 | Specify starting run to be plotted.
lastRun | 325175 | Specify the last run to be considered.
nWorkers | 20 | Number of threads.
doUnitTest | false | Disable certain settings for unit test.
