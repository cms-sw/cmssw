The following document summarises usage of IOV/run-driven validations.

## DMR and PV

Example A:
```
validations:
    DMR:
        single:
            TestSingleMC:
                IOV:
                    - 1
                dataset: /path/to/dataset.txt
                ...
            TestSingleDataIOV:
                IOV:
                    - 315257
                    - 315488
                    - 315489
                dataset: /path/to/dataset_IOV_{}.txt
                goodlumi: /path/to/IOV_Vali_{}.json
            TestSingleDataRun:
                IOV:
                    - 315257
                    - 315258
                    - 315259
                    ...
                    - 315488
                dataset: /path/to/dataset_Run_{}.txt
                goodlumi: /path/to/Run_Vali_{}.json 
            TestSingleDataFromFile:
                IOV: /path/to/listOfRunsOrIOVs.txt
                dataset: /path/to/listOfAllFiles.txt
```
TestSingleMC: Run number 1 is reserved for MC objects only.
TestSingleDataIOV/Run: In case of data, selected numbers can represent both IOV and run numbers. Luminosity is assigned from 'goodlumi' json file where you can define if number should be understood as IOV or single run. Lumiblock structure should also be defined, see Example: `/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/datasetfiles/UltraLegacy/Run2/forDMRweighted/MuonIsolated2018_new/IOV_Vali_320933.json` defines IOV=320933 consisting o 3 runs.
TestSingleDataFromFile: If list of IOVs/Runs is too long, you can provide it in a form of plain txt list (one number for each line). Dataset file can also contain all input file names (no curly brackets), however it is NOT recommended as it makes jobs much longer.

Example B:
```
trends:
    Run2trend:
        singles:
            - Run2018B
        firstRun: 317087
        lastRun: 317212
```
When defining trend job you can also specify starting and ending run to be plotted. 

```
style:
    PV:
        merge:
            CMSlabel: Preliminary
    trends:
        CMSlabel: Internal
        Rlabel: 2018B
        lumiInputFile: /path/to/lumiperIOV.txt
```
`lumiInputFile` is used for trend plotting step only and it defines integrated luminosity for each IOV/run considered. It needs to be a plain format with two columns (<run> <lumi>). Following schemes are supported:

```
RUN 1 <space> lumi for the only run in IOV 1
...
RUN 4 <space> lumi for the starting run (4) of IOV 4
RUN 5 <space> lumi for another run (5) of IOV 4
RUN 6 <space> lumi for another run (6) of IOV 4
```

or

```
IOV 1 <space> lumi for all runs in IOV 1, in this case could be just one run
...
IOV 4 <space> lumi for all runs in IOV 4, in this case sum of lumi for RUN 4,5 and 6
```
