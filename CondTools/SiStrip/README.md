## CondTools/SiStrip

- _SiStripApvGainReader_: reads APV gains from input sqlite. Dumps TTree with following infos: detectorID, APV number, gain
                        Infos are also dumped in txt file, in format : DetId APV1 APV2 APV3 APV4 (APV5) (APV6)
  - To run: 
    - `cmsRun test/SiStripApvGainReader_cfg.py inputFiles=sqlite_input  tag=my_tag runN=run_in_IOV`
    - `readSiStripApvGain.py  inputFiles=sqlite_input  tag=my_tag runN=run_in_IOV`
