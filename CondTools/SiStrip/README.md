## CondTools/SiStrip

- _SiStripApvGainReader_: reads APV gains from input sqlite. Dumps TTree with following infos: detectorID, APV number, gain
                        Infos are also dumped in txt file, in format : DetId APV1 APV2 APV3 APV4 (APV5) (APV6)
  - To run: 
    - `cmsRun test/SiStripApvGainReader_cfg.py inputFiles=sqlite_input  tag=my_tag runN=run_in_IOV`
    - `readSiStripApvGain.py  inputFiles=sqlite_input  tag=my_tag runN=run_in_IOV`

- _SiStripChannelGainFromDBMiscalibrator_: reads APV gains (either G1 or G2) from DB and applies hierarchically a scale factor and/or gaussian smearing for each APV gain, down to the individual layer or disk level
  - To run: 
    - `cmsRun test/SiStripChannelGainFromDBMiscalibrator_cfg.py globalTag=<inputGT> runNumber=<inputIOV>`
  
- _SiStripNoisesFromDBMiscalibrator_: reads Noise from DB and applies hierarchically a scale factor and/or gaussian smearing for each APV gain, down to the individual layer or disk level
  - To run: 
    - `cmsRun test/SiStripNoiseFromDBMiscalibrator_cfg.py globalTag=<inputGT> runNumber=<inputIOV>`