
dataFile=/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/D65FC93B-5BAB-DF11-A9AF-001D09F2516D.root
rootFile=step2_DT2_1_RAW2DIGI_RECO_DQM.root

cmsDriver.py step2_DT2_1 -s RAW2DIGI,RECO:reconstructionCosmics,DQM -n 1000 --eventcontent RECO --conditions auto:craft09 --geometry Ideal --filein $dataFile --data --scenario cosmics

cmsDriver.py step3_DT2_1 -s HARVESTING:dqmHarvesting --conditions auto:craft09 --filein file:$rootFile --data --scenario=cosmics

