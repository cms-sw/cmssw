#!/bin/bash
curl -s -k https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_fragment/BTV-RunIISummer20UL17GEN-00002 --retry 3 --create-dirs -o ${LOCALTOP}/src/Configuration/GenProduction/python/BTV-RunIISummer20UL17GEN-00002-fragment.py

cd ${LOCALTOP}/src
scram b
cd ../..

cmsDriver.py Configuration/GenProduction/python/BTV-RunIISummer20UL17GEN-00002-fragment.py --python_filename test_BTV-RunIISummer20UL17GEN-00002_1_cfg.py --eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN --fileout file:test_BTV-RunIISummer20UL17GEN-00002.root --conditions auto:run2_mc --beamspot Realistic25ns13TeVEarly2017Collision --customise_commands process.source.numberEventsInLuminosityBlock="cms.untracked.uint32(100)" --step GEN --geometry DB:Extended --era Run2_2017 --no_exec --mc -n 10 --nThreads 2 --nConcurrentLumis 0

cmsRun test_BTV-RunIISummer20UL17GEN-00002_1_cfg.py
rm ${LOCALTOP}/src/Configuration/GenProduction/python/BTV-RunIISummer20UL17GEN-00002-fragment.py
