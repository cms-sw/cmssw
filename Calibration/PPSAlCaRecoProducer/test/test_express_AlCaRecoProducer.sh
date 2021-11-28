#!/bin/bash

customise_commands="process.GlobalTag.toGet = cms.VPSet()\n\
process.GlobalTag.toGet.append(cms.PSet(record = cms.string(\"AlCaRecoTriggerBitsRcd\"),tag =  cms.string(\"AlCaRecoHLTpaths_PPS2022_express_v1\"), connect = cms.string(\"frontier://FrontierProd/CMS_CONDITIONS\")))\
\nprocess.ALCARECOPPSCalMaxTracksFilter.TriggerResultsTag = cms.InputTag(\"TriggerResults\",\"\",\"HLTX\")"

cmsDriver.py testExpressPPSAlCaRecoProducer  -s ALCAPRODUCER:PPSCalMaxTracks,ENDJOB \
--process ALCARECO \
--scenario pp \
--era ctpps_2018 \
--conditions auto:run3_data_express \
--data  \
--datatier ALCARECO \
--eventcontent ALCARECO \
-n 100  --filein file:/eos/project-c/ctpps/subsystems/Software/Off-line/AlCaTest/outputALCAPPS_single.root \
--fileout file:outputALCAPPS_RECO_express.root \
--customise_commands="$customise_commands"