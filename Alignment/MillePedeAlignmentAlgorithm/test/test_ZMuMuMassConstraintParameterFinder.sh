#!/bin/bash
function die { echo $1: status $2; exit $2; }

(cmsRun ${SCRAM_TEST_PATH}/ZMuMuMassConstraintParameterFinder/zmumudistribution_cfg.py inputFiles=/store/relval/CMSSW_14_0_0_pre2/RelValZMM_14/GEN-SIM/133X_mcRun3_2024_realistic_v5_STD_2024_PU-v1/2590000/c38cee3f-99d7-48aa-b236-86f6bbc869b3.root,/store/relval/CMSSW_14_0_0_pre2/RelValZMM_14/GEN-SIM/133X_mcRun3_2024_realistic_v5_STD_2024_PU-v1/2590000/5bf98cca-d491-4e95-98b0-d3acb6ea0807.root,/store/relval/CMSSW_14_0_0_pre2/RelValZMM_14/GEN-SIM/133X_mcRun3_2024_realistic_v5_STD_2024_PU-v1/2590000/1e362cc1-235b-4c32-bb24-178ccac4659f.root) || die 'failed running ZMuMuMassConstraintParameterFinder' $?
-- dummy change --
