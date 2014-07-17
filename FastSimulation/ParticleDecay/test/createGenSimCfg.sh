#!/bin/bash

pid=$1
nevents=$2
ofile="pgun_$1.py"

cmsDriver.py FastSimulation/ParticleDecay/python/genfragment_ptgun_cfg.py --fast --conditions auto:upgradePLS1 -n 10 \
--eventcontent FEVTDEBUG -s GEN,SIM \
--no_exec \
--python_file $ofile

extra="process.generator.PGunParameters.PartID=[$1]
process.famosSimHits.ParticleFilter.EProton = 0.0
process.famosSimHits.ParticleFilter.etaMax = 99999999
process.famosSimHits.ParticleFilter.pTMin = 0.0
process.famosSimHits.ParticleFilter.EMin = 0.0
process.famosSimHits.MaterialEffects.Bremsstrahlung = False
process.famosSimHits.MaterialEffects.NuclearInteraction = False
process.famosSimHits.MaterialEffects.PairProduction = False
process.famosSimHits.MaterialEffects.MuonBremsstrahlung = False
process.famosSimHits.MaterialEffects.MultipleScattering = False
process.famosSimHits.MaterialEffects.EnergyLoss = False
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.UseMagneticField = False
process.maxEvents.input = $2
process.FEVTDEBUGoutput.fileName = \"pgun_$1.root\"
process.FEVTDEBUGoutput.outputCommands.extend(['drop *','keep *_famosSimHits_*_*'])"

echo "$extra" >> $ofile
