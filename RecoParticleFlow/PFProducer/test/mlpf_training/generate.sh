#!/bin/bash
set -e
set -x

#sleep randomly up to 120s to stagger job start times
#sleep $((RANDOM % 120))

#seed must be greater than 0
SAMPLE=$1
SEED=$2
N=$3

#examples:
#SAMPLE=TTbar_14TeV_TuneCUETP8M1_cfi
#SEED=1
#N=5

PILEUP=Run3_Flat55To75_PoissonOOTPU
#PILEUP=NoPileUp

#in case of locally downloaded minbias files, use the following
#PILEUP_INPUT=filelist:/storage/user/jpata/particleflow/test/pu_files.txt
PILEUP_INPUT=dbs:/MinBias_TuneCP5_13TeV-pythia8/RunIIFall18GS-102X_upgrade2018_realistic_v9-v1/GEN-SIM
#and add this line to cmsDriver for step2
#--pileup_input $PILEUP_INPUT \

#Generate the MC
cmsDriver.py $SAMPLE \
  --conditions auto:phase1_2022_realistic \
  -n $N \
  --era Run3 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT \
  --datatier GEN-SIM \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --pileup_input $PILEUP_INPUT \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

#Run the reco sequences
cmsDriver.py step3 \
  --conditions auto:phase1_2022_realistic \
  --era Run3 \
  -n -1 \
  --eventcontent FEVTDEBUGHLT \
  --runUnscheduled \
  -s RAW2DIGI,L1Reco,RECO,RECOSIM \
  --datatier GEN-SIM-RECO \
  --geometry DB:Extended \
  --no_exec \
  --filein file:step2_phase1_new.root \
  --fileout step3_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step3 \
  --python_filename=step3_phase1_new.py

pwd
ls -lrt

echo "process.RandomNumberGeneratorService.generator.initialSeed = $SEED" >> step2_phase1_new.py
cmsRun step2_phase1_new.py
cmsRun step3_phase1_new.py
cmsRun $CMSSW_BASE/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py

mv pfntuple.root pfntuple_${SEED}.root
