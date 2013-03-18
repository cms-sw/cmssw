#!/bin/sh

path_to_rootfiles=/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1_SETEST/CMSSW_5_3_4_patch1/src/Rootfiles/RootFiles_SCtagger_TTJets_newSL_newCSV_noFit_OffsetFix

echo "Filling the 2D pt/eta histograms and calculating the pt/eta weights" 

g++ histoJetEtaPt.cpp `root-config --cflags --glibs` -o histos
./histos $path_to_rootfiles

echo "saving the relevant variables " 
nohup cmsRun MVATrainer_B_cfg.py &
nohup cmsRun MVATrainer_C_cfg.py &
nohup cmsRun MVATrainer_DUSG_cfg.py &

hadd train_save_all.root train_B_save.root train_C_save.root train_DUSG_save.root

echo "Do the actual training"

TRAINING_TAG=SC_woJP_noweights
mkdir $TRAINING_TAG
cd $TRAINING_TAG
nohup mvaTreeTrainer -w ../SuperCombined_woJP.xml SC_noweights.mva ../train_save_all.root &
cd ..

TRAINING_TAG=SC_woJP
mkdir $TRAINING_TAG
cd $TRAINING_TAG
nohup mvaTreeTrainer ../SuperCombined_woJP.xml SC_weights.mva ../train_save_all.root &
cd ..

#echo "adapt and run copyMVAToSQLite_cfg.py to get the training output to sqlite format"
#echo "run the validation -> usually on ttbar events, make sure you read in the *db file produced in the previous step"
