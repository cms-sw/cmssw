#!/bin/sh
#Rootfiles merged using:
#/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1_FINAL/CMSSW_5_3_4_patch1/src/merge_from_storage.sh
#
path_to_rootfiles=/afs/cern.ch/work/p/pvmulder/public/BTagging/FINAL_NOSPIKES/CMSSW_5_3_4_patch1/src/

echo "Filling the 2D pt/eta histograms and calculating the pt/eta weights" 

g++ histoJetEtaPt.cpp `root-config --cflags --glibs` -o histos
./histos $path_to_rootfiles

echo "saving the relevant variables " 
nohup cmsRun MVATrainer_B_cfg.py &
nohup cmsRun MVATrainer_C_cfg.py &
nohup cmsRun MVATrainer_DUSG_cfg.py &

#hadd train_save_all.root train_B_save.root train_C_save.root train_DUSG_save.root

#echo "Do the actual training"

#TRAINING_TAG1=SC_CSVSL
#mkdir $TRAINING_TAG1
#cd $TRAINING_TAG1
#nohup mvaTreeTrainer ../SuperCombined_CSV_SL.xml SC_CSV_SL.mva #../train_save_all.root &
#cd ..
#
#TRAINING_TAG2=SC_CSVJP
#mkdir $TRAINING_TAG2
#cd $TRAINING_TAG2
#nohup mvaTreeTrainer ../SuperCombined_CSV_JP.xml SC_CSV_JP.mva ../train_save_all.root &
#cd ..

#TRAINING_TAG3=SC_CSVJPSL
#mkdir $TRAINING_TAG3
#cd $TRAINING_TAG3
#nohup mvaTreeTrainer ../SuperCombined_CSV_JP_SL.xml SC_CSV_JP_SL.mva #../train_save_all.root &
#cd ..

#TRAINING_TAG4=SC_CSVJPSM
#mkdir $TRAINING_TAG4
#cd $TRAINING_TAG4
#nohup mvaTreeTrainer ../SuperCombined_CSV_JP_SM.xml SC_CSV_JP_SM.mva ../train_save_all.root &
#cd ..
#
#TRAINING_TAG5=SC_CSVSM
#mkdir $TRAINING_TAG5
#cd $TRAINING_TAG5
#nohup mvaTreeTrainer ../SuperCombined_CSV_SM.xml SC_CSV_SM.mva ../train_save_all.root &
#cd ..

echo "do now manually cmsRun ../copyMVAToSQLite_cfg.py to copy the mva training output to sqlite format"
#echo "run the validation from Validation/RecoB/test/ afterwards -> usually on ttbar events, make sure you read in the *db file produced in the previous step"
