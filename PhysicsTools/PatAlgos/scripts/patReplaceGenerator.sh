#!/bin/bash

# warning: these scripts don't work for replaces on parameters in parameter's psets
#          see the added comment in PATLayer1_ReplaceDefaults_fast.cff
#
#
# file name settings
export baseDir=${CMSSW_BASE}/src/PhysicsTools/PatAlgos
if [ ! -d $baseDir ]; then 
  echo "*** " $baseDir not found
  echo "*** " Please make sure to run scramv1 run -[c]sh first
  exit -1
fi
export layer0FamosSetup=${baseDir}/data/famos/patLayer0_FamosSetup.cff
export layer0FileFull=${baseDir}/test/patLayer0_ReplaceDefaults_full.cff
export layer0FileFast=${baseDir}/test/patLayer0_ReplaceDefaults_fast.cff
export layer1FamosSetup=${baseDir}/data/famos/patLayer1_FamosSetup.cff
export layer1FileFull=${baseDir}/test/patLayer1_ReplaceDefaults_full.cff
export layer1FileFast=${baseDir}/test/patLayer1_ReplaceDefaults_fast.cff

# make backups of the old ones
if [ -e $layer0FileFull ]; then mv $layer0FileFull $layer0FileFull.bak; fi
if [ -e $layer0FileFast ]; then mv $layer0FileFast $layer0FileFast.bak; fi
if [ -e $layer1FileFull ]; then mv $layer1FileFull $layer1FileFull.bak; fi
if [ -e $layer1FileFast ]; then mv $layer1FileFast $layer1FileFast.bak; fi

# Define input directories
export dataDir=${baseDir}/data
export recDir=${dataDir}/recoLayer0
export clDir=${dataDir}/cleaningLayer0
export mcDir=${dataDir}/mcMatchLayer0
export prodDir=${dataDir}/producersLayer1
export selDir=${dataDir}/selectionLayer1

# produce the replace-file
cat > $layer0FileFull << EOF

############################
### PAT Layer-0 cleaning ###
############################
`./patReplaceParser.pl ${clDir}/caloJetCleaner.cfi`
`./patReplaceParser.pl ${clDir}/caloMetCleaner.cfi`
`./patReplaceParser.pl ${clDir}/electronCleaner.cfi`
`./patReplaceParser.pl ${clDir}/muonCleaner.cfi`
`./patReplaceParser.pl ${clDir}/pfTauCleaner.cfi`
`./patReplaceParser.pl ${clDir}/photonCleaner.cfi`

###############################
### PAT Layer-0 MC matching ###
###############################
`./patReplaceParser.pl ${mcDir}/muonMatch.cfi`
`./patReplaceParser.pl ${mcDir}/electronMatch.cfi`
`./patReplaceParser.pl ${mcDir}/photonMatch.cfi`
`./patReplaceParser.pl ${mcDir}/tauMatch.cfi`
`./patReplaceParser.pl ${mcDir}/jetMatch.cfi`

###############################
### PAT Layer-0 jets        ###
###############################
`./patReplaceParser.pl ${recDir}/jetTracksCharge.cff`


EOF

# adapt layer 1 replace-file for fast simulation
cp $layer0FileFull $layer0FileFast
./patReplaceFast.pl $layer0FileFast $layer0FamosSetup


# produce the replace-file
cat > $layer1FileFull << EOF

####################################
### PAT Layer-1 object producers ###
####################################
`./patReplaceParser.pl ${prodDir}/muonProducer.cfi`
`./patReplaceParser.pl ${prodDir}/electronProducer.cfi`
`./patReplaceParser.pl ${prodDir}/photonProducer.cfi`
`./patReplaceParser.pl ${prodDir}/tauProducer.cfi`
`./patReplaceParser.pl ${prodDir}/jetProducer.cfi`
`./patReplaceParser.pl ${prodDir}/metProducer.cfi`

####################################
### PAT Layer-1 Object Selectors ###
####################################
`./patReplaceParser.pl ${selDir}/muonSelector.cfi`
`./patReplaceParser.pl ${selDir}/electronSelector.cfi`
`./patReplaceParser.pl ${selDir}/photonSelector.cfi`
`./patReplaceParser.pl ${selDir}/tauSelector.cfi`
`./patReplaceParser.pl ${selDir}/jetSelector.cfi`
`./patReplaceParser.pl ${selDir}/metSelector.cfi`

#################################
### PAT Layer-1 Count Filters ###
#################################
`./patReplaceParser.pl ${selDir}/muonMinFilter.cfi`
`./patReplaceParser.pl ${selDir}/muonMaxFilter.cfi`
`./patReplaceParser.pl ${selDir}/electronMinFilter.cfi`
`./patReplaceParser.pl ${selDir}/electronMaxFilter.cfi`
`./patReplaceParser.pl ${selDir}/photonMinFilter.cfi`
`./patReplaceParser.pl ${selDir}/photonMaxFilter.cfi`
`./patReplaceParser.pl ${selDir}/tauMinFilter.cfi`
`./patReplaceParser.pl ${selDir}/tauMaxFilter.cfi`
`./patReplaceParser.pl ${selDir}/jetMinFilter.cfi`
`./patReplaceParser.pl ${selDir}/jetMaxFilter.cfi`
`./patReplaceParser.pl ${selDir}/leptonCountFilter.cfi`

EOF

# adapt layer 1 replace-file for fast simulation
cp $layer1FileFull $layer1FileFast
./patReplaceFast.pl $layer1FileFast $layer1FamosSetup
