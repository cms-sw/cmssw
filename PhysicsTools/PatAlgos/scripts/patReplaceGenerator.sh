#!/bin/bash

# warning: these scripts don't work for replaces on parameters in parameter's psets
#          see the added comment in PATLayer1_ReplaceDefaults_fast.cff
#
#
# file name settings
export baseDir=${CMSSW_BASE}/src/PhysicsTools/PatAlgos
export parser=${CMSSW_BASE}/src/PhysicsTools/PatAlgos/scripts/patReplaceParser.pl
export fastReplacer=${CMSSW_BASE}/src/PhysicsTools/PatAlgos/scripts/patReplaceFast.pl
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
`${parser} ${clDir}/caloJetCleaner.cfi`
`${parser} ${clDir}/caloMetCleaner.cfi`
`${parser} ${clDir}/electronCleaner.cfi`
`${parser} ${clDir}/muonCleaner.cfi`
`${parser} ${clDir}/pfTauCleaner.cfi`
`${parser} ${clDir}/photonCleaner.cfi`

###############################
### PAT Layer-0 MC matching ###
###############################
`${parser} ${mcDir}/muonMatch.cfi`
`${parser} ${mcDir}/electronMatch.cfi`
`${parser} ${mcDir}/photonMatch.cfi`
`${parser} ${mcDir}/tauMatch.cfi`
`${parser} ${mcDir}/jetMatch.cfi`

###############################
### PAT Layer-0 jets        ###
###############################
`${parser} ${recDir}/jetTracksCharge.cff`


EOF

# adapt layer 1 replace-file for fast simulation
cp $layer0FileFull $layer0FileFast
${fastReplacer} $layer0FileFast $layer0FamosSetup


# produce the replace-file
cat > $layer1FileFull << EOF

####################################
### PAT Layer-1 object producers ###
####################################
`${parser} ${prodDir}/muonProducer.cfi`
`${parser} ${prodDir}/electronProducer.cfi`
`${parser} ${prodDir}/photonProducer.cfi`
`${parser} ${prodDir}/tauProducer.cfi`
`${parser} ${prodDir}/jetProducer.cfi`
`${parser} ${prodDir}/metProducer.cfi`

####################################
### PAT Layer-1 Object Selectors ###
####################################
`${parser} ${selDir}/muonSelector.cfi`
`${parser} ${selDir}/electronSelector.cfi`
`${parser} ${selDir}/photonSelector.cfi`
`${parser} ${selDir}/tauSelector.cfi`
`${parser} ${selDir}/jetSelector.cfi`
`${parser} ${selDir}/metSelector.cfi`

#################################
### PAT Layer-1 Count Filters ###
#################################
`${parser} ${selDir}/muonMinFilter.cfi`
`${parser} ${selDir}/muonMaxFilter.cfi`
`${parser} ${selDir}/electronMinFilter.cfi`
`${parser} ${selDir}/electronMaxFilter.cfi`
`${parser} ${selDir}/photonMinFilter.cfi`
`${parser} ${selDir}/photonMaxFilter.cfi`
`${parser} ${selDir}/tauMinFilter.cfi`
`${parser} ${selDir}/tauMaxFilter.cfi`
`${parser} ${selDir}/jetMinFilter.cfi`
`${parser} ${selDir}/jetMaxFilter.cfi`
`${parser} ${selDir}/leptonCountFilter.cfi`

EOF

# adapt layer 1 replace-file for fast simulation
cp $layer1FileFull $layer1FileFast
${fastReplacer} $layer1FileFast $layer1FamosSetup
