#!/bin/bash

# warning: these scripts don't work for replaces on parameters in parameter's psets
#          see the added comment in PATLayer1_ReplaceDefaults_fast.cff

# file name settings
export layer0FamosSetup=PATLayer0_FamosSetup.cff
export layer0FileFull=PATLayer0_ReplaceDefaults_full.cff
export layer0FileFast=PATLayer0_ReplaceDefaults_fast.cff
export layer1FamosSetup=PATLayer1_FamosSetup.cff
export layer1FileFull=PATLayer1_ReplaceDefaults_full.cff
export layer1FileFast=PATLayer1_ReplaceDefaults_fast.cff

# make backups of the old ones
if [ -e $layer0FileFull ]; then mv $layer0FileFull $layer0FileFull.bak; fi
if [ -e $layer0FileFast ]; then mv $layer0FileFast $layer0FileFast.bak; fi
if [ -e $layer1FileFull ]; then mv $layer1FileFull $layer1FileFull.bak; fi
if [ -e $layer1FileFast ]; then mv $layer1FileFast $layer1FileFast.bak; fi

# build up the data dir path
cd .. ;
export datadir="`pwd`/data"
cd $OLDPWD

# produce the replace-file
cat > $layer0FileFull << EOF

############################
### PAT Layer-0 cleaning ###
############################

# Generated from ${datadir}/PATLayer0.cff
`./ReplaceParser.sh ../data/PATLayer0.cff`

###############################
### PAT Layer-0 MC matching ###
###############################

# Generated from ${datadir}/muonMatch.cfi
`./ReplaceParser.sh ../data/muonMatch.cfi`
# Generated from ${datadir}/electronMatch.cfi
`./ReplaceParser.sh ../data/electronMatch.cfi`
# Generated from ${datadir}/photonMatch.cfi
`./ReplaceParser.sh ../data/photonMatch.cfi`
# Generated from ${datadir}/tauMatch.cfi
`./ReplaceParser.sh ../data/tauMatch.cfi`
# Generated from ${datadir}/jetMatch.cfi
`./ReplaceParser.sh ../data/jetMatch.cfi`

EOF

# adapt layer 0 replace-file for fast simulation
export extraReplaces=`cat $layer0FamosSetup | \
  sed 's/#/\nREMOVE/' | sed 's/\/\//\nREMOVE/' | grep -v REMOVE | \
  awk '/replace/ { print $2 }' | tr "\n" " "`
cp $layer0FileFull $layer0FileFast
for matchStr in $extraReplaces ; do
  export replaceStr=`grep $matchStr $layer0FamosSetup | sed 's/^[ ]*//'`
  cat $layer0FileFast | \
  awk -v matchStr="$matchStr" -v replaceStr="$replaceStr" \
    '{ do { if (matchStr == $2) print replaceStr; else print $0; } while (getline) }' \
    > $layer0FileFast.tmp
  mv $layer0FileFast.tmp $layer0FileFast
done

# produce the replace-file
cat > $layer1FileFull << EOF

####################################
### PAT Layer-1 object producers ###
####################################

# Generated from ${datadir}/PATMuonProducer.cfi
`./ReplaceParser.sh ../data/PATMuonProducer.cfi`
# Generated from ${datadir}/PATElectronProducer.cfi
`./ReplaceParser.sh ../data/PATElectronProducer.cfi`
# Generated from ${datadir}/PATPhotonProducer.cfi
`./ReplaceParser.sh ../data/PATPhotonProducer.cfi`
# Generated from ${datadir}/PATTauProducer.cfi
`./ReplaceParser.sh ../data/PATTauProducer.cfi`
# Generated from ${datadir}/PATJetProducer.cfi
`./ReplaceParser.sh ../data/PATJetProducer.cfi`
# Generated from ${datadir}/PATMETProducer.cfi
`./ReplaceParser.sh ../data/PATMETProducer.cfi`

####################################
### PAT Layer-1 Object Selectors ###
####################################

# Generated from ${datadir}/PATMuonSelector.cfi
`./ReplaceParser.sh ../data/PATMuonSelector.cfi`
# Generated from ${datadir}/PATElectronSelector.cfi
`./ReplaceParser.sh ../data/PATElectronSelector.cfi`
# Generated from ${datadir}/PATPhotonSelector.cfi
`./ReplaceParser.sh ../data/PATPhotonSelector.cfi`
# Generated from ${datadir}/PATTauSelector.cfi
`./ReplaceParser.sh ../data/PATTauSelector.cfi`
# Generated from ${datadir}/PATJetSelector.cfi
`./ReplaceParser.sh ../data/PATJetSelector.cfi`
# Generated from ${datadir}/PATMETSelector.cfi
`./ReplaceParser.sh ../data/PATMETSelector.cfi`

#################################
### PAT Layer-1 Count Filters ###
#################################

# Generated from ${datadir}/PATMuonMinFilter.cfi
`./ReplaceParser.sh ../data/PATMuonMinFilter.cfi`
# Generated from ${datadir}/PATMuonMaxFilter.cfi
`./ReplaceParser.sh ../data/PATMuonMaxFilter.cfi`
# Generated from ${datadir}/PATElectronMinFilter.cfi
`./ReplaceParser.sh ../data/PATElectronMinFilter.cfi`
# Generated from ${datadir}/PATElectronMaxFilter.cfi
`./ReplaceParser.sh ../data/PATElectronMaxFilter.cfi`
# Generated from ${datadir}/PATPhotonMinFilter.cfi
`./ReplaceParser.sh ../data/PATPhotonMinFilter.cfi`
# Generated from ${datadir}/PATPhotonMaxFilter.cfi
`./ReplaceParser.sh ../data/PATPhotonMaxFilter.cfi`
# Generated from ${datadir}/PATTauMinFilter.cfi
`./ReplaceParser.sh ../data/PATTauMinFilter.cfi`
# Generated from ${datadir}/PATTauMaxFilter.cfi
`./ReplaceParser.sh ../data/PATTauMaxFilter.cfi`
# Generated from ${datadir}/PATJetMinFilter.cfi
`./ReplaceParser.sh ../data/PATJetMinFilter.cfi`
# Generated from ${datadir}/PATJetMaxFilter.cfi
`./ReplaceParser.sh ../data/PATJetMaxFilter.cfi`
# Generated from ${datadir}/PATLeptonCountFilter.cfi
`./ReplaceParser.sh ../data/PATLeptonCountFilter.cfi`

EOF

# adapt layer 1 replace-file for fast simulation
export extraReplaces=`cat $layer1FamosSetup | \
  sed 's/#/\nREMOVE/' | sed 's/\/\//\nREMOVE/' | grep -v REMOVE | \
  awk '/replace/ { print $2 }' | tr "\n" " "`
cp $layer1FileFull $layer1FileFast
for matchStr in $extraReplaces ; do
  export replaceStr=`grep $matchStr $layer1FamosSetup | sed 's/^[ ]*//'`
  cat $layer1FileFast | \
  awk -v matchStr="$matchStr" -v replaceStr="$replaceStr" \
    '{ do { if (matchStr == $2) print replaceStr; else print $0; } while (getline) }' \
    > $layer1FileFast.tmp
  mv $layer1FileFast.tmp $layer1FileFast
done
