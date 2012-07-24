#!/bin/bash

if [ -z "$CMSSW_VERSION" ];then
    eval `scramv1 runtime -sh`
fi




#echo "[STATUS] ElectronRecalibSuperClusterAssociator replacing"
#cp ElectronRecalibSuperClusterAssociator.cc.tmp ../../src/ElectronRecalibSuperClusterAssociator.cc
#cp ElectronRecalibSuperClusterAssociator.h ../../interface

echo "[STATUS] GsfElectron class patching"
if [ ! -d "../../DataFormats/EgammaCandidates/" ]; then
    echo "[ERROR] please check out DataFormats/EgammaCandidates folder before running the patch:" >> /dev/stderr
    echo "[ERROR] addpkg DataFormats/EgammaCandidates" >> /dev/stderr
    exit 1
fi


#patch ../../../../Data
sed -i 's|assert|//assert|' ../../DataFormats/EgammaCandidates/src/GsfElectron.cc 

#echo "[STATUS] Copying new AlCaElectronsProducer.cc"
#cp AlCaElectronsProducer.cc ../../../EcalAlCaRecoProducers/src/


#echo "[STATUS] Compiling EDM plugins "
#if [ "`grep -c EDM_PLUGIN ../../BuildFile.xml`" == "0" ]; then
#    echo "<flags   EDM_PLUGIN=\"1\"/>" >> ../../BuildFile.xml
#fi
