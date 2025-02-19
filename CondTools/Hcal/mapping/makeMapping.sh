#!/bin/sh
if [ $# -ne "1" ] 
then
    echo Generates XML mapping files for HCAL objects
    echo Usage $0 version_name
    exit 1
fi

version=$1

# redundant variables
CORAL_AUTH_USER=blah
CORAL_AUTH_PASSWORD=blah
export CORAL_AUTH_USER
export CORAL_AUTH_PASSWORD

#for object in HcalPedestals HcalPedestalWidths HcalGains HcalGainWidths HcalQIEData HcalElectronicsMap HcalChannelQuality
for object in HcalPedestals HcalPedestalWidths HcalGains HcalGainWidths HcalElectronicsMap HcalChannelQuality
do
    echo processing $object...
    defaultname=$object-mapping-cmsdefault.xml
    outname=$object"-mapping-custom_$version.xml"
    rm -f $defaultname $outname
    ../../Utilities/bin/create_default_mapping -v $object CondFormatsHcalObjects
# now modify it according to https://uimon.cern.ch/twiki/bin/view/CMS/O2O-HOWTO
    tablename=`echo $object | sed 's/[a-z]/\u&/g'`
    cat $defaultname | sed 's/id_columns="ID"/id_columns="IOV_VALUE_ID"/g' | sed 's/ID_ID/IOV_VALUE_ID/g' | sed 's/MITEMS_//g' | sed 's/version="cmsdefault"/version="'$version'"/g' > $outname
done
