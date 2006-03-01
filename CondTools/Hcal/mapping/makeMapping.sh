#!/bin/sh
if [ $# -ne "1" ] 
then
    echo Generates XML mapping files for HCAL objects
    echo Usage $0 version_name
    exit 1
fi

version=$1

for object in HcalPedestals HcalPedestalWidths HcalGains HcalGainWidths HcalQIEData HcalElectronicsMap HcalChannelQuality
do
    echo processing $object...
    # making template
    template_name=mapping_template_$object.xml
    rm -f $template_name
    echo "<?xml version='1.0' encoding='UTF-8'?>" > $template_name
    echo "<!DOCTYPE Mapping SYSTEM 'InMemory'>" >> $template_name
    echo "<Mapping version='"$object"-"$version"' >" >> $template_name
    echo "<Class name='"$object"' >" >> $template_name
    echo "</Class >" >> $template_name
    echo "</Mapping >" >> $template_name

    #processing template
    echo Generating mapping file for $object...
    pool_build_object_relational_mapping -f $template_name -o $object"_mapping_"$version.xml -d CondFormatsHcalObjects -c sqlite_file:trash.db -b -u whoever -p whatever

done
