#!/bin/sh

function create_mapping(){

    [ -e $1-mapping-default.xml ] && rm -f $1-mapping-default.xml 

    pool_build_object_relational_mapping -f mapping-template-$1.xml -o $1-mapping-default.xml -b -d CondFormatsSiStripObjects -c sqlite_file:dummy.db -b
    
    rm -f dummy.db
    
    [ -e $1-mapping-custom.xml ] && rm -f $1-mapping-custom.xml 
    cp $1-mapping-default.xml $1-mapping-custom.xml
 
    echo -e "\nnow... \nedit $1-mapping-custom.xml change to meaningful names"

    echo -e "\nPlease follow instructions reported in file OracleDBAProcedure\n"
    
    echo "----------------------------------------------"
}


##################
###  MAIN 
#################

export CORAL_AUTH_USER=whoever
export CORAL_AUTH_PASSWORD=whatever
eval `scramv1 runtime -sh`

[ `echo $@ | grep  -c "\-help[ ]*"` = 1 ] && echo "[usage] SiStrip-create-default-mapping.sh <-noise> <-ped> <-cabling> <-all>"

if [ `echo $@ | grep  -c "\-all"` = 1 ];
    then
    create_mapping SiStripNoises    
    create_mapping SiStripPedestals 
    create_mapping SiStripFedCabling
else
    [ `echo $@ | grep  -c "\-noise[ ]*"` = 1 ]   && create_mapping SiStripNoises
    [ `echo $@ | grep  -c "\-ped"` = 1 ]     && create_mapping SiStripPedestals
    [ `echo $@ | grep  -c "\-cabling"` = 1 ] && create_mapping SiStripFedCabling
fi


