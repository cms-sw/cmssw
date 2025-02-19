#!/bin/sh                                                                                                                                                                                                         
connectstring=$1
USER=$2
PASSWD=$3

eval `scramv1 runtime -sh`

export TNS_ADMIN=/nfshome0/xiezhen/conddb/

workdir=`pwd`

path=$CMSSW_BASE/src/CondFormats/SiStripObjects/xml/
if [ ! -e $path ] ;then
path=$CMSSW_RELEASE_BASE/src/CondFormats/SiStripObjects/xml/
if [ ! -e $path ]; then
echo -e "Error: CondFormats/SiStripObjects/xml doesn't exist\nplease install that package\nexit"
exit 1
fi
fi


echo -e "\n-----------\nCreating tables for db ${connectstring} \n-----------\n"

#cmscond_bootstrap_detector.pl --offline_connect ${connectstring} --auth /afs/cern.ch/cms/DB/conddb/authentication.xml STRIP                                                                                      
for obj in `ls $path/*xml`
  do
  echo -e  "\npool_build_object_relational_mapping -f $obj   -d CondFormatsSiStripObjects -c ${connectstring}\n"
  pool_build_object_relational_mapping -f $obj   -d CondFormatsSiStripObjects -c ${connectstring} -u $USER -p $PASSWD
done
