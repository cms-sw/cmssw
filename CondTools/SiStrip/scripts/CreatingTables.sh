#!/bin/sh
connectstring=$1
aUSER=$2
aPASSWD=$3

eval `scramv1 runtime -sh`

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
  echo -e  "\npool_build_object_relational_mapping -f $obj   -d CondFormatsSiStripObjects -c ${connectstring} -s $aUSER -p $aPASSWD \n"
  pool_build_object_relational_mapping -f $obj   -d CondFormatsSiStripObjects -c ${connectstring} -u $aUSER -p $aPASSWD
done
