#!/bin/bash

# pattern test data harvesting script
#
# 
#
#
# usage :
#     getPattData.sh pattId run
#

# set variables
ecalDir='/nfshome0/ecaldev/trigg_patt_test'
hcalDir='/nfshome0/toole/hcal_pattern_test_files'
rctDir='/nfshome0/rctpro'
scDir='/home/hwtest/patt_test'
gctDir='/nfshome0/gctdev/patt_test'

# get dataset info
pattId=$1
run=$2

pattDir=`awk "/$pattId/ { print "'$2'" }" pattIndex.dat`
pattName=`awk "/$pattId/ { print "'$3'" }" pattIndex.dat`

# create local directory
targDir=$pattName-$run
if [[ ! -d $targDir ]]; then mkdir $targDir; fi;

# copy data files
echo "Copying data for run $run of pattern $pattId from $pattDir/$pattName"
ecalFile=$ecalDir/$pattDir/$pattName-$pattId-ecal-0-*.txt
hcalFile=$hcalDir/$pattDir/$pattName-$pattId-hcal-0-.xml
rctFile=$rctDir/$pattDir/$pattName-$pattId-rct-$run-*.txt
scFile=$scDir/$pattDir/$pattName-$pattId-sc-$run-0.txt
gctFile=$gctDir/$pattDir/$pattName-$pattId-gct-$run-0.txt

scp "cmsusr0:\"$ecalFile\"" $targDir/.
scp "cmsusr0:\"$hcalFile\"" $targDir/.
scp "cmsusr0:\"$rctFile\"" $targDir/.
scp "cmsusr0:\"$gctFile\"" $targDir/.
scp "cms-bris-pc01:\"$scFile\"" $targDir/.

#copy config files
conf=`awk "/$pattId/ { print "'$4'" }" pattIndex.dat`
confFile=src/L1Trigger/HardwareValidation/hwtest/$conf.cfg
if [ -e $CMSSW_BASE/$confFile ]; then 
    cp $CMSSW_BASE/$confFile .
    echo "Copying $CMSSW_BASE/$confFile to local directory"
else 
    if [ -e $CMSSW_RELEASE_BASE/$confFile ]; then 
	cp $CMSSW_RELEASE_BASE/$confFile .
	echo "Copying $CMSSW_RELEASE_BASE/$confFile to local directory"
    else echo "Could not find $confFile in local area or release. Did you do eval scram ru -sh?"
    fi
fi

# replace filename placeholders
mv $conf.cfg $conf.tmp
sed -e "s/PATTERN/$pattName/" -e "s/ID/$pattId/" -e "s/RUN/$run/" < $conf.tmp > $conf.cfg
rm $conf.tmp


