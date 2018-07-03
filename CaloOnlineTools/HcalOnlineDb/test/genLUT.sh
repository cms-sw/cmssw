#!/bin/bash

FullPath=`pwd -P`
BaseDir=${FullPath#${CMSSW_BASE}/src/}
CondDir=conditions
templatefile=template.py

CheckParameter(){
    if [[ -z ${!1} ]]
    then
	echo "ERROR: Parameter '$1' is not provided."
	exit 1
    fi
}

CheckFile(){
    if [[ ! -f $1 ]]
    then
	echo "ERROR: file $1 not found."
	exit 1
    fi
}

###
###Parse input 
###

cmd=$1
if [[ -z $cmd ]]
then
    echo "ERROR: no command provided"
    exit 1
fi

echo ">>>>>>>>>>>>>>>>> genLUT::$cmd <<<<<<<<<<<<<<<<<<<<<"
for var in "$@"
do
    if [[ $var == *=* ]]
    then
	map=(${var//=/ })
	echo "${map[0]}:\"${map[@]:1}\"" 
	eval "${map[0]}=\"${map[@]:1}\""
    fi
done



###
### Run
###

if [[ "$cmd" == "dump" ]]
then
    CheckParameter record 
    CheckParameter run
    CheckParameter GT

    dumpCmd="cmsRun $CMSSW_RELEASE_BASE/src/CondTools/Hcal/test/runDumpHcalCond_cfg.py geometry=DB prefix="""

    if [ -z $frontier ]
    then
	frontier="frontier://FrontierProd/CMS_CONDITIONS"
    fi

    if [ ! -z $tag ]
    then
	if ! $dumpCmd dumplist=$record run=$run globaltag=$GT  frontierloc=$frontier frontierlist=Hcal${record}Rcd:$tag  
	then
	    exit 1
	fi
    else 
	if ! $dumpCmd dumplist=$record run=$run globaltag=$GT 
	then
	    exit 1
	fi
    fi

    mkdir -p $CondDir/$record
    mv ${record}_Run$run.txt $CondDir/$record/.



elif [[ "$cmd" == "generate" ]]
then
    CheckParameter tag 
    CheckParameter GT
    CheckParameter run 
    CheckParameter HO_master_file 
    CheckParameter comment

    CheckFile $HO_master_file

    rm -rf $CondDir/$tag

    echo "genLUT.sh::generate: Preparing the configuration file..."

    sed -e "s#__LUTtag__#$tag#g" \
    -e "s#__RUN__#$run#g" \
    -e "s#__CONDDIR__#$BaseDir/$CondDir#g" \
    -e "s#__GlobalTag__#$GT#g" \
    -e "s#__HO_master_file__#$HO_master_file#g" \
    $templatefile > $tag.py

    echo "genLUT.sh::generate: Running..."

    if ! cmsRun $tag.py
    then
	echo "ERROR: LUT Generation has failed. Exiting..." 
	exit 1
    fi

    echo "genLUT.sh::generate: Wrapping up..."

    for file in $tag*.xml; do mv $file $file.dat; done

    echo "-------------------"
    echo "-------------------"
    echo "Creating LUT Loader..."
    echo 
    timestamp=$(date +%s)
    flist=$(ls $tag*_[0-9]*.xml.dat | xargs)
    if ! hcalLUT create-lut-loader outputFile="$flist" tag="$tag" storePrepend="$comment"
    then
	echo "ERROR: LUT loader generation has failed. Exiting..."
	exit 1
    fi
    echo
    echo "LUT Loader created."
    echo "-------------------"
    echo "-------------------"
    echo

    zip -j $tag.zip *$tag*.{xml,dat}

    mkdir -p $CondDir/$tag
    mkdir -p $CondDir/$tag/Deploy
    mv $tag.zip $tag.py Gen_L1TriggerObjects_$tag.txt $CondDir/$tag/Deploy

    mkdir -p $CondDir/$tag/Debug
    hcalLUT merge storePrepend="$flist" outputFile=$CondDir/$tag/${tag}.xml
    mv *$tag*.{xml,dat} $CondDir/$tag/Debug


elif [[ "$cmd" == "validate" ]]
then
    CheckParameter GT
    CheckParameter run 
    CheckParameter tags 
    CheckParameter pedestals
    CheckParameter gains 
    CheckParameter respcorrs
    CheckParameter quality

    ptags=(${tags//,/ })
    mkdir -p $CondDir/${ptags[1]}/Figures
    cmsRun PlotLUT.py globaltag=$GT run=$run \
	inputDir=$BaseDir/$CondDir plotsDir=$CondDir/${ptags[1]}/Figures/ \
	tags=$tags gains=$gains respcorrs=$respcorrs pedestals=$pedestals quality=$quality 

elif [ "$cmd" == "upload" ]
then
    CheckParameter tag 
    CheckParameter dropbox
    lutfile=$CMSSW_BASE/src/$BaseDir/$CondDir/$tag/Deploy/$tag.zip
    CheckFile $lutfile

    scp $lutfile cmsusr:~/.

    cmd="scp $tag.zip $dropbox" 
    ssh -XY cmsusr $cmd

elif [ "$cmd" == "diff" ]
then
    if [[ $# -lt 3 ]]
    then
	echo "Bad input."
	exit 1
    fi

    CheckFile $2
    CheckFile $3
    echo $BaseDir/$2,$BaseDir/$3

    if [[ -z $verbosity ]]
    then
	verbosity=0
    fi

    hcalLUT diff inputFiles=$BaseDir/$2,$BaseDir/$3 section=$verbosity



##
## Unknown command
##
else
    echo "Unknown command"
    exit 1;
fi


