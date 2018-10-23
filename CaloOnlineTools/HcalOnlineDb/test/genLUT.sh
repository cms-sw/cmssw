#!/bin/bash

FullPath=`pwd -P`
BaseDir=${FullPath#${CMSSW_BASE}/src/}
CondDir=conditions
templatefile=template.py

inputConditions=(ElectronicsMap LutMetadata LUTCorrs QIETypes QIEData SiPMParameters TPParameters TPChannelParameters ChannelQuality Gains Pedestals RespCorrs )



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


dump(){
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
}

if [[ "$cmd" == "dump" ]]
then
    dump 

elif [[ "$cmd" == "dumpAll" ]]
then
    CheckParameter card
    CheckFile $card
    source $card
    for i in ${inputConditions[@]}; do
	t=$i
	./genLUT.sh dump run=$Run record=$t GT=$GlobalTag tag=${!t}
    done



elif [[ "$cmd" == "generate" ]]
then
    CheckParameter card
    CheckFile $card
    source $card

    CheckFile $HOAsciiInput

    rm -rf $CondDir/$Tag

    echo "genLUT.sh::generate: Preparing the configuration file..."

    sed -e "s#__LUTtag__#$Tag#g" \
    -e "s#__RUN__#$Run#g" \
    -e "s#__CONDDIR__#$BaseDir/$CondDir#g" \
    -e "s#__GlobalTag__#$GlobalTag#g" \
    -e "s#__HO_master_file__#$HOAsciiInput#g" \
    $templatefile > $Tag.py

    echo "genLUT.sh::generate: Running..."

    if ! cmsRun $Tag.py
    then
	echo "ERROR: LUT Generation has failed. Exiting..." 
	exit 1
    fi

    echo "genLUT.sh::generate: Wrapping up..."

    for file in $Tag*.xml; do mv $file $file.dat; done

    echo "-------------------"
    echo "-------------------"
    echo "Creating LUT Loader..."
    echo 
    timestamp=$(date +%s)
    flist=$(ls $Tag*_[0-9]*.xml.dat | xargs)
    if ! hcalLUT create-lut-loader outputFile="$flist" tag="$Tag" storePrepend="$description"
    then
	echo "ERROR: LUT loader generation has failed. Exiting..."
	exit 1
    fi
    echo
    echo "LUT Loader created."
    echo "-------------------"
    echo "-------------------"
    echo

    zip -j $Tag.zip *$Tag*.{xml,dat}

    mkdir -p $CondDir/$Tag
    mkdir -p $CondDir/$Tag/Deploy
    mv $Tag.zip $Tag.py Gen_L1TriggerObjects_$Tag.txt $CondDir/$Tag/Deploy

    mkdir -p $CondDir/$Tag/Debug
    hcalLUT merge storePrepend="$flist" outputFile=$CondDir/$Tag/${Tag}.xml
    mv *$Tag*.{xml,dat} $CondDir/$Tag/Debug


elif [[ "$cmd" == "validate" ]]
then
    CheckParameter card
    CheckFile $card
    source $card
    runs=$OldRun,$Run
    mkdir -p $CondDir/$Tag/Figures
    cmsRun PlotLUT.py globaltag=$GlobalTag run=$Run \
	inputDir=$BaseDir/$CondDir plotsDir=$CondDir/$Tag/Figures/ \
	tags=$OldTag,$Tag gains=$runs respcorrs=$runs pedestals=$runs quality=$runs 

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


elif [ "$cmd" == "makeTriggerKey" ]
then
    CheckParameter card
    CheckParameter l1totag 
    CheckParameter production
    CheckParameter o2oL1TriggerObjects
    CheckParameter o2oInputs
    CheckFile $card
    source $card
    tktag=HCAL_$Tag
    dd=$(date +"%Y-%m-%d %H:%M:%S")
    inputs=""
    for i in ${inputConditions[@]}; do
	t=$i
	v=${!t}
	if [[ -n $v ]]; then
           inputs="""$inputs
    <Parameter type=\"string\" name=\"$t\">$v</Parameter>"""
	fi 
    done

    echo """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>
<CFGBrickSet>
<CFGBrick>
    <Parameter type=\"string\" name=\"INFOTYPE\">TriggerKey</Parameter>
    <Parameter type=\"string\" name=\"CREATIONSTAMP\">$dd</Parameter>
    <Parameter type=\"string\" name=\"CREATIONTAG\">$tktag</Parameter> 
    <Parameter type=\"string\" name=\"HCAL_LUT_TAG\">$Tag</Parameter>  
    <Parameter type=\"boolean\" name=\"updateInputs\">$o2oInputs</Parameter>  
    <Parameter type=\"boolean\" name=\"updateL1TriggerObjects\">$o2oL1TriggerObjects</Parameter>  
    <Parameter type=\"string\" name=\"GlobalTag\">$GlobalTag</Parameter> 
    <Parameter type=\"string\" name=\"CMSSW\">$CMSSW_VERSION</Parameter> 
    <Parameter type=\"string\" name=\"InputRun\">$Run</Parameter>  
    <Parameter type=\"string\" name=\"L1TriggerObjects\">$l1totag</Parameter>$inputs
    <Data elements=\"1\">$production</Data> 
</CFGBrick>
</CFGBrickSet>""" > $CondDir/$Tag/$tktag.xml
    

##
## Unknown command
##
else
    echo "Unknown command"
    exit 1;
fi


