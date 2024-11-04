#!/bin/bash

dumpHelpAndExit(){
cat << EOF

examples:
    ./genLUT.sh dumpAll card=cardPhysics.sh
    ./genLUT.sh generate card=cardPhysics.sh
    ./genLUT.sh diff conditions/newtag/newtag.xml conditions/oldtag/oldtag.xml
    ./genLUT.sh validate card=cardPhysics.sh

EOF
exit 0
}

FullPath=`pwd -P`
BaseDir=${FullPath#${CMSSW_BASE}/src/}
CondDir=conditions
templatefile=template.py

inputConditions=(ElectronicsMap LutMetadata LUTCorrs QIETypes QIEData SiPMParameters TPParameters TPChannelParameters ChannelQuality Gains Pedestals EffectivePedestals PedestalWidths EffectivePedestalWidths RespCorrs L1TriggerObjects)



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
    CheckParameter Run
    CheckParameter GlobalTag

    dumpCmd="cmsRun $CMSSW_RELEASE_BASE/src/CondTools/Hcal/test/runDumpHcalCond_cfg.py geometry=DB prefix="""
    PedSTR='Pedestal'
    PedWidthSTR='PedestalWidths'
    EffSTR='Effective'

    if [ -z $frontier ]
    then
	frontier="frontier://FrontierProd/CMS_CONDITIONS"
    fi

    if [ ! -z $tag ]
    then
        if [[ ${record} == *"$EffSTR"* ]]; then
	    if [[ ${record} == *"$PedWidthSTR"* ]]; then
                if ! $dumpCmd dumplist=$record run=$Run globaltag=$GlobalTag  frontierloc=$frontier frontierlist=HcalPedestalWidthsRcd:effective:$tag
                then
                    exit 1
                fi
	    elif [[ ${record} == *"$PedSTR"* ]]; then
                if ! $dumpCmd dumplist=$record run=$Run globaltag=$GlobalTag  frontierloc=$frontier frontierlist=HcalPedestalsRcd:effective:$tag
                then
                    exit 1
                fi
	    fi
	else
	    if ! $dumpCmd dumplist=$record run=$Run globaltag=$GlobalTag  frontierloc=$frontier frontierlist=Hcal${record}Rcd:$tag  
	    then
	        exit 1
	    fi
	fi
    else 
	if ! $dumpCmd dumplist=$record run=$Run globaltag=$GlobalTag 
	then
	    exit 1
	fi
    fi

    mkdir -p $CondDir/$record
    mv ${record}_Run$Run.txt $CondDir/$record/.
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
	record=$i
	tag=${!i}
	dump 
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
    sed -i 's:UTF-8:ISO-8859-1:g' $CondDir/$Tag/${Tag}.xml
    sed -i 's:"no" :'\''no'\'':g' $CondDir/$Tag/${Tag}.xml
    sed -i '/^$/d' $CondDir/$Tag/${Tag}.xml
    mv *$Tag*.{xml,dat} $CondDir/$Tag/Debug

    echo "-------------------"
    echo "-------------------"
    echo "Creating Trigger Key..."
    echo 

    HcalInput=( "${inputConditions[@]/#/Hcal}" )
    declare -A tagMap
    eval $(conddb list $GlobalTag | grep -E "$(export IFS="|"; echo "${HcalInput[*]}")" | \
	awk '{if($1~/^HcalPed/ && $2=="effective") print "tagMap["$1"+"$2"]="$3; else print "tagMap["$1"]="$3}')

    EffSTR='Effective'
    individualInputTags=""
    for i in ${inputConditions[@]}; do
	t=$i
	v=${!t}
        if [[ -z $v ]]; then
            if [[ ${i} == *"$EffSTR"* ]]; then
                v=${tagMap[Hcal${i:9}Rcd+effective]}
                l="effective"
            else
                v=${tagMap[Hcal${i}Rcd]}
                l=""
            fi
        else
            if [[ ${i} == *"$EffSTR"* ]]; then
                l="effective"
            else
                l=""
	    fi
	fi

	if ! [[ -z $v ]]; then
	    individualInputTags="""$individualInputTags
    <Parameter type=\"string\" name=\"$t\" label=\"$l\">$v</Parameter>"""
        fi
    done

    dd=$(date +"%Y-%m-%d %H:%M:%S")
    echo """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>
<CFGBrickSet>
<CFGBrick>
    <Parameter type=\"string\" name=\"INFOTYPE\">TRIGGERKEY</Parameter>
    <Parameter type=\"string\" name=\"CREATIONSTAMP\">$dd</Parameter>
    <Parameter type=\"string\" name=\"CREATIONTAG\">$Tag</Parameter> 
    <Parameter type=\"string\" name=\"HCAL_LUT_TAG\">$Tag</Parameter>  
    <Parameter type=\"boolean\" name=\"applyO2O\">$applyO2O</Parameter>  
    <Parameter type=\"string\" name=\"CMSSW\">$CMSSW_VERSION</Parameter> 
    <Parameter type=\"string\" name=\"InputRun\">$Run</Parameter> 
    <Parameter type=\"string\" name=\"GlobalTag\">$GlobalTag</Parameter>$individualInputTags 
    <Data elements=\"1\">1</Data> 
</CFGBrick>
</CFGBrickSet>""" > $CondDir/$Tag/tk_$Tag.xml
    
elif [[ "$cmd" == "validate" ]]
then
    CheckParameter card
    CheckFile $card
    source $card

    echo "Comparing input and re-generated L1TriggerObjects files"
    diff $CondDir/L1TriggerObjects/L1TriggerObjects_Run$Run.txt $CondDir/$Tag/Deploy/Gen_L1TriggerObjects_$Tag.txt 

    #parse LUT xml file and check that changes are consistent with changes in input conditions
    runs=$OldRun,$Run
    mkdir -p $CondDir/$Tag/Figures
    cmsRun PlotLUT.py globaltag=$GlobalTag run=$Run \
	inputDir=$BaseDir/$CondDir plotsDir=$CondDir/$Tag/Figures/ \
	tags=$OldTag,$Tag gains=$runs respcorrs=$runs pedestals=$runs effpedestals=$runs quality=$runs 

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

else
    dumpHelpAndExit
fi


