#!/bin/bash

    ############
    ## MAIN  ###
    ############
    
    basePath=/data/local1/borgia/OffLineAnalysis/Layer
    dataPath=/data/local1/borgia/Cruzet3Data

    if [ -e  $dataPath/FileList.txt ]; then 
	rm $dataPath/FileList.txt
    fi

    export k
    export output=$basePath/output.txt
    export List=$dataPath/FileList.txt
    export outFile=$basePath/LayerTree.root
    export outFinal=$basePath/LayerTrendPlots.root

    cd ${basePath}/../CMSSW_2_0_10/src
    eval `scramv1 runtime -sh`
    cd $dataPath

    echo "...Running"
    rootFileList=(`ls -ltr ${dataPath} |  awk '{print $9}' | grep ".root"`)
    k=0
    ListSize=${#rootFileList[*]}
    while [ "$k" -lt "$ListSize" ]
     do
      rootFile=${rootFileList[$k]}
      runNumberList[$k]=${rootFile:9:5}
      if [ ! -e $basePath/Fits/Run_${runNumberList[$k]}/TIB ]; then 
	  mkdir -p $basePath/Fits/Run_${runNumberList[$k]}/TIB
      fi
      if [ ! -e $basePath/Fits/Run_${runNumberList[$k]}/TOB ]; then 
	  mkdir -p $basePath/Fits/Run_${runNumberList[$k]}/TOB
      fi	  
      if [ ! -e $basePath/Fits/Run_${runNumberList[$k]}/TID ]; then 
	  mkdir -p $basePath/Fits/Run_${runNumberList[$k]}/TID
      fi
      if [ ! -e $basePath/Fits/Run_${runNumberList[$k]}/TEC ]; then 
	  mkdir -p $basePath/Fits/Run_${runNumberList[$k]}/TEC
      fi
      echo ${runNumberList[$k]} $dataPath/${rootFileList[$k]} >> $List
      let "k+=1"
    done

    cd $basePath
    echo "root.exe -q -b -l \"$basePath/RunMain.C(\"$List\",\"$outFile\")\" > output"
    root.exe -b -q -l  "$basePath/RunMain.C(\"$List\",\"$outFile\")" > output

    echo "root.exe -b -q -l \"$basePath/RunPlotMacro.C(\"$outFile\",\"$outFinal\")"
    root.exe -b -q -l "$basePath/RunPlotMacro.C(\"$outFile\",\"$outFinal\")"

