#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 28/3/2007

function MakePlots(){

  StoreDir=$1
  echo $StoreDir

  FileNum=`ls *.root 2>/dev/null | grep -c root`

  do="true"

  if [ -e ${StoreDir}/plots.txt ] && [ `cat plots.txt` == ${FileNum} ]; then
    do="false"
  fi

  if [ `ls | grep -c root` -ne 0 ] && [ $do == "true" ]; then
    jobsList=`ls *.root 2>/dev/null | awk -F_ '{print $4}' | awk -F. '{print $1}' | tr '\n' - | sed -e 's/-*$//'`

    Name=$1_${Config}_${Flag}
    PSFile=$1_${Config}_${Flag}.ps

    echo Executing TIFmacro_chain on Run "${Flag}"
    echo "root -x -l -b -q '${local_crab_path}/TIFmacro_chain.C("'$listaFile'","'$PSFile'",true,true)' 1>output_root_macro 2>error_root_macro"
    root -x -l -b -q '${local_crab_path}/TIFmacro_chain.C("'$Name'","'$jobsList'","'$PSFile'",true,true)' 1>output_root_macro 2>error_root_macro

    echo ${FileNum} > plots.txt

  fi
}

function MergePlots(){

  StoreDir=$1
  echo $StoreDir

  [ ! -e  ${StoreDir} ] && return

  Name=$2

  FileNum=`ls ${Name}*.root 2>/dev/null | grep -c root`
  #echo FileNum $FileNum #RIM

  [ "$FileNum" == "0" ] && return

  jobsList=`ls ${Name}*.root 2>/dev/null | awk -F"${Name}_" '{print $2}' | awk -F. '{print $1}' | tr '\n' - | sed -e 's/-*$//'`
  FulljobsList=`ls ${Name}*.root 2>/dev/null | cut -d. -f1 | awk -F"${Name}_" '{if($2) print $0".root"}' | tr '\n' ' '`
  #echo joblist $jobsList #RIM
  #echo Full $FulljobsList #RIMU
  
  if [ "$FileNum" == "1" ]; then
      if [ "$jobsList" != "" ]; then
	  echo joblist $jobsList
	  mkdir -p  ${StoreDir}/Basket
	  cp ${FulljobsList} ${Name}.root
	  mv ${FulljobsList} Basket
	  mv `echo ${FulljobsList} | sed -e "s@.root@.ps@g"` Basket
	  gzip Basket/*.ps
      fi
      return
  fi

  # If there is more then one file
  
  # If the merged root file already exists add it to the list of file to be merged,
  if [ -e ${StoreDir}/${Name}.root ]; then
      mv -f ${StoreDir}/${Name}.root ${StoreDir}/${Name}_0.root
      jobsList=`echo ${jobsList}-0`
      echo "already merged file moved to ${StoreDir}/${Name}_0.root"
  fi
  
  echo "Executing Merging ClusterAnalysis hitograms for Run ${Flag}"
  echo "root -x -l -b -q \"${local_crab_path}/AddHisto.C(\"$Name\",\"$jobsList\",\"DBNoise DBPed DBBadStrip\")\" 1>output_root_macro 2>error_root_macro"
  root -x -l -b -q "${local_crab_path}/AddHisto.C(\"$Name\",\"$jobsList\",\"DBNoise DBPedestals DBBadStrips\")" 1>output_root_macro 2>error_root_macro
  exit_status=$?
  echo exit_status $exit_status
  if [ "${exit_status}" == "0" ]; then
      mkdir -p  ${StoreDir}/Basket
      [ -e ${StoreDir}/${Name}_0.root ] &&  rm ${StoreDir}/${Name}_0.root  
      mv ${FulljobsList} Basket
      mv `echo ${FulljobsList} | sed -e "s@.root@.ps@g"` Basket	
  fi
}

############
## MAIN  ###
############

Version=""
[ "$1" != "" ] && Version=$1

export outFile
basePath=/analysis/sw/CRAB
Tpath=/data1/CrabAnalysis

macroPath=${basePath}/macros

cd ${basePath}/CMSSW/CMSSW_1_3_0/src
eval `scramv1 runtime -sh`
cd -

for Type in `ls $Tpath`
  do
  for path in `ls $Tpath/$Type`
    do
    [ "$Version" != "" ] && [ "$Version" != "$path" ] && continue
  #[ "$path" != "FNAL_pre6_v17" ] && [ "$path" != "FNAL_pre6_v17" ]&& continue  
    echo "...Running on $Tpath/$path"
    for dir in `ls $Tpath/$Type/$path`
      do

    workdir=$Tpath/$Type/$path/$dir/res

    [ ! -e $workdir ] && continue 
    cd $workdir

    # Make the plots for TIFNtupleMaker
    #if [ $Type == "TIFNtupleMaker" ] || [ $Type == "TIFNtupleMakerZS" ]; then
    #  MakePlots $workdir
    #fi

    # Merge the plots for ClusterAnalysis
    if [ $Type == "ClusterAnalysis" ]; then
	MergePlots $workdir $dir
    fi
    cd -
    done
  done
done
