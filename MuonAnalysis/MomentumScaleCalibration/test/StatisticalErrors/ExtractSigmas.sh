#!/bin/bash

if [ $# -lt 2 ]
    then
    echo "Error: first and last line requested."
    exit
fi

if [ $1 -gt $2 ] 
    then
    echo "Error: the second parameter must be >= than the first."
    exit
fi

isCmsswEnv=`echo $PATH | awk -F: '{print $1}' | grep CMSSW`

if [ "$isCmsswEnv" == "" ]
    then
    echo "Warning: CMSSW environment not set."
    echo "         Trying \"cmsenv\" here."
    eval `scramv1 runtime -sh`
    isCmsswEnv=`echo $PATH | awk -F: '{print $1}' | grep CMSSW`
    if [ "$isCmsswEnv" == "" ]
	then
	echo "Error: \"cmsenv\" did not work, this is not a CMSSW workarea."
	echo "       Move to some CMSSW workarea and type \"cmsenv\"."
	exit
    fi
fi

if [ $# -eq 3 ]
    then
    dir=$3
else
    dir=""
fi

ii=$1
#numpar=0

if [ -f Values.txt ]
    then
    rm Values.txt
fi

if [ -f Sigmas.txt ] 
    then
    rm Sigmas.txt
fi

touch Sigmas.txt

first=1

while [ $ii -le $2 ]
  do
  ./TakeParameterFromBatch.sh $ii $dir
  if [ ! -f Values.txt ] 
      then
      echo "Warning: no parameters at line "$ii"."
  else
      numpar=$(sed -n "1p" Values.txt | awk '{print $2}')
      root -l -b -q MakePlot.C > OutputFit_param_${numpar}.txt
      grep sigma_final OutputFit_param_${numpar}.txt | awk '{print $2}' >> Sigmas.txt
#      if [ -f plot_param_x.gif ]
#	  then
#	  mv plot_param_x.gif plot_param_${numpar}.gif
#      fi
      rm Values.txt
  fi
  (( ii++ ))
done

exit

