#!/bin/bash
eval `scramv1 runtime -sh`
#needed infos to run script
if [ $# -lt 3 ];
then echo "Usage: "
     echo "./run_full_validation.sh \"tag_orcoff\" \"tag_sqlite\" \"dbfile\""
     echo "TAGs without leading CondDB!!!"
     exit 0;
     
fi

#input variables
tag1=$1;
tag2=$2;
dbfile=$3;

### FedCabling Validation
echo "Beginning FedCabling validation!";
echo -e "++++++++++++++++++++++++++++++++++++++\n";

for i in `ls CondDB_$tag1* | sed -e s@_@.@g | awk -F'.' '{print $6}'`
do
echo -e "Doing pedestal and noise validation for run $i:\n"
./fedcabling_validation.sh $dbfile $i SiStripFedCabling_$tag1 > validation_fedcabling_$i.txt
if [ `cat validation_fedcabling_$i.txt | grep "O2O was successful:" | wc -l` -gt 0 ]
 then  echo -e '\033[1;32m'"FEDCabling Validation successful!"`tput sgr0`
else echo -e '\033[1;31m'"FEDCabling Cabling has changed, please check output!"`tput sgr0`
fi
echo -e "\n"
done


### Ped&Noise Validation
echo "Beginning pedestal & noise validation!"
echo -e "++++++++++++++++++++++++++++++++++++++\n"


make

for i in `ls CondDB_$tag1* | sed -e s@_@.@g | awk -F'.' '{print $6}'`
do
echo -e "Doing pedestal and noise validation for run $i:\n"
./pednoise_validation $i $i CondDB_$tag1 CondDB_$tag2 > validation_pednoise_$i.txt
sum=0;
  for j in `cat validation_pednoise_$i.txt | grep "Nr of missing modules:"| awk '{print $5}'`
   do
   sum=`expr $sum + $j`    
  done

  if [ $sum -lt 1 ]
      then if [ `cat  validation_pednoise_$i.txt | grep "Number of non matching DetIds:"| awk '{print $6}'` -lt 1 ]
	  then  echo -e '\033[1;32m'"[Ped&Noise Validation] Validation successful!"`tput sgr0`
      else echo -e '\033[1;31m'"[Ped&Noise Validation] Noise and/or Pedestal of some modules don't match, please check output!"`tput sgr0`
      fi
  else echo '\033[1;31m'"[Ped&Noise Validation] Some modules are missing in one file, please check output!"`tput sgr0`;
  fi
echo -e "\n";
done
