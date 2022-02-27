#!/bin/tcsh
#!/bin/bash

set time=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep last_modification_date |  awk -F : '{print $2}'`

set nevents=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep nevents |  awk -F : '{print $2}'`

set size=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep size |  awk -F : '{print $2}'`

set time1=`date -d @${time}`

set time2=`date -d @${time} +%Y-%m-%d:%H-%M-%S`


echo ${time} ${nevents}

echo ${time1}

echo ${time2}

echo ${time} ${nevents} ${size}

dasgoclient  --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep file.size, file.nevents, file.modification_time "  > tmp

  echo "111"
cat tmp 
set sizetmp=`cat tmp | awk '{print $1}'`
set neventstmp=`cat tmp | awk '{print $2}'`
set timetmp=`cat tmp | awk '{print $3}'`
set timetmp2=`date -d @${timetmp} +%Y-%m-%d:%H-%M-%S`
  echo "final:run=316636"
echo ${sizetmp} ${neventstmp} ${timetmp2}
  echo "222"
set nls=`dasgoclient --query="summary dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep summary.nfiles, summary.nlumis, summary.nevents" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep nlumis |  awk -F : '{print $2}'`


  echo "start:run=316700"


set time=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316700 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep last_modification_date |  awk -F : '{print $2}'`


set nevents=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316700 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep nevents |  awk -F : '{print $2}'`

echo ${time} ${nevents}




  echo "222222222222222222222222222222222222   start:run=316700"




dasgoclient  --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316700 | grep file.size, file.nevents, file.modification_time "  > tmp
cat tmp | head -n 1 
  echo "333"
set sizetmp=`cat tmp | head -n 1 | awk '{print $1}'`

  echo "444"
set neventstmp=`cat tmp | head -n 1 | awk '{print $2}'`

  echo "555"
set timetmp=`cat tmp | head -n 1 | awk '{print $3}'`


  echo "666"
set timetmp2=`date -d @${timetmp} +%Y-%m-%d:%H-%M-%S`
  echo "final:run=316700"
echo ${sizetmp} ${neventstmp} ${timetmp2}
#echo ${sizetmp} ${neventstmp} 

  echo "777"
#set nls=`dasgoclient --query="summary dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep summary.nfiles, summary.nlumis, summary.nevents" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep nlumis |  awk -F : '{print $2}'`
#echo ${nls}

#  echo "for starting:"
#for i in `cat tmp`; do
#NRUN=$i
#echo $NRUN
#done


  echo "final END"












#dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW run=316636 | grep file.size, file.nevents, file.modification_time"
#6991615   63   1527727485


#dasgoclient --query="summary dataset=/HcalNZS/Run2018A-v1/RAW run=316636 |grep summary.nlumis, summary.nevents"
#19   63 

#date -d @1527727485
#Thu May 31 02:44:45 CEST 2018

#date -d @1527727485 +%Y-%m-%d:%H-%M-%S
#2018-05-31:02-44-45

