#!/bin/tcsh
set time=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep last_modification_date |  awk -F : '{print $2}'`


set nevents=`dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=316636 | grep file.name, file.size, file.date" | awk '{print $3}' | awk -F , '{for ( k=1 ; k <= NF ; k++ ) {print $k"\n"}}' | grep nevents |  awk -F : '{print $2}'`

echo ${time} ${nevents}

