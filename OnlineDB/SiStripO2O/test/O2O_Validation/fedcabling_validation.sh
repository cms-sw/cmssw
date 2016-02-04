#!/bin/bash
# start the Validation of FEDCabling
eval `scramv1 runtime -sh`

### check if everything needed to connect to DB is there
if [ "$CONFDB" == ""  ];
then echo "\$CONFDB not set, please set it before You continue"; exit 0;
else echo "\$CONFDB="$CONFDB;
fi 

if [ "$TNS_ADMIN" == ""  ];
then echo "\$TNS_ADMIN not set, please set it before You continue"; exit 0;
else echo "\$TNS_ADMIN="$TNS_ADMIN;
fi 

if [ `ps aux |grep 10121|wc -l` -lt 2 ];
then echo "No Tunnel to cmsusr active, please activate before starting!"; exit 0;
fi

#describe what this script does
echo -e "\n-------------------------------------------------------------------------------------"
echo "#This scripts validates the SiStripFedCabling O2O";
echo "#It awaits a sqlite.db file and assumes that only one Tag is there for the FedCabling";
echo "#If this is not the case please change inline the sqlite_partition variable!";
echo -e "-------------------------------------------------------------------------------------\n"


#needed infos to run script
if [ $# -lt 3 ];
then echo "Usage: "
     echo "./Cabling_Validation.sh \"dbfile\" runnr \"tag_orcoff\""
     exit 0;
     
fi

#set input variables to script variables
dbfile_name=$1;
runnr=$2
tag_orcoff=$3;

echo -e "Sqlite Tag for run "$runnr" is retrieved from "$dbfile_name" !\n";
sqlite_tag=`cmscond_list_iov -c sqlite_file:$dbfile_name | grep FedCabling`;


# create .py files
cat template_Validate_FEDCabling_O2O_cfg.py | sed -e "s@template_runnr@$runnr@g" | sed -e "s@template_database@sqlite_file:$dbfile_name@g" | sed -e "s@template_tag@$sqlite_tag@g">> validate_sqlite_cfg.py
cat template_Validate_FEDCabling_O2O_cfg.py | sed -e "s@template_runnr@$runnr@g" | sed -e "s@template_database@oracle://cms_orcoff_prod/CMS_COND_21X_STRIP@g" | sed -e "s@template_tag@$tag_orcoff@g">> validate_orcoff_cfg.py

#cmsRun
cmsRun validate_sqlite_cfg.py > "Reader_"$runnr"_sqlite.txt"
cmsRun validate_orcoff_cfg.py > "Reader_"$runnr"_orcoff.txt"

#check if cmsRun was ok
if [ `cat Reader_"$runnr"_sqlite.txt | grep "\[SiStripFedCablingReader::beginRun\] VERBOSE DEBUG" | wc -l` -lt 1 ]
then echo "There is a problem with cmsRun for the sqlite file: validate_sqlite_cfg.py! Please check the file";
     exit 0;
fi

if [ `cat Reader_"$runnr"_orcoff.txt | grep "\[SiStripFedCablingReader::beginRun\] VERBOSE DEBUG" | wc -l` -lt 1 ]
then echo "There is a problem with cmsRun for the orcoff file: validate_orcoff_cfg.py! Please check the file ";
     exit 0;
fi

#Validation procedure
if [ `diff "Reader_"$runnr"_sqlite.txt" "Reader_"$runnr"_orcoff.txt" | grep ">  DcuId"| sort -u | wc -l` -lt 1 ]; 
     then if [ `diff "Reader_"$runnr"_sqlite.txt" "Reader_"$runnr"_orcoff.txt" | grep "<  DcuId"| sort -u |wc -l` -lt 1 ]; 
          then echo -e '\033[1;32m'"No Difference between OrcOff FEDCabling and sqlite FEDCabling, O2O was successful!!!"`tput sgr0`;
     fi;
else echo -n -e '\033[1;31m'"File Reader_"$runnr"_orcoff.txt contains ";
     echo -n `diff Reader_"$runnr"_orcoff.txt Reader_"$runnr"_sqlite.txt | grep ">  DcuId"| sort -u | wc -l`;
     echo " differing lines! Check Your O2O !!!"`tput sgr0`;
     echo -n -e '\033[1;31m'"File Reader_"$runnr"_sqlite.txt contains ";
     echo -n `diff Reader_"$runnr"_sqlite.txt Reader_"$runnr"_orcoff.txt | grep "<  DcuId"| sort -u | wc -l`;
     echo " differing lines! Check Your O2O !!!"`tput sgr0`;
     echo "Attaching diff to File: dcudetid_diff_"$runnr".txt!!!" ;
     touch dcudetid_diff_$runnr.txt;
     for i in  `diff Reader_"$runnr"_orcoff.txt Reader_"$runnr"_sqlite.txt | grep DcuId| sort -u`;
       do echo $i >> dcudetid_diff_$runnr.txt;
     done;
fi;

#clean up 
rm "Reader_"$runnr"_sqlite.txt";
rm "Reader_"$runnr"_orcoff.txt";
rm validate_sqlite_cfg.py;
rm validate_orcoff_cfg.py;

