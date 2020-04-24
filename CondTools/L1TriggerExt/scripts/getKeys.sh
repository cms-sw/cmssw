#!/bin/bash

tflag=0
rflag=0
gflag=0
while getopts 'trgh' OPTION
  do
  case $OPTION in
      t) tflag=1
          ;;
      r) rflag=1
          ;;
      g) gflag=1
	  ;;
      h) echo "Usage: [-tr] L1_KEY"
          echo "  -t: print TSC key"
          echo "  -r: print RS keys"
	  echo "  -g: GT RS keys only"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

getColumnFromL1Key()
{
  COLUMN=$1

  DB="cms_omds_lb"
  USER="cms_trg_r"
  PASSWORD_FILE=/nfshome0/centraltspro/secure/$USER.txt
  PASSWORD=`cat $PASSWORD_FILE`

  RESULT=`sqlplus -s <<!
    $USER/$PASSWORD@$DB
    SET FEEDBACK OFF;
    SET HEADING OFF;
    SET LINESIZE 500;
    select $COLUMN
    from CMS_TRG_L1_CONF.L1_CONF_DETAILS_VIEW
    where L1_KEY='$L1_KEY';
    !
!`
  
  echo $RESULT
}

if [[ $# != 1 ]]; then
  echo "Usage:"
  echo "$0 <L1 key>"
  exit 1
fi

L1_KEY=$1

if [ ${tflag} -eq 1 ]
    then
    TSC_KEY=`getColumnFromL1Key TSC_KEY`
    echo ${TSC_KEY}
fi

exit 0

EOF
