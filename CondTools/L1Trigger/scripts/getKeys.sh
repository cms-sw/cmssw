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
#  PASSWORD_FILE=$HOME/secure/$USER.txt
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

if [ ${rflag} -eq 1 ]
    then
    GT_RS_FINAL_OR_ALGO_KEY=`getColumnFromL1Key GT_RS_FINAL_OR_ALGO_KEY`
    GT_RS_FINAL_OR_TECH_KEY=`getColumnFromL1Key GT_RS_FINAL_OR_TECH_KEY`
    GT_RS_VETO_TECH_KEY=`getColumnFromL1Key GT_RS_VETO_TECH_KEY`
    GT_RS_KEY=`getColumnFromL1Key GT_RS_KEY`
    GMT_RS_KEY=`getColumnFromL1Key GMT_RS_KEY`
    GCT_RS_KEY=`getColumnFromL1Key GCT_RS_KEY`
    RCT_RS_KEY=`getColumnFromL1Key RCT_RS_KEY`
    DTTF_RS_KEY=`getColumnFromL1Key DTTF_RS_KEY`
    echo "L1GtTriggerMaskAlgoTrigRcdKey=$GT_RS_FINAL_OR_ALGO_KEY L1GtTriggerMaskTechTrigRcdKey=$GT_RS_FINAL_OR_TECH_KEY L1GtTriggerMaskVetoTechTrigRcdKey=$GT_RS_VETO_TECH_KEY L1GtPrescaleFactorsAlgoTrigRcdKey=$GT_RS_KEY L1GtPrescaleFactorsTechTrigRcdKey=$GT_RS_KEY L1MuGMTChannelMaskRcdKey=$GMT_RS_KEY L1GctChannelMaskRcdKey=$GCT_RS_KEY L1RCTChannelMaskRcdKey=$RCT_RS_KEY L1MuDTTFMasksRcdKey=$DTTF_RS_KEY"
fi

if [ ${gflag} -eq 1 ]
    then
    GT_RS_FINAL_OR_ALGO_KEY=`getColumnFromL1Key GT_RS_FINAL_OR_ALGO_KEY`
    GT_RS_FINAL_OR_TECH_KEY=`getColumnFromL1Key GT_RS_FINAL_OR_TECH_KEY`
    GT_RS_VETO_TECH_KEY=`getColumnFromL1Key GT_RS_VETO_TECH_KEY`
    GT_RS_KEY=`getColumnFromL1Key GT_RS_KEY`
    echo "L1GtTriggerMaskAlgoTrigRcdKey=$GT_RS_FINAL_OR_ALGO_KEY L1GtTriggerMaskTechTrigRcdKey=$GT_RS_FINAL_OR_TECH_KEY L1GtTriggerMaskVetoTechTrigRcdKey=$GT_RS_VETO_TECH_KEY L1GtPrescaleFactorsAlgoTrigRcdKey=$GT_RS_KEY L1GtPrescaleFactorsTechTrigRcdKey=$GT_RS_KEY"
fi

exit 0

EOF