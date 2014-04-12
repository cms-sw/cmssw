#!/bin/bash

    export dqmFileName=$1
    export Run_numb=$2

    export PREDQMDATASET=${dqmFileName#*__}
    export DQMDATASET=`echo ${PREDQMDATASET%%.*} | sed 's/__/\//g' | sed 's/^/\//'` 
    export RECODATASET=`echo ${PREDQMDATASET%%.*} | sed 's/__/\//g' | sed 's/^/\//' | sed 's/DQM/RECO/'`
    export FEVTDATASET=`echo ${PREDQMDATASET%%.*} | sed 's/__/\//g' | sed 's/^/\//' | sed 's/DQM/FEVT/'`
#    echo $DQMDATASET
#    echo $RECODATASET
#    echo $FEVTDATASET
#    export EDMDQMFILES=`dbs lsf --path="$DQMDATASET" --run=$Run_numb --site="T2_CH_CERN" | grep "/store"`
#    export EDMRECOFILES=`dbs lsf --path="$RECODATASET" --run=$Run_numb --site="T2_CH_CERN" | grep "/store"`
#    export EDMFEVTFILES=`dbs lsf --path="$FEVTDATASET" --run=$Run_numb --site="T2_CH_CERN" | grep "/store"`
    export EDMDQMFILES=`dbs --search --query="find file where dataset=$DQMDATASET and run=$Run_numb" | grep "/store"`
    FIRSTEDMDQMFILE=`expr "$EDMDQMFILES" : '\(/store[A-Za-z0-9_/.\-]*root\)'`
    export EDMDQMFILE=`dbs --search --query="find file where file=${FIRSTEDMDQMFILE} and site=T2_CH_CERN" | grep "/store"`
    if [ "$EDMDQMFILE" == "" ];  then
      export EDMRECOFILES=`dbs --search --query="find file where dataset=$RECODATASET and run=$Run_numb" | grep "/store"`
      FIRSTEDMRECOFILE=`expr "$EDMRECOFILES" : '\(/store[A-Za-z0-9_/.\-]*root\)'`
      export EDMRECOFILE=`dbs --search --query="find file where file=${FIRSTEDMRECOFILE} and site=T2_CH_CERN" | grep "/store"`
      if [ "$EDMRECOFILE" == "" ];  then
        export EDMFEVTFILES=`dbs --search --query="find file where dataset=$FEVTDATASET and run=$Run_numb" | grep "/store"`
        FIRSTEDMFEVTFILE=`expr "$EDMFEVTFILES" : '\(/store[A-Za-z0-9_/.\-]*root\)'`
        export EDMFEVTFILE=`dbs --search --query="find file where file=${FIRSTEDMFEVTFILE}  and site=T2_CH_CERN" | grep "/store"`
      fi
    fi
#    echo $EDMFEVTFILE
#    echo $EDMRECOFILE
#    echo $EDMDQMFILE
    for EDMFILETMP in $EDMFEVTFILE $EDMRECOFILE $EDMDQMFILE 
      do 
      export EDMFILE=$EDMFILETMP
    done

    export COMPLETEEDMFILE=$EDMFILE
#    echo $COMPLETEEDMFILE
    export GTNAMES=`edmProvDump $COMPLETEEDMFILE | grep -o --regexp "[A-Z0-9_]*::All"` 
    for GTNAME in $GTNAMES
      do 
      export FINALGTNAME=$GTNAME
    done
    echo $FINALGTNAME
