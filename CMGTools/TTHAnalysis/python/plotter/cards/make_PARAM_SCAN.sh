#!/bn/bash

STEPS="10000"; if [[ "$1" == "-i" ]]; then STEPS="$2"; shift; shift; fi;
JOBS="20"; if [[ "$1" == "-j" ]]; then JOBS="$2"; shift; shift; fi;
INDEX="$1"; STRIDE=1
if [[ "$INDEX" == "" ]]; then echo "Usage: $0 index mass [what ]"; exit 1; fi;
if seq 0 $(( $JOBS - 1 )) | grep -q $INDEX; then
    STRIDE=$(( $STEPS / $JOBS ))
    echo "Will run job $INDEX of $JOBS, processing $STRIDE points.";
    shift; 
else
    echo "Usage: $0 index mass [what ]"; exit 1; 
fi;

WHAT="SCAN"
POST=".$INDEX"
OPT="--algo=grid --points=$STEPS --firstPoint=$(( $INDEX * $STRIDE)) --lastPoint=$(( ($INDEX+1)*$STRIDE - 1))"

if [[ "$1" == "--exp" ]]; then
    OPT="${OPT} -t -1 --expectSignal=1  --X-rtd TMCSO_AdaptivePseudoAsimov"
    WHAT="SCANES"
    shift;
fi
if [[ "$1" == "--exp0" ]]; then
    OPT="${OPT} -t -1 --expectSignal=0.5  --X-rtd TMCSO_AdaptivePseudoAsimov"
    WHAT="SCANES"
    shift;
fi


if [[ "$1" == "--noTHU" ]]; then
    OPT="${OPT} --freezeNuisances=QCDscale_ggH,QCDscale_VH,QCDscale_qqH,QCDscale_ttH"
    WHAT="${WHAT}_NOTHU"
    shift;
fi

if [[ "$1" == "--1d-fix"  ]]; then
    OPT="${OPT} -P $2 --floatOtherPOI=0";
    WHAT=${WHAT}_1D_$2;
    shift; shift;
elif [[ "$1" == "--1d-float"  ]]; then
    WHAT=${WHAT}_1D_$2;
    OPT="${OPT} -P $2 --floatOtherPOI=1";
    shift; shift;
elif [[ "$1" == "--2d-fix"  ]]; then
    WHAT=${WHAT}_2D_$2_$3;
    OPT="${OPT} -P $2 -P $3  --floatOtherPOI=0";
    shift; shift; shift
elif [[ "$1" == "--2d-float"  ]]; then
    WHAT=${WHAT}_2D_$2_$3;
    OPT="${OPT} -P $2 -P $3  --floatOtherPOI=1";
    shift; shift; shift
fi;


if [[ "$1" == "--fast" ]]; then 
    WHAT=${WHAT}_FAST; 
    OPT="${OPT} --fastScan"; 
    shift; 
elif [[ "$1" == "--3x3" ]]; then 
    OPT="${OPT/--algo=grid/--algo=grid3x3}"
    shift; 
fi;


#if test -d $1; then PURPOSE=$1; else echo "Usage: $0 purpose mass what"; exit 1; fi;
#cd $PURPOSE; shift;
#if test -d $1; then MASS=$1; else echo "Usage: $0 purpose mass what "; exit 1; fi; 
#cd $MASS; shift;
MASS=125.7;

WHO=$1;
NAM=$(echo $1 | sed -e s/comb_*// -e s/.root//   | tr '[a-z]' '[A-Z]' | tr '.' '_')
WORKSPACE="$1"
shift

if test -f $WORKSPACE; then
     test -f ${WORKSPACE/.root/.log}.$WHAT$POST && rm ${WORKSPACE/.root/.log}.$WHAT$POST;
     [[ "$COMBINE_NO_LOGFILES" != "1" ]] && DO_LOG="tee -a ${WORKSPACE/.root/.log}.$WHAT$POST" || DO_LOG="dd of=/dev/null" 
     echo "c -M MultiDimFit $WORKSPACE -m $MASS -n ${NAM}_${WHAT}$POST $OPT $* "    | $DO_LOG; 
     combine -M MultiDimFit $WORKSPACE -m $MASS -n ${NAM}_${WHAT}$POST $OPT $* 2>&1 | $DO_LOG;
else 
    echo "Missing workspace $WORKSPACE at $MASS";
fi;
