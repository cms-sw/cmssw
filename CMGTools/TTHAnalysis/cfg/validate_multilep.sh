RUN=/bin/false;
if [[ "$1" == "-run" ]]; then RUN=/bin/true; shift; fi;

if [[ "$1" == "" ]]; then echo "Usage: validate_multilep.sh [ -run ] <what> [dir = 'Trash' + <what> ]"; exit 1; fi;  
WHAT=$1; shift;
DIR=$2; if [[ "$2" == "" ]]; then DIR="Trash$WHAT"; fi;
DIR=$PWD/$DIR; ## make absolute

function do_run {
    name=$1; [[ "$name" == "" ]] && return; shift;
    echo "Will run as $name";
    rm -r $name 2> /dev/null
    heppy $name run_susyMultilepton_cfg.py -p 0 -o nofetch $*
    if ls -1 $name/ | grep -q _Chunk0; then (cd $name; haddChunks.py -c .); fi; 
    echo "Run done. press enter to continue (ctrl-c to break)";
    read DUMMY;
}
function do_plot {
    PROC=$1; PROCR=$2; LABEL=$3
    if [[ "${PROCR}" == "" ]]; then return; fi;
    if test \! -d ${DIR}/${PROC}; then echo "Did not find ${PROC} in ${DIR}"; exit 1; fi
    test -L ${DIR}/Ref && rm ${DIR}/Ref    
    test -L ${DIR}/New && rm ${DIR}/New    
    if test -d ~/Reference_74X_${PROCR}${LABEL}; then
         ln -sd ~/Reference_74X_${PROCR}${LABEL} ${DIR}/Ref;
    else
         ln -sd $PWD/Reference_74X_${PROCR}${LABEL} ${DIR}/Ref;
    fi
    ln -sd ${DIR}/${PROC} ${DIR}/New
    ( cd ../python/plotter;
      # ---- MCA ---
      MCA=susy-multilepton/validation_mca.txt
      # ---- CUT FILE ---
      CUTS=susy-multilepton/validation.txt
      test -f susy-multilepton/validation-${PROC}.txt && CUTS=susy-multilepton/validation-${PROC}.txt
      echo $PROC | grep -q Run2015 && CUTS=susy-multilepton/validation-data.txt
      python mcPlots.py -f --s2v --tree treeProducerSusyMultilepton  -P ${DIR} $MCA $CUTS ${CUTS/.txt/_plots.txt} \
              --pdir plots/74X/validation/${PROCR} -p new,ref -u -e \
              --plotmode=nostack --showRatio --maxRatioRange 0.65 1.35 --flagDifferences
    );
}


case $WHAT in
    Data)
        $RUN && do_run $DIR -o test=PromptReco  -N 5000;
        do_plot DoubleMuon_Run2015B_run251252 DoubleMuon_Run2015B_run251252
        do_plot DoubleEG_Run2015B_run251252 DoubleEG_Run2015B_run251252
        ;;
    DataHS)
        $RUN && do_run $DIR -o test=PromptReco  -N 25000;
        do_plot DoubleMuon_Run2015B_run251252 DoubleMuon_Run2015B_run251252 .25k
        do_plot DoubleEG_Run2015B_run251252 DoubleEG_Run2015B_run251252 .25k
        ;;
    MC)
        $RUN && do_run $DIR -o test=74X-MC -o sample=TTLep -N 2000;
        do_plot TTLep_pow TTLep_pow
        ;;
    MCHS)
        $RUN && do_run $DIR -o test=74X-MC -o sample=TTLep -N 10000;
        do_plot TTLep_pow TTLep_pow .10k
        ;;
    MCOld)
        $RUN && do_run $DIR -o test=74X-MC -o sample=TT -o all;
        do_plot TT_bx25 TT_bx25
        ;;
    *)
        echo "Test for $WHAT not implemented";
        ;;
esac;
