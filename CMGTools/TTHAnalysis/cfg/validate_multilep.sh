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
    if test -d $name/*_Chunk0; then (cd $name; haddChunks.py -c .); fi; 
    echo "Run done. press enter to continue (ctrl-c to break)";
    read DUMMY;
}
function do_plot {
    PROC=$1; PROCR=$2; 
    if [[ "${PROCR}" == "" ]]; then return; fi;
    if test \! -d ${DIR}/${PROC}; then echo "Did not find ${PROC} in ${DIR}"; exit 1; fi
    test -L ${DIR}/Reference_${PROCR} || ln -sd $PWD/Reference_${PROCR} ${DIR}/ -v;
    ( cd ../python/plotter;
      # ---- MCA ---
      MCA=susy-multilepton/validation_mca.txt
      # ---- CUT FILE ---
      CUTS=susy-multilepton/validation.txt
      test -f susy-multilepton/validation-${PROC}.txt && CUTS=susy-multilepton/validation-${PROC}.txt
      python mcPlots.py -f --s2v --tree treeProducerSusyMultilepton  -P ${DIR} $MCA $CUTS ${CUTS/.txt/_plots.txt} \
              --pdir plots/74X/validation/${PROCR} -p ref_${PROC},${PROC} -u -e \
              --plotmode=nostack --showRatio --maxRatioRange 0.65 1.35 --flagDifferences
    );
}


case $WHAT in
    72X)
        $RUN && do_run $DIR -o test=1 -N 2000;
        do_plot TTH 72X_TTH
        ;;
    Data)
        echo "Test for Data not implemented";
        ;;
    MC)
        $RUN && do_run $DIR -o test=74X-MC -o sample=TT -o all;
        do_plot TT_bx25 TT_bx25
        ;;
    *)
        echo "Test for $WHAT not implemented";
        ;;
esac;
