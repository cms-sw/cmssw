if [[ "$1" == "-run" ]]; then
    shift;
    name="$1"; if [[ "$name" == "" ]]; then name="TrashML"; fi;
    echo "Will run as $name";
    rm -r $name 2> /dev/null
    heppy $name run_susyMultilepton_cfg.py -N 2000 -o test=1 -o nofetch  -p 0
    echo "Run done. press enter to continue (ctrl-c to break)";
    read DUMMY;
    X=$PWD/$name; 
else
    X=$PWD/$1; 
fi;

if test -d $X/TTH_Chunk0; then (cd $X; haddChunks.py -c .); fi;
if test \! -d $X/TTH; then echo "Did not find TTH in $X"; exit 1; fi
test -L $PWD/Reference_TTH || ln -sd $PWD/Reference_TTH $X/ -v;
( cd ../python/plotter; 
  python mcPlots.py -f --s2v --tree treeProducerSusyMultilepton  -P $X susy-multilepton/validation_mca.txt susy-multilepton/validation.txt susy-multilepton/validation_plots.txt --pdir plots/72X/validation -p ref_ttH,ttH -u -e --plotmode=nostack --showRatio --maxRatioRange 0.65 1.35 --flagDifferences
)
