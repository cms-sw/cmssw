#!/bin/bash
if [[ "$HOSTNAME" == "cmsphys05" ]]; then
    T="/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD";
    J=6;
elif [[ "$HOSTNAME" == "olsnba03" ]]; then
    T="/data/gpetrucc/TREES_250513_HADD";
    J=16;
elif [[ "$HOSTNAME" == "lxbse14c09.cern.ch" ]]; then
    T="/var/ssdtest/gpetrucc/TREES_250513_HADD";
    J=10;
else
    T="/afs/cern.ch/work/g/gpetrucc/TREES_250513_HADD";
    J=4;
fi

OPTIONS="-f -G -j 0 -P $T  -l 19.6   --FM sf/t $T/0_SFs_v2/sfFriend_{cname}.root  -p (TT.|WZ|ZZ)(_btag_.*)?"
BLoose=" -I 2B "
BTight="  "
BAny=" -X 2B "

while [[ "$1" != "" ]]; do
    if echo $1 | grep -q 2lss; then
        OPT2L="${OPTIONS}  -W puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l";

        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BAny   ) | tee btagYields.$1.txt;
        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BTight ) | tee btagYields.$1.btight.txt;
        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BLoose ) | tee btagYields.$1.bloose.txt;

    elif echo $1 | grep -q 3l; then
        OPT2L="${OPTIONS}  -W puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l";

        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BAny    ) | tee btagYields.$1.txt;
        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BTight  ) | tee btagYields.$1.btight.txt;
        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BLoose  ) | tee btagYields.$1.bloose.txt;
    elif echo $1 | grep -q 4l; then
        OPT2L="${OPTIONS}  -W puWeight*Eff_4lep*SF_btag*SF_LepMVALoose_4l";

        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BAny    ) | tee btagYields.$1.txt;
        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BTight  ) | tee btagYields.$1.btight.txt;
        (cd ../../python/plotter; python mcAnalysis.py syst/mca-expSyst.txt bins/$1.txt $OPT2L $BLoose  ) | tee btagYields.$1.bloose.txt;
     fi;
    shift;
done

