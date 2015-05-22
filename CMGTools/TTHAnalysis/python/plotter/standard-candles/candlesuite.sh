#!/bin/bash

WHAT=$1; if [[ "$1" == "" ]]; then echo "frsuite.sh <what>"; exit 1; fi

if [[ "$HOSTNAME" == "cmsphys05" ]]; then
    T="/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD";
    J=6;
elif [[ "$HOSTNAME" == "olsnba03" ]]; then
    T="/data/gpetrucc/TREES_250513_HADD";
    J=6;
else
    #T="/afs/cern.ch/user/g/gpetrucc/ttH/TREES_250513_LITE";
    #J=10;
    T="/afs/cern.ch/user/g/gpetrucc/w/TREES_250513_HADD";
    J=5;
fi

CORE="mcPlots.py -P $T -j $J --print=png,pdf -l 19.6 --showRatio --maxRatioRange 0.4 1.7 -f "
CORE="${CORE} --FM sf/t $T/0_SFs_v2/sfFriend_{cname}.root "

ROOT="plots/250513/standard-candles/$WHAT"

case $WHAT in
zjets)
    RUN="${CORE} mca-incl.txt standard-candles/zjets.txt standard-candles/zjet-plots.txt "
    SFL="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVALoose_2l'"
    SFT="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l'"
    TIGHT=" -R 'lep MVA' mu 'min(LepGood1_mva,LepGood2_mva) > 0.7 && (abs(LepGood1_pdgId) == 13 || (LepGood1_convVeto > 0 && LepGood1_innerHits == 0)) && (abs(LepGood2_pdgId) == 13 || (LepGood2_convVeto > 0 && LepGood2_innerHits == 0))' "
    echo "python $RUN $SFL --pdir $ROOT/mm       -A 'lep MVA' mu 'abs(LepGood1_pdgId) == 13' "
    echo "python $RUN $SFL --pdir $ROOT/ee       -A 'lep MVA' mu 'abs(LepGood1_pdgId) == 11' "
    echo "python $RUN $SFT --pdir $ROOT/mm/tight -A 'lep MVA' mu 'abs(LepGood1_pdgId) == 13' $TIGHT "
    echo "python $RUN $SFT --pdir $ROOT/ee/tight -A 'lep MVA' mu 'abs(LepGood1_pdgId) == 11' $TIGHT"
;;
wz_3l)
    SFL="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVALoose_3l'"
    SFT="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    TIGHT=" -R 'lep MVA 3' mu 'min(LepGood1_mva,min(LepGood2_mva,LepGood3_mva)) > 0.7 && (abs(LepGood1_pdgId) == 13 || (LepGood1_convVeto > 0 && LepGood1_innerHits == 0)) && (abs(LepGood2_pdgId) == 13 || (LepGood2_convVeto > 0 && LepGood2_innerHits == 0)) && (abs(LepGood3_pdgId) == 13 || (LepGood3_convVeto > 0 && LepGood3_innerHits == 0))' "
    MTW40=" -A 'Z peak' mtw 'mtw_wz3l(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass,LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass,LepGood3_pt,LepGood3_eta,LepGood3_phi,LepGood3_mass,mZ1,met,met_phi) > 40' "
    RUN="${CORE} mca-incl.txt standard-candles/wz_3l.txt standard-candles/wz_3l_plots.txt "
    echo "python $RUN $SFL --pdir $ROOT/ "
    echo "python $RUN $SFT --pdir $ROOT/tight $TIGHT "
    echo "python $RUN $SFL --pdir $ROOT/mtw40 $MTW40"
    echo "python $RUN $SFT --pdir $ROOT/tight/mtw40 $MTW40 $TIGHT"
;;
ztt_emu)
    SFL="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVALoose_2l'"
    SFT="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l'"
    TIGHT=" -R 'lep MVA' mu 'min(LepGood1_mva,LepGood2_mva) > 0.7 && (abs(LepGood1_pdgId) == 13 || (LepGood1_convVeto > 0 && LepGood1_innerHits == 0)) && (abs(LepGood2_pdgId) == 13 || (LepGood2_convVeto > 0 && LepGood2_innerHits == 0))' "
    RUN="${CORE} mca-incl.txt standard-candles/ztt_emu.txt standard-candles/ztt_emu_plots.txt "
    echo "python $RUN $SFL --pdir $ROOT/ "
    echo "python $RUN $SFT --pdir $ROOT/tight $TIGHT"
;;
esac;

