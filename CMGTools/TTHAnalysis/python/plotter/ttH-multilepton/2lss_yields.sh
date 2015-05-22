#!/bin/bash

#T="~/w/TREES_72X_SYNC"
T="/data/g/gpetrucc/TREES_72X_210315_NoIso/skim_2lss_zero"
CORE="-P $T --s2v --tree treeProducerSusyMultilepton "
CORE="${CORE} -F sf/t $T/1_lepJetReClean_v0/evVarFriend_{cname}.root "
#CORE="${CORE} -X exclusive --mcc ttH-multilepton/susy_2lssinc_oldpresel.txt"

POST="";
WHAT="$1"; shift;
if [[ "$WHAT" == "mcyields" ]]; then
    GO="python mcAnalysis.py $CORE ttH-multilepton/mca-Phys14-2lss-refsel-frmc.txt ttH-multilepton/2lss_refsel.txt -p 'TTH,TTV,Fakes_TT,FRmc_TT,Fakes_VJets,FRmc_VJets' -j 4 -G -f -l 10.0  ";
    POST="tab";
elif [[ "$WHAT" == "mcfr" ]]; then
    GO="python mcAnalysis.py $CORE ttH-multilepton/mca-Phys14-2lss-refsel-frmc.txt ttH-multilepton/2lss_refsel.txt -p 'Fakes_[lcb]_TT,FRmc_[lcb]_TT,FR1_[lcb]_TT' -j 4 -G -f -l 10.0 -e  ";
    POST="tab";
elif [[ "$WHAT" == "mcplots" ]]; then
    GO="python mcPlots.py $CORE mca-Phys14.txt ttH-multilepton/susy_2lss_refsel.txt -f -G -l 4.0  -p  T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD,TT.,TT,WJets,DY,T,TW,WZ  -j 8 -f   --showIndivSigs --noStackSig --legendWidth 0.30 --ss 5 ttH-multilepton/susy_2lss_plots.txt";
else
    echo "I don't know what you want"
    exit;
fi

SAVE="${GO}"
#for LId in cb mva06 mva08 mva06i; do
for LId in mva06i mva06ib; do
for LL  in mm ee; do 
for LPt in hh ; do
for SR  in bl bt; do 
#for LL  in mm em ee ll; do 

GO="${SAVE}"
case $SR in
bl)  GO="${GO} -I 2B " ;;
bt)  GO="${GO}       " ;;
ba)   GO="${GO} -X 2B " ;;
0b)  GO="${GO} -X 2b -R 2B 0B 'nBJetMedium25_Old == 0' " ;;
1b)  GO="${GO} -X 2b -R 2B 1B 'nBJetMedium25_Old == 1' " ;;
2b)  GO="${GO} -X 2b -R 2B 2B 'nBJetMedium25_Old == 2' " ;;
3b)  GO="${GO} -X 2b -R 2B 3B 'nBJetMedium25_Old >= 3' " ;;
esac;
case $LL in
ee)  GO="${GO} -R anyll ee 'abs(LepGood_pdgId[iL1p_Old]) == 11 && abs(LepGood_pdgId[iL2p_Old]) == 11' " ;;
em)  GO="${GO} -R anyll em 'abs(LepGood_pdgId[iL1p_Old])    !=    abs(LepGood_pdgId[iL2p_Old])'       " ;;
mm)  GO="${GO} -R anyll mm 'abs(LepGood_pdgId[iL1p_Old]) == 13 && abs(LepGood_pdgId[iL2p_Old]) == 13' " ;;
esac;
case $LPt in
hh)  GO="${GO}               " ;;
hl)  GO="${GO} -I lep2_pt20  " ;;
ll)  GO="${GO} -I lep1_pt20 -X lep2_pt20" ;;
ii)  GO="${GO} -X lep1_pt20 -X lep2_pt20" ;;
esac;
case $LId in
cb) GO="${GO//refsel/cb} " ;;
mva06) GO="${GO//refsel/mva06} " ;;
mva06i) GO="${GO//refsel/mva06i} " ;;
mva06ib) GO="${GO//refsel/mva06ib} " ;;
mva08) GO="${GO//refsel/mva08} " ;;
esac;

if [[ "${WHAT}" == "mcplots" || "${WHAT}" == "mcrocs" ]]; then
    case $SR in
    #0[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T5qqqqWW_H.,T5qqqqWWD,T6ttWW_H.}"   ;;
    #1[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HM,T5tttt_MLDx,T6ttWW_H.,T5qqqqWW_H.,T5qqqqWWD}"   ;;
    #2[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL,T1tttt_HM,T5tttt_MLDx,T1ttbbWW_HL10,T6ttWW_H.}"   ;;
    #2[1-9X]+) GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL,T1tttt_HM,T5tttt_MLDx,T1ttbbWW_HL10,T6ttWW_H.}"   ;;
    #3[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL,T1tttt_HM,T5tttt_MLDx,T1ttbbWW_HL10,T1ttbbWW_MM5}"   ;;
    #0[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T5qqqqWW_HM,T5qqqqWWD}"   ;;
    #1[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD}"   ;;
    #2[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL,T1tttt_HM,T5tttt_MLDx}"   ;;
    #2[1-9X]+) GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL,T1tttt_HM,T5tttt_MLDx}"   ;;
    #3[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL,T1tttt_HM,T5tttt_MLDx}"   ;;
    0[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T5qqqqWW_HM.*,T5qqqqWWD,T5qqqqWWD_.*}"   ;;
    1[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HM.*,T5tttt_MLDx.*,T5qqqqWW_HM.*,T5qqqqWWD,T5qqqqWWD_.*}"   ;;
    2[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL.*,T1tttt_HM.*,T5tttt_MLDx.*}"   ;;
    2[1-9X]+) GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL.*,T1tttt_HM.*,T5tttt_MLDx.*}"   ;;
    3[1-9X])  GO=" ${GO/T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD/T1tttt_HL.*,T1tttt_HM.*,T5tttt_MLDx.*}"   ;;
    esac
    if [[ "${WHAT}" == "mcplots" ]]; then
        echo "$GO --pdir plots/72X/v2/4fb/vars/2lss_${MOD}/${LL}_pt_${LPt}/${SR}${PF}/"
    else
        echo "$GO -o plots/72X/v2/4fb/vars/2lss_${MOD}/${LL}_pt_${LPt}/${SR}${PF}/rocs.root"
    fi
else
    if [[ "$POST" == "tab" ]]; then
        PTAB="| sed -e 's#^all \|^ CUT#SR $SR${PF} $LL $LPt $MOD $LId#' | grep -v -- -----"
        echo "$GO $PTAB";
    else
        echo "echo; echo \" ===== SR $SR${PF} $LL $LPt $MOD $LId ===== \"; $GO $POST"
    fi;
fi

done
done
done
done
