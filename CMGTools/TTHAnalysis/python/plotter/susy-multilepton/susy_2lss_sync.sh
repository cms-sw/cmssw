#!/bin/bash

#T="~/w/TREES_72X_SYNC"
#CORE="-P $T --s2v --tree treeProducerSusyMultilepton "
#CORE="${CORE} -F sf/t $T/1_susyVars_2lssInc_v0/evVarFriend_{cname}.root -X exclusive --mcc bins/susy_2lssinc_lepchoice_sync.txt"
T=" /data/g/gpetrucc/TREES_72X_040115/9_skim_2lssSR_sync_v1"
CORE="-P $T --s2v --tree treeProducerSusyMultilepton "
CORE="${CORE} -F sf/t $T/0_allfriends/evVarFriend_{cname}.root -X exclusive --mcc bins/susy_2lssinc_lepchoice_sync.txt"

POST="";
if [[ "$1" == "mccounts" ]]; then
    GO="python mcAnalysis.py $CORE mca-Phys14.txt bins/susy_2lss_sync.txt -p T1tttt_HM -f -G -u "
    POST="| awk '/all/{print \$2}' "
elif [[ "$1" == "mcyields" ]]; then
    GO="python mcAnalysis.py $CORE mca-Phys14.txt bins/susy_2lss_sync.txt -p 'T1tttt_HM,T5tttt_MLDx,T5qqqqWW_HM,T5qqqqWWD,TTX,WZ,TT,WJets,DY' --pgroup TTX=TT[WZH] -f -G -l 4.0"
elif [[ "$1" == "mcdumps" ]]; then
    FMT='{run:1d} {lumi:9d} {evt:12d}\t{nLepGood10:2d}\t{LepGood1_pdgId:+2d} {LepGood1_pt:5.1f}\t{LepGood2_pdgId:+2d} {LepGood2_pt:5.1f}\t{nJet40}\t{nBJetMedium40:2d}\t{met_pt:5.1f}\t{htJet40j:6.1f}'
    python mcDump.py $CORE  mca-Phys14.txt bins/susy_2lss_sync.txt -p T1tttt_HM -X lep1_pt25 -X lep2_pt25   | sort -n -k1 -k2 > 2lssInc_all.txt
    python mcDump.py $CORE  mca-Phys14.txt bins/susy_2lss_sync.txt -p T1tttt_HM -X lep1_pt25 -X lep2_pt25 -X 'lep id' -X 'lep iso' -X 'lep dxy' -X 'ele cuts' $FMT  | sort -n -k1 -k2 > 2lssInc_all_relaxLept.txt
    wc -l  2lssInc_all.txt  2lssInc_all_relaxLept.txt
    exit;
else
    echo "I don't know what you want"
    exit;
fi

SAVE="${GO}"
for LL  in ee em mm; do 
for SR  in 00 10 20 30; do 
for LPt in hh hl ll; do

GO="${SAVE}"
case $SR in
0)   GO="${GO} -R nBjet nBjet0 'nBJetMedium25 >= 0' " ;;
00)  GO="${GO} -R nBjet nBjet0 'nBJetMedium25 == 0' " ;;
10)  GO="${GO} -R nBjet nBjet1 'nBJetMedium25 == 1' " ;;
20)  GO="${GO} -R nBjet nBjet2 'nBJetMedium25 == 2' " ;;
30)  GO="${GO} -R nBjet nBjet3 'nBJetMedium25 >= 3' " ;;
0[1-9X])  GO="${GO} -R nBjet nBjet0 'nBJetMedium25 == 0' -R met metSR 'met_pt > 50' -R ht htSR 'htJet40j > 200'" ;;
1[1-9X])  GO="${GO} -R nBjet nBjet1 'nBJetMedium25 == 1' -R met metSR 'met_pt > 50' -R ht htSR 'htJet40j > 200'" ;;
2[1-9X])  GO="${GO} -R nBjet nBjet2 'nBJetMedium25 == 2' -R met metSR 'met_pt > 50' -R ht htSR 'htJet40j > 200'" ;;
3[1-9X])  GO="${GO} -R nBjet nBjet3 'nBJetMedium25 >= 3' -R met metSR 'met_pt > 50' -R ht htSR 'htJet40j > 200'" ;;
esac;
case $LL in
ee)  GO="${GO} -R anyll ee 'abs(LepGood1_pdgId) == 11 && abs(LepGood2_pdgId) == 11' " ;;
em)  GO="${GO} -R anyll em 'abs(LepGood1_pdgId) != abs(LepGood2_pdgId)' " ;;
mm)  GO="${GO} -R anyll mm 'abs(LepGood1_pdgId) == 13 && abs(LepGood2_pdgId) == 13' " ;;
esac;
case $LPt in
hl)  GO="${GO} -I lep2_pt25" ;;
ll)  GO="${GO} -I lep1_pt25 -X lep2_pt25" ;;
ii)  GO="${GO} -X lep1_pt25 -X lep2_pt25" ;;
esac;

echo "echo; echo \" ===== SR $SR $LL $LPt ===== \""
echo "$GO $POST"

done
done
done
