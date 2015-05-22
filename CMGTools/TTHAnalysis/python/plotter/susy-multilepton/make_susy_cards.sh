#!/bin/bash

if [[ "$1" == "afs" ]]; then
    T="/afs/cern.ch/work/g/gpetrucc/TREES_72X_040115";
    J=4;
elif [[ "$HOSTNAME" == "cmsphys10" ]]; then
    T="/data/g/gpetrucc/TREES_72X_040115";
    J=8;
else
    T="/afs/cern.ch/work/g/gpetrucc/TREES_72X_040115";
    J=4;
fi

LUMI=10.0
OUTDIR="susy_cards"
OPTIONS=" -P $T -j $J -l $LUMI -f --s2v --tree treeProducerSusyMultilepton --od $OUTDIR --asimov "
#OPTIONS=" $OPTIONS -F sf/t $T/0_lepMVA_v1/evVarFriend_{cname}.root "


function makeCard_2lss {
    local EXPR=$1; local BINS=$2; local SYSTS=$3; local OUT=$4; local GO=$5

    # b-jet cuts
    case $SR in
    0[0-9X])  GO="${GO} -R nBjet nBjet0 nBJetMedium40==0 " ;;
    1[0-9X])  GO="${GO} -R nBjet nBjet1 nBJetMedium40==1 " ;;
    2[0-9X])  GO="${GO} -R nBjet nBjet2 nBJetMedium40==2 " ;;
    3[0-9X])  GO="${GO} -R nBjet nBjet3 nBJetMedium40>=3 " ;;
    2[0-9X]+)  GO="${GO} -R nBjet nBjet2 nBJetMedium40>=2 " ;;
    0[0-9X]s)  GO="${GO} -R nBjet nBjet3 nBJetMedium40+min(nBJetMedium25+nSoftBTight25-nBJetMedium40,1)==0 " ;;
    1[0-9X]s)  GO="${GO} -R nBjet nBjet3 nBJetMedium40+min(nBJetMedium25+nSoftBTight25-nBJetMedium40,1)==1 " ;;
    2[0-9X]s)  GO="${GO} -R nBjet nBjet3 nBJetMedium40+min(nBJetMedium25+nSoftBTight25-nBJetMedium40,1)==2 " ;;
    3[0-9X]s)  GO="${GO} -R nBjet nBjet3 nBJetMedium40+min(nBJetMedium25+nSoftBTight25-nBJetMedium40,1)>=3 " ;;
    esac;

    # kinematics
    case $SR in
    [0-3]X|[0-3]X+)  GO="${GO} -R met met met_pt>50 -R ht ht htJet40j>200 " ;;
    esac;

    # lepton final state
    case $LL in
    ee)  GO="${GO} -R anyll ee abs(LepGood1_pdgId)==11&&abs(LepGood2_pdgId)==11 " ;;
    em)  GO="${GO} -R anyll em abs(LepGood1_pdgId)!=abs(LepGood2_pdgId) " ;;
    mm)  GO="${GO} -R anyll mm abs(LepGood1_pdgId)==13&&abs(LepGood2_pdgId)==13 " ;;
    3l)  GO="${GO} -I exclusive -X same-sign -R anyll lep3-cuts LepGood3_relIso03<0.1&&LepGood3_tightId>(abs(LepGood3_pdgId)==11)&&LepGood3_sip3d<4&&(abs(LepGood3_pdgId)==13||(LepGood3_convVeto&&LepGood3_lostHits==0&&LepGood3_tightCharge>1))"
    esac;

    # lepton pt categories
    case $LPt in
    hl)  GO="${GO} -I lep2_pt25" ;;
    ll)  GO="${GO} -I lep1_pt25 -X lep2_pt25" ;;
    ii)  GO="${GO} -X lep1_pt25 -X lep2_pt25" ;;
    2020)  GO="${GO} -R lep1_pt25 lep2020 LepGood2_pt>20 -X lep2_pt25" ;;
    esac;

    # inclusive vs exclusive
    case $MOD in
    inc) GO="${GO} -X exclusive --mcc bins/susy_2lssinc_lepchoice.txt" ;;
    esac;

    if [[ "$PRETEND" == "1" ]]; then
        echo "making datacard $OUT from makeShapeCardsSusy.py mca-Phys14.txt bins/susy_2lss_sync.txt \"$EXPR\" \"$BINS\" $SYSTS $GO;"
    else
        echo "making datacard $OUT from makeShapeCardsSusy.py mca-Phys14.txt bins/susy_2lss_sync.txt \"$EXPR\" \"$BINS\" $SYSTS $GO;"
        python makeShapeCardsSusy.py mca-Phys14.txt bins/susy_2lss_sync.txt "$EXPR" "$BINS" $SYSTS -o $OUT $GO;
        echo "  -- done at $(date)";
    fi;
}

function combineCardsSmart {
    CMD=""
    for C in $*; do
        # missing datacards 
        test -f $C || continue;
        # datacards with no event yield
        grep -q "observation 0.0$" $C && continue
        CMD="${CMD} $(basename $C .card.txt)=$C ";
    done
    if [[ "$CMD" == "" ]]; then
        echo "Not any card found in $*" 1>&2 ;
    else
        combineCards.py $CMD
    fi
}

if [[ "$1" == "--pretend" ]]; then
    PRETEND=1; shift;
fi;
if [[ "$1" == "2lss-2012" ]]; then
    OPTIONS=" $OPTIONS -F sf/t $T/1_susyVars_2lssInc_v0/evVarFriend_{cname}.root "
    SYSTS="syst/susyDummy.txt"
    CnC_expr="1+4*(met_pt>120)+(htJet40j>400)+2*(nJet40>=4)"
    CnC_bins="[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]"
    MOD=inc;

    echo "Making individual datacards"
    for LL in ee em mm; do for LPt in 2020; do for SR in 0X 1X 2X+; do
        echo " --- CnC2012_${SR}_${LL} ---"
        #makeCard_2lss $CnC_expr $CnC_bins $SYSTS CnC2012_${SR}_${LL} "$OPTIONS";
    done; done; done
    echo "Making combined datacards"
    for D in $OUTDIR/T[0-9]*; do
        test -f $D/CnC2012_0X_ee.card.txt || continue
        (cd $D;
            for SR in 0X 1X 2X+; do
                combineCards.py CnC2012_${SR}_{ee,em,mm}.card.txt >  CnC2012_${SR}.card.txt
            done
            combineCards.py CnC2012_{0X,1X,2X+}.card.txt >  CnC2012.card.txt
        );
        echo "Made combined card $D/CnC2012.card.txt"
    done
    echo "Done at $(date)";

elif [[ "$1" == "2lss-2015" ]]; then
    OPTIONS=" $OPTIONS -F sf/t $T/1_susyVars_2lssInc_v0/evVarFriend_{cname}.root "
    SYSTS="syst/susyDummy.txt"
    CnC_expr="1+4*(met_pt>120)+(htJet40j>400)+2*(nJet40>=4)"
    CnC_bins="[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]"
    MOD=inc;

    echo "Making individual datacards"
    for LL in ee em mm; do for LPt in hh hl ll; do for SR in 0X 1X 2X 3X 2X+; do
    #for LL in ee em mm; do for LPt in hh hl ll ; do for SR in 0Xs 1Xs 2Xs 3Xs; do
        echo " --- CnC2015_${SR}_${LL}_${LPt} ---"
        makeCard_2lss $CnC_expr $CnC_bins $SYSTS CnC2015_${SR}_${LL}_${LPt} "$OPTIONS";
    done; done; done
    #exit
    echo "Making combined datacards"
    for D in $OUTDIR/T[0-9]*; do
        test -f $D/CnC2015_0X_ee_hh.card.txt || continue
        (cd $D && echo "    $D";
        for SR in 0X 1X 2X 3X 2X+; do
        #for SR in 0Xs 1Xs 2Xs 3Xs; do
            combineCardsSmart CnC2015_${SR}_{ee,em,mm}_hh.card.txt >  CnC2015_${SR}_hh.card.txt
            combineCardsSmart CnC2015_${SR}_{ee,em,mm}_{hh,hl,ll}.card.txt >  CnC2015_${SR}.card.txt
        done
        combineCardsSmart CnC2015_{0X,1X,2X+}.card.txt   >  CnC2015_2b.card.txt
        combineCardsSmart CnC2015_{0X,1X,2X+}_hh.card.txt   >  CnC2015_2b_hh.card.txt
        combineCardsSmart CnC2015_{0X,1X,2X,3X}_hh.card.txt >  CnC2015_3b_hh.card.txt
        combineCardsSmart CnC2015_{0X,1X,2X,3X}.card.txt >  CnC2015_3b.card.txt
        #combineCardsSmart CnC2015_{0Xs,1Xs,2Xs,3Xs}_hh.card.txt >  CnC2015_3bs_hh.card.txt
        #combineCardsSmart CnC2015_{0Xs,1Xs,2Xs,3Xs}.card.txt >  CnC2015_3bs.card.txt
        )
    done
    echo "Done at $(date)";

elif [[ "$1" == "2lss-2015x" ]]; then
    OPTIONS=" $OPTIONS -F sf/t $T/1_susyVars_2lssInc_v0/evVarFriend_{cname}.root "
    SYSTS="syst/susyDummy.txt"
    CnC_expr="1+4*(met_pt>120)+(htJet40j>400)+2*(nJet40>=4)"
    CnC_bins="[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]"
    MOD=excl;

    echo "Making individual datacards"
    for LL in ee em mm 3l; do for LPt in hh hl ll; do for SR in 0X 1X 2X 3X; do
        echo " --- CnC2015X_${SR}_${LL}_${LPt} ---"
        makeCard_2lss $CnC_expr $CnC_bins $SYSTS CnC2015X_${SR}_${LL}_${LPt} "$OPTIONS";
    done; done; done
    #exit
    echo "Making combined datacards"
    for D in $OUTDIR/T[0-9]*; do
        test -f $D/CnC2015X_0X_ee_hh.card.txt || continue
        (cd $D && echo "    $D";
        for SR in 0X 1X 2X 3X; do
            combineCardsSmart CnC2015X_${SR}_{ee,em,mm}_hh.card.txt >  CnC2015X_${SR}_hh.card.txt
            combineCardsSmart CnC2015X_${SR}_{ee,em,mm}_{hh,hl,ll}.card.txt >  CnC2015X_${SR}.card.txt
            combineCardsSmart CnC2015X_${SR}_{ee,em,mm,3l}_hh.card.txt >  CnC2015X_${SR}_hh_w3l.card.txt
            combineCardsSmart CnC2015X_${SR}_{ee,em,mm,3l}_{hh,hl,ll}.card.txt >  CnC2015X_${SR}_w3l.card.txt
        done
        combineCardsSmart CnC2015X_{0X,1X,2X,3X}_hh.card.txt >  CnC2015X_3b_hh.card.txt
        combineCardsSmart CnC2015X_{0X,1X,2X,3X}.card.txt >  CnC2015X_3b.card.txt
        combineCardsSmart CnC2015X_{0X,1X,2X,3X}_hh_w3l.card.txt >  CnC2015X_3b_hh_w3l.card.txt
        combineCardsSmart CnC2015X_{0X,1X,2X,3X}_w3l.card.txt >  CnC2015X_3b_w3l.card.txt
        )
    done
    echo "Done at $(date)";

fi

