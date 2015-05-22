#!/bin/bash

WHAT=$1; if [[ "$1" == "" ]]; then echo "frsuite.sh <what>"; exit 1; fi

if [[ "$HOSTNAME" == "cmsphys05" ]]; then
    T="/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD";
    J=6;
elif [[ "$HOSTNAME" == "olsnba03" ]]; then
    T="/data/gpetrucc/TREES_250513_HADD";
    J=12;
elif [[ "$HOSTNAME" == "lxbse14c09.cern.ch" ]]; then
    T="/var/ssdtest/gpetrucc/TREES_250513_HADD";
    J=16;
elif [[ "$HOSTNAME" == "cmsphys10" ]]; then
    T="/data/g/gpetrucc/TREES_250513_HADD";
    J=8;
else
    T="/afs/cern.ch/work/g/gpetrucc/TREES_250513_HADD";
    J=6;
fi
CORE="mcPlots.py -P $T -j $J -f -l 19.6  "
FSF=" --FM sf/t $T/0_SFs_v2/sfFriend_{cname}.root "

#ROOT="plots/250513/v4.0/$WHAT"
ROOT="plots/250513/v4.2/$WHAT"

RUN2L="${CORE} mca-2lss-data.txt --showRatio --maxRatioRange 0 3.7 --poisson"
RUN2E="${RUN2L} bins/2lss_ee_tight.txt   bins/mvaVars_2lss.txt  "
RUN2M="${RUN2L} bins/2lss_mumu_tight.txt bins/mvaVars_2lss.txt --xp QF_data "
RUN2X="${RUN2L} bins/2lss_em_tight.txt   bins/mvaVars_2lss.txt  "
BINCL=" -X 2B"
BLOOSE=" -I 2B"
BMEDIUM=" -R 2B 2b1B '(nBJetLoose25>=2 && nBJetMedium25>=1)'"
B2LOOSE=" -R 2B 2bl '(nBJetLoose25>=2)'"
BTIGHT=" "
### =============== 2L SS, 2b loose =================
### =============== 2L SS, 2b loose =================
case $WHAT in
2lss)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    MVA="$MVA -F sf/t   /afs/cern.ch/user/g/gpetrucc/w/TREES_250513_HADD/2_finalmva_2lss_2_v1/evVarFriend_{cname}.root "

    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/em   $BINCL --rebin 2   "
    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/ee   $BINCL --rebin 4   "
    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu $BINCL --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/mumu $BLOOSE --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/em   $BTIGHT --rebin 4   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/ee   $BTIGHT --rebin 6   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/mumu $BTIGHT --rebin 4   "

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
#    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/bloose/mumu $BLOOSE --rebin 2   "
    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/btight/em   $BTIGHT --rebin 4   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/btight/ee   $BTIGHT --rebin 4   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/btight/mumu $BTIGHT --rebin 4   "

    POS=" -A pt2010 positive 'LepGood1_charge>0' "
    NEG=" -A pt2010 negative 'LepGood1_charge<0' "
    echo "python ${RUN2M} $SF $MVA --pdir $ROOT/postfit/mumu_pos/  $BINCL  --rebin 2 --sP MVA_2LSS_4j_6var $POS "
    echo "python ${RUN2M} $SF $MVA --pdir $ROOT/postfit/mumu_neg/  $BINCL  --rebin 3 --sP MVA_2LSS_4j_6var $NEG "
    echo "python ${RUN2X} $SF $MVA --pdir $ROOT/postfit/em_pos/    $BINCL  --rebin 2 --sP MVA_2LSS_4j_6var $POS "
    echo "python ${RUN2X} $SF $MVA --pdir $ROOT/postfit/em_neg/    $BINCL  --rebin 3 --sP MVA_2LSS_4j_6var $NEG "
    echo "python ${RUN2E} $SF $MVA --pdir $ROOT/postfit/ee_pos/    $BINCL  --rebin 2 --sP MVA_2LSS_4j_6var $POS "
    echo "python ${RUN2E} $SF $MVA --pdir $ROOT/postfit/ee_neg/    $BINCL  --rebin 3 --sP MVA_2LSS_4j_6var $NEG "
;;
2lss_CB)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/_tight.txt/_CB.txt}"
    RUN2M="${RUN2M/_tight.txt/_CB.txt}"
    RUN2E="${RUN2E/_tight.txt/_CB.txt}"
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataCB}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataCB}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataCB}"
    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/em   $BINCL --rebin 3   "
    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/ee   $BINCL --rebin 5   "
    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu $BINCL --rebin 3   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/mumu $BLOOSE --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/em   $BTIGHT --rebin 4   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/ee   $BTIGHT --rebin 6   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/mumu $BTIGHT --rebin 4   "

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 3   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 3   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 3   "
#    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/bloose/mumu $BLOOSE --rebin 2   "
#    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/btight/em   $BTIGHT --rebin 4   "
#    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/btight/ee   $BTIGHT --rebin 6   "
#    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/btight/mumu $BTIGHT --rebin 4   "
;;
2lss_loose)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVALoose_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss_loose-data}  -R 'lep MVA' 'lep MVAL' 'min(LepGood1_mva,LepGood2_mva) > -0.3'"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss_loose-data}  -R 'lep MVA' 'lep MVAL' 'min(LepGood1_mva,LepGood2_mva) > -0.3'"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss_loose-data}  -R 'lep MVA' 'lep MVAL' 'min(LepGood1_mva,LepGood2_mva) > -0.3'"
    
    #echo "python $RUN2M $SF $MVA --pdir $ROOT/mm/nostack $BINCL --rebin 4 --noStackSig --showSigShape"
    #echo "python $RUN2E $SF $MVA --pdir $ROOT/ee/nostack $BINCL --rebin 6 --noStackSig --showSigShape"
    #echo "python $RUN2X $SF $MVA --pdir $ROOT/em/nostack $BINCL --rebin 4 --noStackSig --showSigShape"

    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/em   $BINCL --rebin 2   "
    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/ee   $BINCL --rebin 4   "
    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu $BINCL --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/mumu $BLOOSE --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/em   $BTIGHT --rebin 4   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/ee   $BTIGHT --rebin 6   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/mumu $BTIGHT --rebin 4   "

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
#    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/bloose/mumu $BLOOSE --rebin 2   "
#    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/btight/em   $BTIGHT --rebin 4   "
#    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/btight/ee   $BTIGHT --rebin 6   "
#    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/btight/mumu $BTIGHT --rebin 4   "
;;
2lss_BCat)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCat}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCat}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCat}"

    #echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/em   $BINCL --rebin 2   "
    #echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/ee   $BINCL --rebin 4   "
    #echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu $BINCL --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/em   $BLOOSE --rebin 2   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/ee   $BLOOSE --rebin 4   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/bloose/mumu $BLOOSE --rebin 2   "
#    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/em   $BTIGHT --rebin 4   "
#    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/ee   $BTIGHT --rebin 6   "
#    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/btight/mumu $BTIGHT --rebin 4   "

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/bloose/em   $BLOOSE --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/bloose/ee   $BLOOSE --rebin 4   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/bloose/mumu $BLOOSE --rebin 2   "
    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/btight/em   $BTIGHT --rebin 4   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/btight/ee   $BTIGHT --rebin 4   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/btight/mumu $BTIGHT --rebin 4   "
;;
2lss_BCat4Plots)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCat4Plots}"

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2  "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2  "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2  "
    #echo "python ${RUN2M} $MVA $SF --pdir $ROOT/bloose/mumu $BLOOSE --rebin 2 --sP MVA_2LSS_4j_6var  "
    #echo "python ${RUN2M} $MVA $SF --pdir $ROOT/btight/mumu $BTIGHT --rebin 2  "
;;
2lss_BCatSB4Plots)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCatSB4Plots}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCatSB4Plots}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCatSB4Plots}"

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
;;

2lss_muSip4BCat)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    SIP4=" -A pt2010 sip4 '(abs(LepGood1_pdgId) != 13 || LepGood1_sip3d < 4) && (abs(LepGood2_pdgId) != 13 || LepGood2_sip3d < 4)' "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCat_muSip4}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCat_muSip4}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCat_muSip4}"

    echo "python ${RUN2X} $MVA $SF $SIP4 --pdir $ROOT/em   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2E} $MVA $SF $SIP4 --pdir $ROOT/ee   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2M} $MVA $SF $SIP4 --pdir $ROOT/mumu $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
;;
2lss_muSip4BCat4Plots)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    SIP4=" -A pt2010 sip4 '(abs(LepGood1_pdgId) != 13 || LepGood1_sip3d < 4) && (abs(LepGood2_pdgId) != 13 || LepGood2_sip3d < 4)' "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCat4Plots_muSip4}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCat4Plots_muSip4}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCat4Plots_muSip4}"

    echo "python ${RUN2X} $MVA $SF $SIP4 --pdir $ROOT/em   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2E} $MVA $SF $SIP4 --pdir $ROOT/ee   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2M} $MVA $SF $SIP4 --pdir $ROOT/mumu $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
;;
2lss_SUS134Plots)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataSUS134Plots}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataSUS134Plots}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataSUS134Plots}"
    RUN2X="${RUN2X/_tight.txt/_SUS13.txt}"
    RUN2E="${RUN2E/_tight.txt/_SUS13.txt}"
    RUN2M="${RUN2M/_tight.txt/_SUS13.txt}"

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
;;
2lss_SUS13C4Plots)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataSUS13C4Plots}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataSUS13C4Plots}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataSUS13C4Plots}"
    RUN2X="${RUN2X/_tight.txt/_SUS13C.txt}"
    RUN2E="${RUN2E/_tight.txt/_SUS13C.txt}"
    RUN2M="${RUN2M/_tight.txt/_SUS13C.txt}"

    #echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    #echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2 --sP nJet25,MVA_2LSS_4j_6var "
;;

2lss_IDCat)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataIDCat}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataIDCat}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataIDCat}"
    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
;;
2lss_SIPCat)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataSIPCat}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataSIPCat}"
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataSIPCat}"
    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2  --sP n_mu_sip35 --mcc mcCorrections.txt "
    #echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/mm   $BINCL --rebin 2  --sP n_mu_sip35 --mcc mcCorrections.txt "
;;

2lss_sip4)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    SIP4=" -A pt2010 sip4 'max(LepGood1_sip3d,LepGood2_sip3d) < 4' "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-data_sip4} --xp data $SIP4 "
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-data_sip4} --xp data $SIP4 "
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-data_sip4} --xp data $SIP4 "
    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
;;
2lss_mcScaled)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-mcScaled} "
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-mcScaled} "
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-mcScaled} "

    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/em   $BINCL --rebin 2   "
    echo "python ${RUN2E/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu $BINCL --rebin 2   "

    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
    echo "python ${RUN2X} $MVA $SF --pdir $ROOT/btight/em   $BTIGHT --rebin 4   "
    echo "python ${RUN2E} $MVA $SF --pdir $ROOT/btight/ee   $BTIGHT --rebin 4   "
    echo "python ${RUN2M} $MVA $SF --pdir $ROOT/btight/mumu $BTIGHT --rebin 4   "
;;
2lss_mcScaledBCat4Plots)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-mcScaledBCat4Plots-mumu} "
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-mcScaledBCat4Plots-em} "
    RUN2M3="${RUN2M/mca-2lss-data/mca-2lss-mcScaledBCat4Plots-mumu-3j}  -R 4j 3j 'nJet25==3' "
    #echo "python ${RUN2M/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu $BINCL --rebin 2  --mcc mcCorrections.txt   "
    echo "python ${RUN2X/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/em   $BINCL --rebin 2  --mcc mcCorrections.txt --sP 'muon_.*'   "
    #echo "python ${RUN2M3/mvaVars_2lss/cr_2lss_lowj_plots} $MVA $SF --pdir $ROOT/kinematics/mumu_3j $BINCL --rebin 2  --mcc mcCorrections.txt  --sP 'worst(SIP|Dxy|Dz)' "
;;
2lss_3j)
    SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    J3="-R 4j 3j nJet25==3"
    #MVA="$MVA -F sf/t   /afs/cern.ch/user/g/gpetrucc/w/TREES_250513_HADD/2_finalmva_2lss_2_v1/evVarFriend_{cname}.root "
    echo "python ${RUN2X} $J3 $MVA $SF --pdir $ROOT/em   $BINCL --rebin 2   "
    echo "python ${RUN2E} $J3 $MVA $SF --pdir $ROOT/ee   $BINCL --rebin 2   "
    echo "python ${RUN2M} $J3 $MVA $SF --pdir $ROOT/mumu $BINCL --rebin 2   "
;;

3l_tight)
    RUN3L="${CORE} mca-3l_tight-data.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5"
#    echo "python $RUN3L $SF $MVA --pdir $ROOT/bloose  $BLOOSE --rebin 5"
#    echo "python $RUN3L $SF $MVA --pdir $ROOT/btight  $BTIGHT --rebin 5"
    POS=" -A pt2010 positive 'LepGood1_charge+LepGood2_charge+LepGood3_charge>0' "
    NEG=" -A pt2010 negative 'LepGood1_charge+LepGood2_charge+LepGood3_charge<0' "
    echo "python $RUN3L $SF $MVA --pdir $ROOT/postfit/pos/  $BINCL  --rebin 4 --sP finalMVA $POS "
    echo "python $RUN3L $SF $MVA --pdir $ROOT/postfit/neg/  $BINCL  --rebin 4 --sP finalMVA $NEG "
;;
3l_tightBCat4Plots)
    RUN3L="${CORE} mca-3l_tight-dataBCat4Plots.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5"
;;
3l_mcScaled)
    RUN3L="${CORE} mca-3l_tight-mcScaled.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5"
;;
3l_mcScaledBCat4Plots)
 RUN3L="${CORE} mca-3l_tight-mcScaledBCat4Plots.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
 SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
 MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
 #echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5"
 echo "python ${RUN3L/mvaVars_3l/cr_3l_lowj_plots} $SF $MVA --pdir $ROOT/kinematics $BINCL  --rebin 2 --mcc mcCorrections.txt  --sP 'max_muon_.*'"
;;
3l_SIPCat)
    RUN3L="${CORE} mca-3l_tight-dataSIPCat.txt bins/3l_tight.txt bins/cr_2lss_lowj_plots.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL   --sP n_mu_sip35_3l --mcc mcCorrections.txt"
;;
3l_sip4)
    RUN3L="${CORE} mca-3l_tight-data_sip4.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SIP4=" -A pt2010 sip4 'max(LepGood1_sip3d,LepGood2_sip3d) < 4 && LepGood3_sip3d < 4' "
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5 $SIP4 --xp data --sP finalMVA"
;;
3l_muSip4BCat)
    RUN3L="${CORE} mca-3l_tight-dataBCat_muSip4.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SIP4=" -A pt2010 sip4 '(abs(LepGood1_pdgId) != 13 || LepGood1_sip3d < 4) && (abs(LepGood2_pdgId) != 13 || LepGood2_sip3d < 4) && (abs(LepGood3_pdgId) != 13 || LepGood3_sip3d < 4)' "
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5 $SIP4  --sP finalMVA"
;;
3l_muSip4BCat4Plots)
    RUN3L="${CORE} mca-3l_tight-dataBCat4Plots_muSip4.txt bins/3l_tight.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SIP4=" -A pt2010 sip4 '(abs(LepGood1_pdgId) != 13 || LepGood1_sip3d < 4) && (abs(LepGood2_pdgId) != 13 || LepGood2_sip3d < 4) && (abs(LepGood3_pdgId) != 13 || LepGood3_sip3d < 4)' "
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5 $SIP4  --sP finalMVA"
;;
3l_SUS134Plots)
    RUN3L="${CORE} mca-3l_tight-dataSUS134Plots.txt bins/3l_tight_SUS13.txt bins/mvaVars_3l.txt  --showRatio --maxRatioRange 0 3.7 --poisson"
    SF="$FSF -W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    MVA="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    echo "python $RUN3L $SF $MVA --pdir $ROOT/        $BINCL  --rebin 5  --sP finalMVA"
    echo "python ${RUN3L/mvaVars_3l.txt/cr_3l_lowj_plots.txt}  $SF $MVA --pdir $ROOT/kinematics $BINCL"
;;
4l)
    RUN4L="${CORE} mca-4l-ttscale.txt bins/4l.txt bins/mvaVars_4l.txt  --showRatio --maxRatioRange 0 3.7 --poisson "
    MVA="-F finalMVA/t $T/0_finalmva_4l/finalMVA_4L_{cname}.root"
    SF="$FSF -W 'puWeight*Eff_4lep*SF_btag*SF_LepMVALoose_4l'"
    echo "python $RUN4L $SF --pdir $ROOT/        $BINCL "
#    echo "python $RUN4L $SF --pdir $ROOT/bloose  $BLOOSE"
#    echo "python $RUN4L $SF --pdir $ROOT/btight  $BTIGHT"
;;
4l4Plots)
    RUN4L="${CORE} mca-4l-ttscale4Plots.txt bins/4l.txt bins/mvaVars_4l.txt  --showRatio --maxRatioRange 0 3.7 --poisson "
    MVA="-F finalMVA/t $T/0_finalmva_4l/finalMVA_4L_{cname}.root"
    SF="$FSF -W 'puWeight*Eff_4lep*SF_btag*SF_LepMVALoose_4l'"
    echo "python $RUN4L $SF --pdir $ROOT/        $BINCL --sP nJet25"
#    echo "python $RUN4L $SF --pdir $ROOT/bloose  $BLOOSE"
#    echo "python $RUN4L $SF --pdir $ROOT/btight  $BTIGHT"
;;
2lss_BCat4Plots_MVA)
    MVA=" -F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    for X in MVA{05,03,00,m05,m03,m07} MVA{03,m03,m07}X; do
        #
        # define scale factors
        case $X in
            MVA05)  SFMVA="SF_LepMVATight_2l";;
            MVA03)  SFMVA="sqrt(SF_LepMVATight_2l*SF_LepMVALoose_2l)";;
            MVA00)  SFMVA="SF_LepMVALoose_2l";;
            MVAm03) SFMVA="SF_LepMVALoose_2l";;
            MVAm05) SFMVA="SF_LepMVALoose_2l";;
            MVAm07) SFMVA="sqrt(SF_LepMVALoose_2l)";;
            MVA03X)  SFMVA="sqrt(SF_LepMVATightI_2l*SF_LepMVALooseX_2l)"; RUN2M="${RUN2M} --FM sf/t $T/0_moreSFs_v1/sfFriend_{cname}.root" ;;
            MVAm03X) SFMVA="SF_LepMVALooseX_2l";                          RUN2M="${RUN2M} --FM sf/t $T/0_moreSFs_v1/sfFriend_{cname}.root" ;;
            MVAm07X) SFMVA="sqrt(SF_LepMVALooseI_2l*SF_LepMVALooseX_2l)"; RUN2M="${RUN2M} --FM sf/t $T/0_moreSFs_v1/sfFriend_{cname}.root" ;;
        esac;
        SF="$FSF -W 'puWeight*Eff_2lep*SF_btag*$SFMVA*SF_LepTightCharge_2l*SF_trig2l'"
        #
        # Nominal plot
        RUN2MI=$(echo "$RUN2M" | \
                    sed "s+mca-2lss-data+syst/mca-2lss-dataBCat4Plots_$X+" | \
                    sed "s+bins/2lss_mumu_tight+syst/2lss_mumu_$X+" );
        echo "python ${RUN2MI} $MVA $SF --pdir $ROOT/mumu/$X $BINCL --rebin 2 --sP MVA_2LSS_4j_6var "
        echo "python ${RUN2MI} $MVA $SF --pdir $ROOT/mumu/$X/fit $BINCL --rebin 2 --sP MVA_2LSS_4j_6var --fitData"
        #
        # Fail MVA plot
        case $X in
            *X)     MVA1F="" ;;
            MVA0*)  MVA1F=" -R MVA 1-1 '(LepGood1_mva>0.${X/MVA0/})+(LepGood2_mva>0.${X/MVA0/})==1' " ;;
            MVAm0*) MVA1F=" -R MVA 1-1 '(LepGood1_mva>-0.${X/MVAm0/})+(LepGood2_mva>-0.${X/MVAm0/})==1' " ;;
        esac;
        RUN2MF="${RUN2M/mca-2lss-data/mca} --sp TT,TW --scaleSigToData $MVA1F "
        [[ "$MVA1F" != "" ]] && echo "python ${RUN2MF} $MVA $SF --pdir $ROOT/mumu/$X/oneFail $BINCL --rebin 2 --sP MVA_2LSS_4j_6var "
    done; 
;;
esac;

