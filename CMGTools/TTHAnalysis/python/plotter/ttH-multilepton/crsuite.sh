#!/bin/bash

WHAT=$1; if [[ "$1" == "" ]]; then echo "frsuite.sh <what>"; exit 1; fi

if [[ "$HOSTNAME" == "cmsphys10" ]]; then
    T="/data/g/gpetrucc/TREES_270314_HADD";
    J=8;
else
    T="/afs/cern.ch/user/g/gpetrucc/w/TREES_270314_HADD";
    #T="/afs/cern.ch/user/g/gpetrucc/ttH/TREES_270314_LITE";
    J=5;
fi

CORE="mcPlots.py -P $T -j $J -l 19.6 --showRatio --maxRatioRange 0 3.7 -f --poisson --s2v --tree ttHLepTreeProducerTTH"
#CORE="${CORE} --FM sf/t $T/0_SFs_v2/sfFriend_{cname}.root "

MVA_2L="-F sf/t   $T/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
MVA_3L="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
MVA_3LC="-F sf/t $T/0_finalmva_3lcat/evVarFriend_{cname}.root "

ROOT="plots/270314/$WHAT"


RUN2L="${CORE} mca-2lss-data.txt $MVA_2L"
RUN2E="${RUN2L} bins/2lss_ee.txt   bins/cr_2lss_lowj_plots.txt "
RUN2M="${RUN2L} bins/2lss_mumu.txt bins/cr_2lss_lowj_plots.txt --xp QF_data "
RUN2X="${RUN2L} bins/2lss_em.txt   bins/cr_2lss_lowj_plots.txt  "
J3="  -R 4j 3j 'nJet25 == 3' "
J4E="  -R 4j 4je 'nJet25 == 4' "
BINCL=" -X 2B"
BLOOSE=" -I 2B"
BMEDIUM=" -R 2B 2b1B '(nBJetLoose25>=2 && nBJetMedium25>=1)'"
B2LOOSE=" -R 2B 2bl '(nBJetLoose25>=2)'"
BTIGHT=" "
### =============== 2L SS, 2b loose =================
case $WHAT in
2lss)
    SF="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    SF0="-W 'puWeight*Eff_2lep*SF_btag*SF_LepTightCharge_2l*SF_trig2l'"
    MVA1F=" -R MVA 1-1 '(LepGood1_mva>0.7)+(LepGood2_mva>0.7)==1' "
    RUN2MF="${RUN2M/mca-2lss-data/mca4failMVAplots}  --fitData $MVA1F "
    RUN2EF="${RUN2E/mca-2lss-data/mca4failMVAplots}  --fitData $MVA1F "
    RUN2XF="${RUN2X/mca-2lss-data/mca4failMVAplots}  --fitData $MVA1F "
    #############################################################################
    ## ---- 4j fail MVA
    echo "python ${RUN2XF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/failLepMVA/em_TTscaled        $BINCL            --lspam 'CMS ttH, e^{#pm}#mu^{#pm} channel' "
    #echo "python ${RUN2XF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/failLepMVA/btight/em_TTscaled $BTIGHT --rebin 4  "
    echo "python ${RUN2EF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/failLepMVA/ee_TTscaled        $BINCL  --rebin 2 --lspam 'CMS ttH, e^{#pm}e^{#pm} channel'  "
    #echo "python ${RUN2EF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/failLepMVA/btight/ee_TTscaled $BTIGHT --rebin 6  "
    echo "python ${RUN2MF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/failLepMVA/mumu_TTscaled        $BINCL           --lspam 'CMS ttH, #mu^{#pm}#mu^{#pm} channel'   "
    #echo "python ${RUN2MF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/failLepMVA/btight/mumu_TTscaled $BTIGHT --rebin 4  "
;;
2lss_3j)
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    SF="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    #############################################################################
    echo "python ${RUN2X/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j/em   $J3 $BINCL --rebin 2  "
    echo "python ${RUN2E/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j/ee   $J3 $BINCL --rebin 2  "
    echo "python ${RUN2M/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j/mumu $J3 $BINCL --rebin 2  "
;;
2lss_4je)
    RUN2M="${RUN2M/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    RUN2E="${RUN2E/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    RUN2X="${RUN2X/mca-2lss-data/mca-2lss-dataBCat4Plots}"
    SF="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    #############################################################################
    echo "python ${RUN2X/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/em   $J4E $BLOOSE --rebin 2  "
    echo "python ${RUN2E/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/ee   $J4E $BLOOSE --rebin 2  "
    echo "python ${RUN2M/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/mumu $J4E $BLOOSE --rebin 3  "
;;
2lss_more)
    ## ---- 3j fail MVA
    echo "python ${RUN2XF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100_failLepMVA/em_TTscaled $J3T $BINCL  "
    echo "python ${RUN2EF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100_failLepMVA/ee_TTscaled $J3T $BINCL  --rebin 2 "
    echo "python ${RUN2MF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100_failLepMVA/mumu_TTscaled $J3T $BINCL  "
    ## ---- 3j fail MVA (b tight)
    echo "python ${RUN2XF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100_failLepMVA/btight/em_TTscaled   $J3T $BTIGHT  "
    echo "python ${RUN2EF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100_failLepMVA/btight/ee_TTscaled   $J3T $BTIGHT  --rebin 2 "
    echo "python ${RUN2MF/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100_failLepMVA/btight/mumu_TTscaled $J3T $BTIGHT  "
    ## ---- 3j_pt2020_htllv100 pass MVA
    echo "python ${RUN2X/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100/em   $J3T $BINCL --rebin 4  "
    echo "python ${RUN2E/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100/ee   $J3T $BINCL --rebin 6  "
    echo "python ${RUN2M/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100/mumu $J3T $BINCL --rebin 4  "
    ## ---- 3j_pt2020_htllv100 pass MVA
    echo "python ${RUN2X/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100/btight/em   $J3T $BTIGHT --rebin 4  "
    echo "python ${RUN2E/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100/btight/ee   $J3T $BTIGHT --rebin 6  "
    echo "python ${RUN2M/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/3j_pt2020_htllv100/btight/mumu $J3T $BTIGHT --rebin 4  "
    ## ---- 4j exclusive
    echo "python ${RUN2X/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/4j_exclusive/bloose_exclusive/em   $J4E $BLOOSE --rebin 4   "
    echo "python ${RUN2E/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/4j_exclusive/bloose_exclusive/ee   $J4E $BLOOSE --rebin 6   "
    echo "python ${RUN2M/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/4j_exclusive/bloose_exclusive/mumu $J4E $BLOOSE --rebin 4   "
    ## ---- 4j exclusive
    echo "python ${RUN2X/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/4j_exclusive_pt2020_htllv100/bloose_exclusive/em   $J4ET $BLOOSE --rebin 2  --sP MVA_2LSS_4j_6var "
    echo "python ${RUN2E/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/4j_exclusive_pt2020_htllv100/bloose_exclusive/ee   $J4ET $BLOOSE --rebin 4  --sP MVA_2LSS_4j_6var "
    echo "python ${RUN2M/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/4j_exclusive_pt2020_htllv100/bloose_exclusive/mumu $J4ET $BLOOSE --rebin 2  --sP MVA_2LSS_4j_6var "
    #############################################################################
    ## ---- 4j fail MVA
    echo "python ${RUN2XF} $SF0 --pdir $ROOT/kinematics/failLepMVA/em_TTscaled        $BINCL             "
    echo "python ${RUN2XF} $SF0 --pdir $ROOT/kinematics/failLepMVA/btight/em_TTscaled $BTIGHT --rebin 4  "
    echo "python ${RUN2EF} $SF0 --pdir $ROOT/kinematics/failLepMVA/ee_TTscaled        $BINCL  --rebin 2  "
    echo "python ${RUN2EF} $SF0 --pdir $ROOT/kinematics/failLepMVA/btight/ee_TTscaled $BTIGHT --rebin 6  "
    echo "python ${RUN2MF} $SF0 --pdir $ROOT/kinematics/failLepMVA/mumu_TTscaled        $BINCL             "
    echo "python ${RUN2MF} $SF0 --pdir $ROOT/kinematics/failLepMVA/btight/mumu_TTscaled $BTIGHT --rebin 4  "

    echo "python ${RUN2XF} $SF0 --pdir $ROOT/kinematics/failLepMVA_pt2020_htllv100/em_TTscaled $LT        $BINCL             "
    echo "python ${RUN2XF} $SF0 --pdir $ROOT/kinematics/failLepMVA_pt2020_htllv100/btight/em_TTscaled $LT $BTIGHT --rebin 4  "
    echo "python ${RUN2EF} $SF0 --pdir $ROOT/kinematics/failLepMVA_pt2020_htllv100/ee_TTscaled $LT        $BINCL  --rebin 2  "
    echo "python ${RUN2EF} $SF0 --pdir $ROOT/kinematics/failLepMVA_pt2020_htllv100/btight/ee_TTscaled $LT $BTIGHT --rebin 6  "
    echo "python ${RUN2MF} $SF0 --pdir $ROOT/kinematics/failLepMVA_pt2020_htllv100/mumu_TTscaled $LT        $BINCL             "
    echo "python ${RUN2MF} $SF0 --pdir $ROOT/kinematics/failLepMVA_pt2020_htllv100/btight/mumu_TTscaled $LT $BTIGHT --rebin 4  "
 
    ## ---- 3j fail MVA
    echo "python ${RUN2XF} $SF0 --pdir $ROOT/kinematics/3j_failLepMVA/em_TTscaled $J3 $BINCL  "
    echo "python ${RUN2EF} $SF0 --pdir $ROOT/kinematics/3j_failLepMVA/ee_TTscaled $J3 $BINCL  --rebin 2 "
    echo "python ${RUN2MF} $SF0 --pdir $ROOT/kinematics/3j_failLepMVA/mumu_TTscaled $J3 $BINCL  "
    ## ---- 3j pass MVA
    echo "python ${RUN2X} $SF0 --pdir $ROOT/kinematics/3j/em   $J3 $BINCL --rebin 4  "
    echo "python ${RUN2E} $SF0 --pdir $ROOT/kinematics/3j/ee   $J3 $BINCL --rebin 6  "
    echo "python ${RUN2M} $SF0 --pdir $ROOT/kinematics/3j/mumu $J3 $BINCL --rebin 4  "
    ## ---- 3j_pt2020_htllv100 fail MVA
    echo "python ${RUN2XF} $SF0 --pdir $ROOT/kinematics/3j_pt2020_htllv100_failLepMVA/em_TTscaled $J3T $BINCL  "
    echo "python ${RUN2EF} $SF0 --pdir $ROOT/kinematics/3j_pt2020_htllv100_failLepMVA/ee_TTscaled $J3T $BINCL  --rebin 2 "
    echo "python ${RUN2MF} $SF0 --pdir $ROOT/kinematics/3j_pt2020_htllv100_failLepMVA/mumu_TTscaled $J3T $BINCL  "
    ## ---- 3j_pt2020_htllv100 pass MVA
    echo "python ${RUN2X} $SF0 --pdir $ROOT/kinematics/3j_pt2020_htllv100/em   $J3T $BINCL --rebin 4  "
    echo "python ${RUN2E} $SF0 --pdir $ROOT/kinematics/3j_pt2020_htllv100/ee   $J3T $BINCL --rebin 6  "
    echo "python ${RUN2M} $SF0 --pdir $ROOT/kinematics/3j_pt2020_htllv100/mumu $J3T $BINCL --rebin 4  "
   ## ---- 4j exclusive
    echo "python ${RUN2X} $SF0 --pdir $ROOT/kinematics/4j_exclusive_pt2020_htllv100/bloose_exclusive/em   $J4ET $BLOOSE --rebin 2   "
    echo "python ${RUN2E} $SF0 --pdir $ROOT/kinematics/4j_exclusive_pt2020_htllv100/bloose_exclusive/ee   $J4ET $BLOOSE --rebin 4   "
    echo "python ${RUN2M} $SF0 --pdir $ROOT/kinematics/4j_exclusive_pt2020_htllv100/bloose_exclusive/mumu $J4ET $BLOOSE --rebin 2   "
    #############################################################################
;;
2lss_closure|2lss_closure_v2)
    SF0="-W 'puWeight*Eff_2lep*SF_btag*SF_LepTightCharge_2l*SF_trig2l'"
    ## ---- 4j fail MVA
    RUN2MMC="${RUN2M/mca-2lss-data/mca} -p TT,TW,WJets,DY "
    RUN2EMC="${RUN2E/mca-2lss-data/mca} -p TT,TW,WJets,DY "
    RUN2XMC="${RUN2X/mca-2lss-data/mca} -p TT,TW,WJets,DY "
    RUN2MDD="${RUN2M/mca-2lss-data/mca-2lss-dd} -p 'FR_.*,QF_.*' "
    RUN2EDD="${RUN2E/mca-2lss-data/mca-2lss-dd} -p 'FR_.*,QF_.*' "
    RUN2XDD="${RUN2X/mca-2lss-data/mca-2lss-dd} -p 'FR_.*,QF_.*' "
    TIGHT="-A pt2010 pt2020 'LepGood2_pt > 20 && LepGood1_pt+LepGood2_pt+met > 100' "
    echo "python ${RUN2XMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/em_MCBG        $BINCL  --rebin 4  "
    echo "python ${RUN2EMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/ee_MCBG        $BINCL  --rebin 6  "
    echo "python ${RUN2MMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/mumu_MCBG      $BINCL  --rebin 4  "
    echo "python ${RUN2XDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/em_MCDD        $BINCL  --rebin 4  "
    echo "python ${RUN2EDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/ee_MCDD        $BINCL  --rebin 6  "
    echo "python ${RUN2MDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/mumu_MCDD      $BINCL  --rebin 4  "

    echo "python ${RUN2XMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/em_tight_MCBG $BINCL $TIGHT --rebin 6  "
    echo "python ${RUN2EMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/ee_tight_MCBG $BINCL $TIGHT --rebin 8  "
    echo "python ${RUN2MMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/mumu_tight_MCBG $BINCL $TIGHT --rebin 6  "
    echo "python ${RUN2XDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/em_tight_MCDD $BINCL $TIGHT --rebin 6  "
    echo "python ${RUN2EDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/ee_tight_MCDD $BINCL $TIGHT --rebin 8  "
    echo "python ${RUN2MDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/mumu_tight_MCDD $BINCL $TIGHT --rebin 6  "


    echo "python ${RUN2XMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/btight/em_MCBG $BTIGHT --rebin 6  "
    echo "python ${RUN2EMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/btight/ee_MCBG $BTIGHT --rebin 8  "
    echo "python ${RUN2MMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/btight/mumu_MCBG $BTIGHT --rebin 6  "
    echo "python ${RUN2XDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/btight/em_MCDD $BTIGHT --rebin 6  "
    echo "python ${RUN2EDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/btight/ee_MCDD $BTIGHT --rebin 8  "
    echo "python ${RUN2MDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 --pdir $ROOT/mvaVars/btight/mumu_MCDD $BTIGHT --rebin 6  "

    ## ---- 3j 
    echo "python ${RUN2XMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 $J3 --pdir $ROOT/mvaVars/3j/em_MCBG $BINCL --rebin 4  "
    echo "python ${RUN2EMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 $J3 --pdir $ROOT/mvaVars/3j/ee_MCBG $BINCL --rebin 6  "
    echo "python ${RUN2MMC/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 $J3 --pdir $ROOT/mvaVars/3j/mumu_MCBG $BINCL --rebin 4  "
    echo "python ${RUN2XDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 $J3 --pdir $ROOT/mvaVars/3j/em_MCDD $BINCL --rebin 4  "
    echo "python ${RUN2EDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 $J3 --pdir $ROOT/mvaVars/3j/ee_MCDD $BINCL --rebin 6  "
    echo "python ${RUN2MDD/cr_2lss_lowj_plots/mvaVars_2lss} $SF0 $J3 --pdir $ROOT/mvaVars/3j/mumu_MCDD $BINCL --rebin 4  "
;;
3l_tight)
    RUN3L="${CORE} mca-3l_tight-data.txt bins/3l_tight.txt bins/PLOTS.txt  $MVA_3L "
    J2=" -A pt2010 lowj 'nJet25 == 2' "
    SF="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    TOPHAD=" -A pt2010 topHad 'bestMTopHad > 0' "
    NOTOPHAD=" -A pt2010 topHad 'bestMTopHad == 0' "
    ## ---- Fail lepton MVA ----
    SF0="-W 'puWeight*Eff_3lep*SF_btag' "
    MVA2F=" -R MVA 2-1 '(LepGood1_mva>0.7)+(LepGood2_mva>0.7)+(LepGood3_mva>0.7)==2' "
    RUN3LF="${RUN3L/mca-3l_tight-data/mca4failMVAplots} --fitData $MVA2F "
    echo "python ${RUN3LF/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/failLepMVA $SF0 $BINCL --rebin 2  --lspam 'CMS ttH, 3l channel' "
    #echo "python ${RUN3LF/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/failLepMVA/btight/TTscaled $SF0 $BTIGHT --rebin 5"
    #echo "python ${RUN3LF/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/failLepMVA/withTopHad/TTscaled $SF0 $TOPHAD $BINCL --rebin 2" 
    ## ---- Invert Z veto ----
    RUN3L="${RUN3L/mca-3l_tight-data/mca-3l_tight-dataBCat4Plots}"
    echo "python ${RUN3L/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/Zpeak $SF $BINCL -I 'Z veto' --rebin 5"
    #echo "python ${RUN3L/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/Zpeak/btight $SF -I 'Z veto' --rebin 8"
;;
diboson)
    SFT="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    RUNWZ="${CORE} mca-wzfit.txt bins/3l_tight.txt standard-candles/wz_3l_plots.txt "
    echo "python $RUNWZ $SFT --pdir $ROOT/wz -I 'Z veto' -R 2b 0b 'nBJetLoose25==0' -R 2B 2j 'nJet25>=2'  --sP mtW --rebin 3  --fitData"
    #echo "python $RUN $SFT --pdir $ROOT/mtw40 $MTW40 $TIGHT"
;;
3l_more)
    ## ---- 2 jet exclusive ----
    echo "python ${RUN3L/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/2j_exclusive $SF $BINCL $J2  --rebin 5 "
    ## --- categorized MVA
    echo "python ${RUN3LF/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/failLepMVA/withTopHad/TTscaled $SF0 $TOPHAD $BINCL $MVA_3LC --rebin 2 --sP finalMVA_Cat" 
    echo "python ${RUN3LF/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/failLepMVA/noTopHad/TTscaled $SF0 $NOTOPHAD $BINCL $MVA_3LC --rebin 2 --sP finalMVA_Cat" 
    echo "python ${RUN3L/PLOTS/mvaVars_3l}  --pdir $ROOT/mvaVars/Zpeak/withTopHad $SF   $TOPHAD $BINCL $MVA_3LC -I 'Z veto' --rebin 6 --sP finalMVA,finalMVA_Cat" 
    echo "python ${RUN3L/PLOTS/mvaVars_3l}  --pdir $ROOT/mvaVars/Zpeak/noTopHad   $SF $NOTOPHAD $BINCL $MVA_3LC -I 'Z veto' --rebin 6 --sP finalMVA,finalMVA_Cat" 
    echo "python ${RUN3L/PLOTS/mvaVars_3l}  --pdir $ROOT/mvaVars/2j_exclusive/withTopHad $SF   $TOPHAD $BINCL $MVA_3LC $J2 --rebin 6 --sP finalMVA,finalMVA_Cat" 
    echo "python ${RUN3L/PLOTS/mvaVars_3l}  --pdir $ROOT/mvaVars/2j_exclusive/noTopHad   $SF $NOTOPHAD $BINCL $MVA_3LC $J2 --rebin 6 --sP finalMVA,finalMVA_Cat" 
;;
3l_closure|3l_closure_v2|3l_closure_v3)
    SF="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    RUN3LMC="${CORE} mca.txt             bins/3l_tight.txt bins/PLOTS.txt  $MVA_3L  -p TT,TW,WJets,DY  "
    RUN3LDD="${CORE} mca-3l_tight-dd.txt bins/3l_tight.txt bins/PLOTS.txt  $MVA_3L  -p 'FR_.*,QF_.*' "
    echo "python ${RUN3LMC/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/MCBG $BINCL --rebin 5"
    echo "python ${RUN3LDD/PLOTS/mvaVars_3l} --pdir $ROOT/mvaVars/MCDD $BINCL --rebin 5"
;;
tt_2mu)
    CORE="${CORE/--maxRatioRange 0 2.4/--maxRatioRange 0.65 1.35}"
    CORE="${CORE/--doStatTest=chi2l/} "
    OS="-I same-sign -A pt2010 Zveto 'abs(mZ1-91.2) > 10 && met*0.00397 + mhtJet25*0.00265 > 0.2'"
    RUN2L="${CORE} mca-2l-data-topPtW.txt bins/2lss_mumu.txt bins/cr_2lss_lowj_plots.txt --xp QF_data ${MVA_2L} $OS "
    SF="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    #echo "python $RUN2L $SF --pdir $ROOT/ "
    echo "python ${RUN2L/mca-2l-data.txt/mca-2l-data-topPtW.txt} $SF --pdir $ROOT/ $BINCL --scaleSigToData --mcc mcCorrections.txt "
;;
tt_2l)
    CORE="${CORE/--maxRatioRange 0 2.4/--maxRatioRange 0.65 1.35}"
    CORE="${CORE/--doStatTest=chi2l/} "
    RUN2L="${CORE} mca-2l-data.txt bins/cr_tt_2l_em.txt bins/cr_2lss_lowj_plots.txt --xp QF_data ${MVA_2L} "
    SF="-W 'puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l'"
    #echo "python $RUN2L $SF --pdir $ROOT/ "
    echo "python ${RUN2L/mca-2l-data.txt/mca-2l-data-topPtW.txt} $SF --pdir $ROOT/topPtW/ "
    echo "python ${RUN2L/mca-2l-data.txt/mca-2l-data-topPtW.txt} $SF --pdir $ROOT/topPtW/fit --fitData "
;;
z_2l)
    CORE="${CORE/--doStatTest=chi2l/}   -W 'puWeight*Eff_2lep' --sp 'DY.*' --scaleSigToData --mcc mcCorrections.txt "
    ZEE="${CORE} mca-incl.txt standard-candles/zmm-4mva.txt standard-candles/zmm-4mva_plots.txt --xf 'DoubleMu.*,MuEG.*' "
    ZMM="${CORE} mca-incl.txt standard-candles/zmm-4mva.txt standard-candles/zmm-4mva_plots.txt --xf 'DoubleEl.*,MuEG.*' "
    ZEENR="${ZEE/--showRatio/} "
    ZMMNR="${ZMM/--showRatio/} "
    echo "python ${ZMMNR} --pdir $ROOT/zmm/highPt/noratio  -R muon thisbin 'abs(LepGood2_pdgId) == 13 && LepGood2_pt > 15'"
    echo "python ${ZEENR} --pdir $ROOT/zee/highPt/noratio  -R muon thisbin 'abs(LepGood2_pdgId) == 11 && LepGood2_pt > 15'"
;;
wz_3l)
    SFT="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    TIGHT=" -R 'lep MVA 3' mu 'min(LepGood1_mva,min(LepGood2_mva,LepGood3_mva)) > 0.7 && (abs(LepGood1_pdgId) == 13 || (LepGood1_convVeto > 0 && LepGood1_innerHits == 0)) && (abs(LepGood2_pdgId) == 13 || (LepGood2_convVeto > 0 && LepGood2_innerHits == 0)) && (abs(LepGood3_pdgId) == 13 || (LepGood3_convVeto > 0 && LepGood3_innerHits == 0))' "
    MTW40=" -A 'Z peak' mtw 'mtw_wz3l(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass,LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass,LepGood3_pt,LepGood3_eta,LepGood3_phi,LepGood3_mass,mZ1,met,met_phi) > 40' "
    RUN="${CORE} mca-incl.txt standard-candles/wz_3l.txt standard-candles/wz_3l_plots.txt "
    echo "python $RUN $SFT --pdir $ROOT/ $TIGHT "
    #echo "python $RUN $SFT --pdir $ROOT/mtw40 $MTW40 $TIGHT"
;;
ttZ_3l_tight)
    RUN3L="${CORE} mca-3l_tight-dataBCat4Plots.txt bins/cr_ttz_tight.txt bins/cr_ttz_plots.txt "
    SF="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l'"
    echo "python $RUN3L $SF --pdir $ROOT/ " 
    echo "python $RUN3L $SF --pdir $ROOT/4j -A 1B 4j 'nJet25 >= 4' --rebin 2"
    echo "python $RUN3L $SF --pdir $ROOT/bloose -A pt2010 4j 'nJet25 >= 2' --rebin 2 -X 1B -R 2b 2bl '(nBJetLoose25 >= 2 || nBJetMedium25 >= 1)'"
    echo "python $RUN3L $SF --pdir $ROOT/bloose/4j -A pt2010 4j 'nJet25 >= 4' --rebin 2 -X 1B -R 2b 2bl '(nBJetLoose25 >= 2 || nBJetMedium25 >= 1)'"
;;
zz_4l)
    CORE="${CORE/--doStatTest=chi2l/} "
    CORE="${CORE/--maxRatioRange 0 2.4/--maxRatioRange 0.4 1.7}"
    #RUN4L="${CORE} mca-4l-data.txt bins/cr_zz4l.txt bins/cr_zz4l_plots.txt "
    RUN4L="${CORE} mca-4l-ttscale4Plots.txt bins/cr_zz4l.txt bins/cr_zz4l_plots.txt "
    S0="-W 'puWeight*Eff_4lep*SF_btag'"
    SF="-W 'puWeight*Eff_4lep*SF_btag*SF_LepMVALoose_4l'"
    FMU="-A pt2010 4mu 'abs(LepGood1_pdgId) == 13 && abs(LepGood1_pdgId) == 13'"
    FEL="-A pt2010 4el 'abs(LepGood1_pdgId) == 11 && abs(LepGood1_pdgId) == 11'"
    echo "python $RUN4L $SF --pdir $ROOT/ " 
    echo "python $RUN4L $SF --pdir $ROOT/z4l/ -A pt2010 Z4l 'm4l > 80 && m4l < 106' " 
    echo "python $RUN4L $S0 --pdir $ROOT/noLepSF/ " 
    echo "python $RUN4L $S0 --pdir $ROOT/z4l/noLepSF/ -A pt2010 Z4l 'm4l > 80 && m4l < 106' " 
    echo "python $RUN4L $SF --pdir $ROOT/4mu/ $FMU" 
    echo "python $RUN4L $SF --pdir $ROOT/4e/  $FEL" 
;;
z_3l)
    CORE="${CORE/--doStatTest=chi2l/} "
    RUN="${CORE} mca-incl-dysplit.txt bins/cr_z_3l.txt bins/cr_z_3l_plots.txt  --scaleSigToData  --sp 'DY.*' "
    RNR="${RUN/--showRatio/} "
    #SF="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_2l' --mcc mcCorrections.txt";
    SF="-W 'puWeight' --mcc mcCorrections.txt";
    EL=" -A Z12 ele 'abs(LepGood3_pdgId) == 11' --sP 'l3.*' "
    MU=" -A Z12 ele 'abs(LepGood3_pdgId) == 13' --sP 'l3.*' "
    echo "python $RUN $SF --pdir $ROOT/ " 
    echo "python $RNR $SF --pdir $ROOT/noratio/ " 
    #echo "python $RUN $SF --pdir $ROOT/el/ $EL " 
    echo "python $RNR $SF --pdir $ROOT/el/noratio/ $EL " 
    #echo "python $RUN $SF --pdir $ROOT/mu/ $MU " 
    echo "python $RNR $SF --pdir $ROOT/mu/noratio/ $MU " 
;;
w_l_fakel)
    CORE="${CORE/--doStatTest=chi2l/} "
    RUN="${CORE} mca-incl-wjsplit.txt bins/cr_wjets_fakel.txt bins/cr_wjets_fakel_plots.txt   --sp 'W[jb].*' --scaleSigToData"
    RNR="${RUN/--showRatio/} "
    #SF="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_2l' --mcc mcCorrections.txt";
    SF="-W 'puWeight' --mcc mcCorrections.txt";
    EL=" -A pt25 ele 'abs(LepGood2_pdgId) == 11'  "
    MU=" -A pt25 mu  'abs(LepGood2_pdgId) == 13'  "
    echo "python $RUN $SF --pdir $ROOT/ " 
    echo "python $RNR $SF --pdir $ROOT/noratio/ " 
    #echo "python $RUN $SF --pdir $ROOT/el/ $EL " 
    echo "python $RNR $SF --pdir $ROOT/el/noratio/ $EL " 
    #echo "python $RUN $SF --pdir $ROOT/mu/ $MU " 
    echo "python $RNR $SF --pdir $ROOT/mu/noratio/ $MU " 
;;

tt_3l)
    CORE="${CORE/--doStatTest=chi2l/} "
    RUN="${CORE} mca-ttsplit.txt bins/cr_tt_3l.txt bins/cr_tt_3l_plots.txt  --scaleSigToData  --sp 'TT[lb]' --poisson "
    RNR="${RUN/--showRatio/} "
    SF="-W 'puWeight*Eff_3lep*SF_btag*SF_LepMVATight_2l' --mcc mcCorrections.txt";
    EL=" -A pt2010 ele 'abs(LepGood3_pdgId) == 11' --sP 'l3.*' "
    MU=" -A pt2010 mu  'abs(LepGood3_pdgId) == 13' --sP 'l3.*' "
    #echo "python $RUN $SF --pdir $ROOT/ -X 2B" 
    echo "python $RNR $SF --pdir $ROOT/noratio/ -X 2B" 
    echo "python $RUN $SF --pdir $ROOT/el/ $EL -X 2B --rebin 2" 
    #echo "python $RNR $SF --pdir $ROOT/el/noratio/ $EL -X 2B" 
    echo "python $RUN $SF --pdir $ROOT/mu/ $MU -X 2B --rebin 2" 
    #echo "python $RNR $SF --pdir $ROOT/mu/noratio/ $MU -X 2B" 
    echo "python $RUN $SF --pdir $ROOT/btight/ " 
    #echo "python $RNR $SF --pdir $ROOT/btight/noratio/ " 
    echo "python $RUN $SF --pdir $ROOT/btight/el/ $EL --rebin 4" 
    #echo "python $RNR $SF --pdir $ROOT/btight/el/noratio/ $EL " 
    echo "python $RUN $SF --pdir $ROOT/btight/mu/ $MU --rebin 4" 
    #echo "python $RNR $SF --pdir $ROOT/btight/mu/noratio/ $MU " 
;;
tt_l_lfake)
    CORE="${CORE/--doStatTest=chi2l/} "
    RUN="${CORE} mca-ttsplit-2l.txt bins/cr_tt_l_lfake.txt bins/cr_tt_l_lfake_plots.txt  --scaleSigToData  --sp 'TT[lb]' --poisson "
    RNR="${RUN/--showRatio/} "
    SF="-W 'puWeight*Eff_2lep*SF_btag*sqrt(SF_LepMVATight_2l)' --mcc mcCorrections.txt";
    EL=" -X 'fail mu' "
    MU=" -X 'fail el' "
    echo "python $RUN $SF --pdir $ROOT/ -X 2B -X 'fail mu' -X 'fail el'" 
    #echo "python $RNR $SF --pdir $ROOT/noratio/ -X 2B" 
    echo "python $RUN $SF --pdir $ROOT/el/ $EL -X 2B " 
    #echo "python $RNR $SF --pdir $ROOT/el/noratio/ $EL -X 2B" 
    echo "python $RUN $SF --pdir $ROOT/mu/ $MU -X 2B " 
    #echo "python $RNR $SF --pdir $ROOT/mu/noratio/ $MU -X 2B" 
    echo "python $RUN $SF --pdir $ROOT/btight/  -X 'fail mu' -X 'fail el' --rebin 2" 
    #echo "python $RNR $SF --pdir $ROOT/btight/noratio/ " 
    echo "python $RUN $SF --pdir $ROOT/btight/el/ $EL --rebin 2" 
    #echo "python $RNR $SF --pdir $ROOT/btight/el/noratio/ $EL " 
    echo "python $RUN $SF --pdir $ROOT/btight/mu/ $MU --rebin 2" 
    #echo "python $RNR $SF --pdir $ROOT/btight/mu/noratio/ $MU " 
;;

esac;

