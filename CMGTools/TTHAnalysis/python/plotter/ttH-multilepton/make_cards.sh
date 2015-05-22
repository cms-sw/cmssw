#!/bin/bash

if [[ "$HOSTNAME" == "cmsphys05" ]]; then
    T="/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD";
    J=6;
elif [[ "$HOSTNAME" == "olsnba03" ]]; then
    T="/data/gpetrucc/TREES_250513_HADD";
    J=16;
elif [[ "$HOSTNAME" == "lxbse14c09.cern.ch" ]]; then
    T="/var/ssdtest/gpetrucc/TREES_250513_HADD";
    J=5;
else
    T="/afs/cern.ch/work/g/gpetrucc/TREES_250513_HADD";
    J=4;
fi

SCENARIO=""
if echo "X$1" | grep -q "scenario"; then SCENARIO="$1"; shift; fi
OPTIONS=" -P $T -j $J -l 19.7 -f  --od cards/new "
if [[ "$SCENARIO" != "" ]]; then
    test -d cards/$SCENARIO || mkdir -p cards/$SCENARIO
    OPTIONS=" -P $T -j $J -l 19.7 -f  --od cards/$SCENARIO --project $SCENARIO --asimov ";
else
    OPTIONS=" -P $T -j $J -l 19.5 -f  --od cards/paper-195-sfv3 --tree ttHLepTreeProducerBase ";
    #OPTIONS=" -P $T -j $J -l 19.6 -f  --od cards/new196";
    OPTIONS="${OPTIONS} --masses masses.txt --mass-int-algo=noeff"
fi
#OPTIONS=" -P $T -j $J -l 19.6 -f  --od cards/mva/ "
#OPTIONS="${OPTIONS} --masses masses.txt --mass-int-algo=noeff"
SYSTS="systsEnv.txt ../../macros/systematics/btagSysts.txt"
BLoose=" -I 2B "
BAny=" -X 2B "
BTight="  "

if [[ "$1" == "" ]] || echo $1 | grep -q 2lss; then
    OPTIONS="${OPTIONS} --FM sf/t $T/0_SFs_v3/sfFriend_{cname}.root --xp FR_data_.* "
    OPT_2L="${OPTIONS} -W puWeight*Eff_2lep*SF_btag*SF_LepMVATight_2l*SF_LepTightCharge_2l*SF_trig2l_new"
    MVA_2L="-F sf/t   /afs/cern.ch/user/g/gpetrucc/w/TREES_250513_HADD/2_finalmva_2lss_v2/evVarFriend_{cname}.root "
    POS=" -A pt2010 positive LepGood1_charge>0 "
    NEG=" -A pt2010 positive LepGood1_charge<0 "
    for X in 2lss_{mumu,ee,em}; do 
        #if [[ "$X" == "2lss_mumu" ]]; then continue; fi
        echo $X; #~gpetrucc/sh/bann $X
        # ---- MVA separated by charge (for nominal result) ----
        python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA_pos $MVA_2L $POS $BAny;
        python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA_neg $MVA_2L $NEG $BAny;

        # ---- n(jet) separated by charge (for crosscheck) ----
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'nJet25' '3,3.5,6.5' $SYSTS $OPT_2L -o ${X}BCat_nJet_pos $POS $BAny; 
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'nJet25' '3,3.5,6.5' $SYSTS $OPT_2L -o ${X}BCat_nJet_neg $NEG $BAny; 

        # ---- unseparated (for making post-fit plots) ----
        python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA $MVA_2L $BAny;
        python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'nJet25' '3,3.5,6.5' $SYSTS $OPT_2L -o ${X}BCat_nJet $BAny; 

        # ----- 3-jet category (for more fits) ----
        J3="-R 4j 3j nJet25==3"
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_23j_6var' $J3 '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_3j_MVA_neg $MVA_2L $NEG $BAny;
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_23j_6var' $J3 '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_3j_MVA_pos $MVA_2L $POS $BAny;
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var' $J3 '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_3j_MVA4j $MVA_2L $BAny;

        J4E="-R 4j 4j nJet25==4"
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var' $J4E '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_4je_MVA $MVA_2L $BLoose;
        #break;

        # ----- cross-checks with sip(mu) < 4 and with SUS-13 analysis  ----
        #SIP4="(abs(LepGood1_pdgId)!=13||LepGood1_sip3d<4)&&(abs(LepGood2_pdgId)!=13||LepGood2_sip3d < 4)"
        #python makeShapeCards.py mca-2lss-dataBCat_muSip4.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCatMuSip4_MVA_neg $MVA_2L $NEG $BAny  -A pt2010 sip4 "$SIP4";
        #python makeShapeCards.py mca-2lss-dataBCat_muSip4.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCatMuSip4_MVA_pos $MVA_2L $POS $BAny  -A pt2010 sip4 "$SIP4";
        #python makeShapeCards.py mca-2lss-dataBCat_muSip4.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCatMuSip4_MVA $MVA_2L $BAny           -A pt2010 sip4 "$SIP4";

        #python makeShapeCards.py mca-2lss-dataSUS13.txt bins/${X}_SUS13.txt 'MVA_2LSS_4j_6var'  '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}SUS13_MVA_neg $MVA_2L $NEG $BAny ;
        #python makeShapeCards.py mca-2lss-dataSUS13.txt bins/${X}_SUS13.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}SUS13_MVA_pos $MVA_2L $POS $BAny ;
        #python makeShapeCards.py mca-2lss-dataSUS13.txt bins/${X}_SUS13.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}SUS13_MVA $MVA_2L $BAny          ;

        # ---- other random tests ----
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '3,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA_neg_bt $MVA_2L $NEG $BTight;
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA_pos_bt $MVA_2L $POS $BTight;
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA_neg_pt2010 $MVA_2L $NEG $BAny -R pt2020_htllv100 htll100 'LepGood1_pt+LepGood2_pt+met > 100';
        #python makeShapeCards.py mca-2lss-dataBCat.txt bins/${X}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_MVA_pos_pt2010 $MVA_2L $POS $BAny -R pt2020_htllv100 htll100 'LepGood1_pt+LepGood2_pt+met > 100';
        if [[ "$X" != "2lss_mumu" ]]; then continue; fi
   
 
        if false; then 
        for M in MVA{05,03,00,m05,m03,m07}; do
        #for M in MVA{03,m07,m03}X; do
            if [[ "$X" != "2lss_mumu" ]]; then continue; fi
            # define scale factors
            case $M in
                MVA05)  SFMVA="SF_LepMVATight_2l";;
                MVA03)  SFMVA="sqrt(SF_LepMVATight_2l*SF_LepMVALoose_2l)";;
                MVA00)  SFMVA="SF_LepMVALoose_2l";;
                MVAm03) SFMVA="SF_LepMVALoose_2l";;
                MVAm05) SFMVA="SF_LepMVALoose_2l";;
                MVAm07) SFMVA="sqrt(SF_LepMVALoose_2l)";;
                MVA03X)  SFMVA="sqrt(SF_LepMVATightI_2l*SF_LepMVALooseX_2l)"; OPT_2L="${OPT_2L} --FM sf/t $T/0_moreSFs_v1/sfFriend_{cname}.root" ;;
                MVAm03X) SFMVA="SF_LepMVALooseX_2l";                          OPT_2L="${OPT_2L} --FM sf/t $T/0_moreSFs_v1/sfFriend_{cname}.root" ;;
                MVAm07X) SFMVA="sqrt(SF_LepMVALooseI_2l*SF_LepMVALooseX_2l)"; OPT_2L="${OPT_2L} --FM sf/t $T/0_moreSFs_v1/sfFriend_{cname}.root" ;;
            esac;
            OPT_2L="${OPT_2L/SF_LepMVATight_2l/$SFMVA}"
            if echo "X$1" | grep -q QMVA; then
            python makeShapeCards.py syst/mca-2lss-dataBCat_${M}.txt syst/${X}_${M}.txt 'MVA_2LSS_4j_6var'  '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_${M}_MVA_neg $MVA_2L $NEG $BAny;
            python makeShapeCards.py syst/mca-2lss-dataBCat_${M}.txt syst/${X}_${M}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_${M}_MVA_pos $MVA_2L $POS $BAny;
            else
            python makeShapeCards.py syst/mca-2lss-dataBCat_${M}.txt syst/${X}_${M}.txt 'MVA_2LSS_4j_6var'  '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_${M}_MVA $MVA_2L $BAny;
            fi;
            #python makeShapeCards.py syst/mca-2lss-dataBCat_${M}.txt syst/${X}_${M}.txt 'MVA_2LSS_23j_6var' $J3 '4,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_3j_${M}_MVA_neg $MVA_2L $NEG $BAny;
            #python makeShapeCards.py syst/mca-2lss-dataBCat_${M}.txt syst/${X}_${M}.txt 'MVA_2LSS_23j_6var' $J3 '6,-0.8,0.8' $SYSTS $OPT_2L -o ${X}BCat_3j_${M}_MVA_pos $MVA_2L $POS $BAny;
        done; fi

        echo "Done at $(date)"
    done
fi


if [[ "$1" == "" || "$1" == "3l_tight" ]]; then
    OPTIONS="${OPTIONS} --FM sf/t $T/0_SFs_v3/sfFriend_{cname}.root --xp FR_data_.*   "
    OPT_3L="${OPTIONS} -W  puWeight*Eff_3lep*SF_btag*SF_LepMVATight_3l*SF_LepTightCharge_3l"
    MVA_3L="-F finalMVA/t $T/0_finalmva_3l/finalMVA_3L_{cname}.root"
    MVA_3L_FIX="-F sf/t $T/1_finalmva_3l/evVarFriend_{cname}.root"
    POS=" -A pt2010 positive LepGood1_charge+LepGood2_charge+LepGood3_charge>0 "
    NEG=" -A pt2010 positive LepGood1_charge+LepGood2_charge+LepGood3_charge<0 "

    # ---- MVA separated by charge (for nominal result) ----
    python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCat_MVA_neg $MVA_3L $NEG $BAny;
    python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCat_MVA_pos $MVA_3L $POS $BAny;

    # ---- n(jet) separated by charge (for crosscheck) ----
    #python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'nJet25' '4,1.5,5.5' $SYSTS $OPT_3L -o 3lBCat_nJet_pos $POS $BAny; 
    #python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'nJet25' '4,1.5,5.5' $SYSTS $OPT_3L -o 3lBCat_nJet_neg $NEG $BAny; 

    # ---- unseparated (for making post-fit plots) ----
    python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCat_MVA $MVA_3L $BAny;
    python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'nJet25' '4,1.5,5.5' $SYSTS $OPT_3L -o 3lBCat_nJet $BAny; 
    
    # ---- Z-peak analysis (for more fits) ---- 
    #python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCat_MVA_Zpeak_neg $MVA_3L $NEG $BAny -I 'Z veto';
    #python makeShapeCards.py mca-3l_tight-dataBCat.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCat_MVA_Zpeak_pos $MVA_3L $POS $BAny -I 'Z veto';

    # ----- cross-checks with sip(mu) < 4 and with SUS-13 analysis  ----
    #SIP4="(abs(LepGood1_pdgId)!=13||LepGood1_sip3d<4)&&(abs(LepGood2_pdgId)!=13||LepGood2_sip3d<4)&&(abs(LepGood3_pdgId)!=13||LepGood3_sip3d<4)"
    #python makeShapeCards.py mca-3l_tight-dataBCat_muSip4.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCatMuSip4_MVA_neg $MVA_3L $NEG $BAny  -A pt2010 sip4 "$SIP4";
    #python makeShapeCards.py mca-3l_tight-dataBCat_muSip4.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCatMuSip4_MVA_pos $MVA_3L $POS $BAny  -A pt2010 sip4 "$SIP4";
    #python makeShapeCards.py mca-3l_tight-dataBCat_muSip4.txt bins/3l_tight.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lBCatMuSip4_MVA     $MVA_3L      $BAny  -A pt2010 sip4 "$SIP4";

    #python makeShapeCards.py mca-3l_tight-dataSUS13.txt bins/3l_tight_SUS13.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lSUS13_MVA_neg $MVA_3L $NEG $BAny;
    #python makeShapeCards.py mca-3l_tight-dataSUS13.txt bins/3l_tight_SUS13.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lSUS13_MVA_pos $MVA_3L $POS $BAny;
    #python makeShapeCards.py mca-3l_tight-dataSUS13.txt bins/3l_tight_SUS13.txt 'FinalMVA_3L_BDTG' '6,-1.0,0.6' $SYSTS $OPT_3L -o 3lSUS13_MVA     $MVA_3L      $BAny;
   echo "Done at $(date)"
fi

if [[ "$1" == "" || "$1" == "4l" ]]; then
    OPTIONS="${OPTIONS} --FM sf/t $T/0_SFs_v3/sfFriend_{cname}.root  "
    OPT_4L="${OPTIONS} -W puWeight*Eff_4lep*SF_btag*SF_LepMVALoose_4l"
    MVA_4L="-F finalMVA/t $T/0_finalmva_4l/finalMVA_4L_{cname}.root"
    python makeShapeCards.py mca-4l-ttscale.txt bins/4l.txt 'nJet25' '3,0.5,3.5' $SYSTS $OPT_4L -o 4l_nJet    $BAny; 
    #python makeShapeCards.py mca-4l-ttscale.txt bins/4l.txt 'nJet25' '3,0.5,3.5' $SYSTS $OPT_4L -o 4l_nJet_bl $BLoose; 
    #python makeShapeCards.py mca-4l-ttscale.txt bins/4l.txt 'nJet25' '3,0.5,3.5' $SYSTS $OPT_4L -o 4l_nJet_bt $BTight; 
    echo "Done at $(date)"
fi
