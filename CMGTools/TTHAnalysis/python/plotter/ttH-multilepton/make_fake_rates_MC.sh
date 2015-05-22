################################
#  use mcEfficiencies.py to make plots of the fake rate
################################

BCORE=" --s2v --tree treeProducerSusyMultilepton object-studies/lepton-mca.txt object-studies/lepton-perlep.txt  "
BASE="python mcEfficiencies.py $BCORE --ytitle 'Fake rate'  "
PBASE="plots/72X/ttH/leptons/"
BG=" "
if [[ "$1" == "-b" ]]; then BG=" & "; shift; fi

what=$1;
case $what in
    iso|isosip)
        T="/afs/cern.ch/user/g/gpetrucc/w/TREES_72X_210315_NoIso"
        if [[ "$what" == "iso" ]]; then
            SipDen="-A pt20 den 'LepGood_sip3d < 4'"; Num="relIso03_01"
        elif [[ "$what" == "isosip" ]]; then
            SipDen="-A pt20 den 'LepGood_sip3d < 6'"; Num="AND_relIso03_01_SIP4"
        else
            exit 1
        fi;
        B0="$BASE -P $T  ttH-multilepton/make_fake_rates_sels.txt ttH-multilepton/make_fake_rates_xvars.txt --groupBy cut --sP ${Num} " 
        B0="$B0  --legend=TL  --yrange 0 0.6 --showRatio --ratioRange 0.31 1.69 --xcut 10 999 "
        JetDen="-A pt20 jet 'nJet40 >= 1 && minMllAFAS > 12'"
        SelDen="-A pt20 den '(LepGood_relIso03 < 0.5)'"
        CommonDen="${SipDen} ${JetDen} ${SelDen}"
        MuDen="${CommonDen} -A pt20 den 'LepGood_mediumMuonId>0' -A pt20 mc 'LepGood_pt > 5' "
        ElDen="${CommonDen} -I mu -A pt20 den 'LepGood_mvaIdPhys14>=0.73+(0.57-0.73)*(abs(LepGood_eta)>0.8)+(+0.05-0.57)*(abs(LepGood_eta)>1.479) && LepGood_convVeto && LepGood_tightCharge >= 2 && LepGood_lostHits == 0'  -A pt20 mc 'LepGood_pt > 7'"
        MuFakeVsPt="$B0 $MuDen   --sP pt_.*,ptGZ_.* " 
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red --sp 'TT.*' -o $PBASE/$what/mu_a_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5' ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red --sp 'TT.*' -o $PBASE/$what/mu_a_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5' ${BG} )"
        ElFakeVsPt="$B0 $ElDen  --sP pt_.*,ptGZ_.* " 
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red --sp 'TT.*' -o $PBASE/$what/el_a_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red --sp 'TT.*' -o $PBASE/$what/el_a_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
        ;;
    mvaTTH)
        T="/afs/cern.ch/user/g/gpetrucc/w/TREES_72X_210315_NoIso"
        WP=08; if [[ "$2" != "" ]]; then WP=$2; fi;
        case $WP in
            08)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.3) && LepGood_sip3d < 4'"; Num="mvaTTH_$WP"; MuIdDen=1;;
            06)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6'"; Num="mvaTTH_$WP"; MuIdDen=1;;
            06i) SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 4'"; Num="mvaTTH_$WP"; MuIdDen=0;;
        esac
        B0="$BASE -P $T  ttH-multilepton/make_fake_rates_sels.txt ttH-multilepton/make_fake_rates_xvars.txt --groupBy cut --sP ${Num} " 
        B0="$B0  --legend=TL  --yrange 0 0.4 --showRatio --ratioRange 0.31 1.69 --xcut 10 999 "
        JetDen="-A pt20 jet 'nJet40 >= 1 && minMllAFAS > 12'"
        CommonDen="${JetDen} ${SelDen}"
        MuDen="${CommonDen} -A pt20 den 'LepGood_mediumMuonId>=${MuIdDen}' -A pt20 mc 'LepGood_pt > 7' "
        ElDen="${CommonDen} -I mu -A pt20 den 'LepGood_convVeto && LepGood_tightCharge >= 2 && LepGood_lostHits == 0'  -A pt20 mc 'LepGood_pt > 7'"
        BDen="-A pt20 bjet 'nBJetMedium40 <= 1 && nBJetMedium25 == 1'"
        MuFakeVsPt="$B0 $MuDen ${BDen} --sP pt_coarse,ptJI_mvaTTH${WP%%i}_.* " 
        ElFakeVsPt="$B0 $ElDen ${BDen} --sP pt_coarse,ptJI_mvaTTH${WP%%i}_.* " 
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red --sp 'TT.*' -o $PBASE/$what/mu_wp${WP}_a_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red --sp 'TT.*' -o $PBASE/$what/mu_wp${WP}_a_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red --sp 'TT.*' -o $PBASE/$what/el_wp${WP}_a_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red --sp 'TT.*' -o $PBASE/$what/el_wp${WP}_a_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
        BDen="-A pt20 bjet 'nBJetMedium40 != 1 && nBJetMedium40 == nBJetMedium25'"
        MuFakeVsPt="$B0 $MuDen ${BDen} --sP pt_coarse,ptJI_mvaTTH${WP%%i}_.* " 
        ElFakeVsPt="$B0 $ElDen ${BDen} --sP pt_coarse,ptJI_mvaTTH${WP%%i}_.* " 
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red --sp 'TT.*' -o $PBASE/$what/mu_wp${WP}_b_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red --sp 'TT.*' -o $PBASE/$what/mu_wp${WP}_b_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red --sp 'TT.*' -o $PBASE/$what/el_wp${WP}_b_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red --sp 'TT.*' -o $PBASE/$what/el_wp${WP}_b_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
        ;;
    mvaTTH-suite-QCD)
        BCORE="${BCORE/object-studies\/lepton-mca.txt/ttH-multilepton/lepton-mca-frstudies.txt}" 
        BASE="${BASE/object-studies\/lepton-mca.txt/ttH-multilepton/lepton-mca-frstudies.txt}" 
        T="/afs/cern.ch/user/g/gpetrucc/w/TREES_72X_210315_NoIso"
        WP=08; if [[ "$2" != "" ]]; then WP=$2; fi;
        case $WP in
            08)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.3) && LepGood_sip3d < 4'"; Num="mvaTTH_$WP"; MuIdDen=1;;
            06)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6'"; Num="mvaTTH_$WP"; MuIdDen=1;;
            06i) SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 4'"; Num="mvaTTH_$WP"; MuIdDen=0;;
        esac
        B0="$BASE -P $T ttH-multilepton/make_fake_rates_sels.txt ttH-multilepton/make_fake_rates_xvars.txt --groupBy cut --sP ${Num} " 
        B0="$B0 --legend=TL  --yrange 0 0.4 --showRatio --ratioRange 0.31 1.69 --xcut 10 999 "
        JetDen="-A pt20 mll 'minMllAFAS > 12'"
        CommonDen="${JetDen} ${SelDen}"
        MuDen="${CommonDen} -A pt20 den 'LepGood_mediumMuonId>=${MuIdDen} && LepGood_pt > 5'  "
        ElDen="${CommonDen} -I mu -A pt20 den 'LepGood_convVeto && LepGood_tightCharge >= 2 && LepGood_lostHits == 0 && LepGood_pt > 7' "
        PtJBin=" -A pt20 ptj20 'LepGood_pt*if3(LepGood_mvaTTH>${WP%%i}, 1.0, 0.85/LepGood_jetPtRatio) > 20'"
        RunMCA="python mcAnalysis.py -G -e -f -P $T $BCORE $PtJBin -j 3"
        for RVar in 25 30 35 40 50 60; do
            for BVar in bVeto bTag bTight bAny; do
                case $BVar in
                    bAny)   BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar ' " ;;
                    bVeto)  BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar && LepGood_awayJet_btagCSV < 0.423' " ;;
                    bTag)   BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar && LepGood_awayJet_btagCSV > 0.423' " ;;
                    bTight) BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar && LepGood_awayJet_btagCSV > 0.941' " ;;
                esac;
                MuFakeVsPt="$B0 $MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i}_.*' --sp TT_red " 
                ElFakeVsPt="$B0 $ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i}_.*' --sp TT_red " 
                Me="wp${WP}_rec${RVar}_${BVar}"
                echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
                echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
                echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
                echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
                echo "( $RunMCA $MuDen ${BDen} -p QCDMu_.jets -R pt20 eta 'abs(LepGood_eta)<1.5' > $PBASE/$what/mu_${Me}_eta_00_15.txt  ${BG} )"
                # Now the "triggering low pt leg"
                MuFakeVsPt="$B0 $MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 8' " 
                ElFakeVsPt="$B0 $ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 12' " 
                Me="wp${WP}_rec${RVar}_${BVar}_trig2l"
                echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
                echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
                echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
                echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
                echo "( $RunMCA $MuDen ${BDen} -p QCDMu_.jets -R pt20 eta 'abs(LepGood_eta)<1.5' > $PBASE/$what/mu_${Me}_eta_00_15.txt  ${BG} )"
                # Now the "triggering low pt leg"
                MuFakeVsPt="$B0 $MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 17' " 
                ElFakeVsPt="$B0 $ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 23' " 
                Me="wp${WP}_rec${RVar}_${BVar}_trig2h"
                echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
                echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
                echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
                echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
                echo "( $RunMCA $MuDen ${BDen} -p QCDMu_.jets -R pt20 eta 'abs(LepGood_eta)<1.5' > $PBASE/$what/mu_${Me}_eta_00_15.txt  ${BG} )"
            done
        done
        ;;
    mvaTTH-test-QCD)
        BCORE="${BCORE/object-studies\/lepton-mca.txt/ttH-multilepton/lepton-mca-frstudies.txt}" 
        BASE="${BASE/object-studies\/lepton-mca.txt/ttH-multilepton/lepton-mca-frstudies.txt}" 
        T="/afs/cern.ch/user/g/gpetrucc/w/TREES_72X_210315_NoIso"
        WP=08; if [[ "$2" != "" ]]; then WP=$2; fi;
        case $WP in
            08)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.3) && LepGood_sip3d < 4'"; Num="mvaTTH_$WP"; MuIdDen=1;;
            06)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 8'"; Num="mvaTTH_$WP"; MuIdDen=1;;
            06i) SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 4'"; Num="mvaTTH_$WP"; MuIdDen=0;;
            06il) SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 8'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06im) SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.941'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06it) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.814'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06is0) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.814 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.0*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06is1) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.814 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.5*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06is1b) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.5*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06is1c) SelDen="-A pt20 den '(LepGood_relIso03 < 0.5) && LepGood_sip3d < 4 && LepGood_jetBTagCSV < 0.814 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.5*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06is2) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.814 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.6*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06is3) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.814 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.7*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
        esac
        B0="$BASE -P $T ttH-multilepton/make_fake_rates_sels.txt ttH-multilepton/make_fake_rates_xvars.txt --groupBy cut --sP ${Num} " 
        B0="$B0 --legend=TL  --yrange 0 0.4 --showRatio --ratioRange 0.31 1.69 --xcut 10 999 "
        JetDen="-A pt20 mll 'minMllAFAS > 12'"
        CommonDen="${JetDen} ${SelDen}"
        MuDen="${CommonDen} -A pt20 den 'LepGood_mediumMuonId>=${MuIdDen} && LepGood_pt > 5'  "
        ElDen="${CommonDen} -I mu -A pt20 den 'LepGood_convVeto && LepGood_tightCharge >= 2 && LepGood_lostHits == 0 && LepGood_pt > 7' "
        RVar=40; BVar=bAny;
        case $BVar in
            bAny)   BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar ' " ;;
            bVeto)  BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar && LepGood_awayJet_btagCSV < 0.423' " ;;
            bTag)   BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar && LepGood_awayJet_btagCSV > 0.423' " ;;
            bTight) BDen="-A pt20 jet 'LepGood_awayJet_pt > $RVar && LepGood_awayJet_btagCSV > 0.941' " ;;
        esac;
        Me="wp${WP}_rec${RVar}_${BVar}"
        # ttbar variations, zoomed
        MuFakeVsPt="$MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_unity' --sp TT_red " 
        ElFakeVsPt="$ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_unity' --sp TT_red " 
        BZ=${B0/ratioRange 0.31 1.69/ratioRange 0.77 1.29}; BZ=${BZ/yrange 0 0.4/yrange 0 0.25}; BZ=${BZ/legend=TL/legend=TR}
        echo "( $BZ $MuFakeVsPt -p TT_red,TT_fw._red  -o $PBASE/$what/mu_${Me}_eta_00_25_ttfws.root  -R pt20 eta 'abs(LepGood_eta)<2.5'   ${BG} )"
        echo "( $B0 $MuFakeVsPt -p TT_red,TT_SS.*_red -o $PBASE/$what/mu_${Me}_eta_00_25_ttvars.root -R pt20 eta 'abs(LepGood_eta)<2.5'   ${BG} )"
        BZ=${B0/ratioRange 0.31 1.69/ratioRange 0.67 1.39}; BZ=${BZ/yrange 0 0.4/yrange 0 0.25}; BZ=${BZ/legend=TL/legend=TR}
        echo "( $BZ $ElFakeVsPt -p TT_red,TT_fw._red  -o $PBASE/$what/el_${Me}_eta_00_15_ttfws.root  -R pt20 eta 'abs(LepGood_eta)<1.479'   ${BG} )"
        echo "( $BZ $ElFakeVsPt -p TT_red,TT_SS.*_red -o $PBASE/$what/el_${Me}_eta_00_15_ttvars.root -R pt20 eta 'abs(LepGood_eta)<1.479'   ${BG} )"
        echo "( $BZ $ElFakeVsPt -p TT_red,TT_fw._red  -o $PBASE/$what/el_${Me}_eta_15_25_ttfws.root  -R pt20 eta 'abs(LepGood_eta)>1.479'   ${BG} )"
        echo "( $BZ $ElFakeVsPt -p TT_red,TT_SS.*_red -o $PBASE/$what/el_${Me}_eta_15_25_ttvars.root -R pt20 eta 'abs(LepGood_eta)>1.479'   ${BG} )"

        MuFakeVsPt="$MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_mid' --sp TT_red " 
        ElFakeVsPt="$ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_mid' --sp TT_red " 
        echo "( $B0 $MuFakeVsPt -p 'TT_red,TT_pt(8|17|24)_red'  -o $PBASE/$what/mu_${Me}_eta_00_25_ttpt.root  -R pt20 eta 'abs(LepGood_eta)<2.5'   ${BG} )"
        echo "( $B0 $ElFakeVsPt -p 'TT_red,TT_pt(12|23|32)_red' -o $PBASE/$what/el_${Me}_eta_00_15_ttpt.root  -R pt20 eta 'abs(LepGood_eta)<1.479'   ${BG} )"

        MuFakeVsPt="$MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_coarse' --sp TT_red " 
        ElFakeVsPt="$ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_coarse' --sp TT_red " 
        echo "( $B0 $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $B0 $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
        echo "( $B0 $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $B0 $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"

        #AwayJet pt variations
        MuFakeVsPt="$MuDen --sP 'ptJI_mvaTTH${WP%%i*}_coarse' --sp TT_red " 
        ElFakeVsPt="$ElDen --sP 'ptJI_mvaTTH${WP%%i*}_coarse' --sp TT_red " 
        echo "( $B0 $MuFakeVsPt -p 'TT_red,QCDMu_red_aj[2-6].*' -o $PBASE/$what/mu_${Me}_eta_00_15_ajpt.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $B0 $ElFakeVsPt -p 'TT_red,QCDEl_red_aj[2-6].*' -o $PBASE/$what/el_${Me}_eta_00_15_ajpt.root -R pt20 eta 'abs(LepGood_eta)<1.479'   ${BG} )"

        #AwayJet b-tag
        MuFakeVsPt="$MuDen --sP 'ptJI_mvaTTH${WP%%i*}_coarse' --sp TT_red -A pt20 jet 'LepGood_awayJet_pt > 40' " 
        ElFakeVsPt="$ElDen --sP 'ptJI_mvaTTH${WP%%i*}_coarse' --sp TT_red -A pt20 jet 'LepGood_awayJet_pt > 40' " 
        echo "( $B0 $MuFakeVsPt -p 'TT_red,QCDMu_red_ajb.*' -o $PBASE/$what/mu_${Me}_eta_00_15_ajb.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $B0 $MuFakeVsPt -p 'TT_red,QCDEl_red_ajb.*' -o $PBASE/$what/el_${Me}_eta_00_15_ajb.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"


        # Now the "triggering low pt leg"
        MuFakeVsPt="$B0 $MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 8' " 
        ElFakeVsPt="$B0 $ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 12' " 
        Me="wp${WP}_rec${RVar}_${BVar}_trig2l"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
        echo "( $RunMCA $MuDen ${BDen} -p QCDMu_.jets -R pt20 eta 'abs(LepGood_eta)<1.5' > $PBASE/$what/mu_${Me}_eta_00_15.txt  ${BG} )"
        # Now the "triggering low pt leg"
        MuFakeVsPt="$B0 $MuDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 17' " 
        ElFakeVsPt="$B0 $ElDen ${BDen} --sP 'ptJI_mvaTTH${WP%%i*}_.*' --sp TT_red -A pt20 pt10 'LepGood_pt > 23' " 
        Me="wp${WP}_rec${RVar}_${BVar}_trig2h"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_${Me}_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
        echo "( $RunMCA $MuDen ${BDen} -p QCDMu_.jets -R pt20 eta 'abs(LepGood_eta)<1.5' > $PBASE/$what/mu_${Me}_eta_00_15.txt  ${BG} )"
        ;;
    mvaTTH-prod)
        BCORE="${BCORE/object-studies\/lepton-mca.txt/ttH-multilepton/lepton-mca-frstudies.txt}" 
        BASE="${BASE/object-studies\/lepton-mca.txt/ttH-multilepton/lepton-mca-frstudies.txt}" 
        T="/afs/cern.ch/user/g/gpetrucc/w/TREES_72X_210315_NoIso"
        WP=06i; if [[ "$2" != "" ]]; then WP=$2; fi;
        case $WP in
            06i)  SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6                               && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.5*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
            06ib) SelDen="-A pt20 den '(LepGood_relIso03 < 0.4) && LepGood_sip3d < 6 && LepGood_jetBTagCSV < 0.814 && (LepGood_mvaTTH>${WP/i*/} || LepGood_jetPtRatio > 0.5*0.85)'"; Num="mvaTTH_${WP/i*/i}"; MuIdDen=0;;
        esac
        B0="$BASE -P $T  ttH-multilepton/make_fake_rates_sels.txt ttH-multilepton/make_fake_rates_xvars.txt --groupBy cut --sP ${Num} " 
        B0="$B0  --legend=TL  --yrange 0 0.4 --showRatio --ratioRange 0.31 1.69 --xcut 10 999 "
        JetDen="-A pt20 jet 'LepGood_awayJet_pt > 40 && minMllAFAS > 12'"
        CommonDen="${JetDen} ${SelDen}"
        MuDen="${CommonDen} -A pt20 den 'LepGood_mediumMuonId>=${MuIdDen}' -A pt20 mc 'LepGood_pt > 5' "
        ElDen="${CommonDen} -I mu -A pt20 den 'LepGood_convVeto && LepGood_tightCharge >= 2 && LepGood_lostHits == 0'  -A pt20 mc 'LepGood_pt > 7'"
        MuFakeVsPt="$B0 $MuDen --sP ptJI_mvaTTH${WP%%i*}_coarse --sp 'TT.*' " 
        ElFakeVsPt="$B0 $ElDen --sP ptJI_mvaTTH${WP%%i*}_coarse --sp 'TT.*' " 
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.5'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.5'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_00_15.root -R pt20 eta 'abs(LepGood_eta)<1.479' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_15_24.root -R pt20 eta 'abs(LepGood_eta)>1.479' ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_00_15_pt8.root -R pt20 eta 'abs(LepGood_eta)<1.5 && LepGood_pt > 8'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_15_24_pt8.root -R pt20 eta 'abs(LepGood_eta)>1.5 && LepGood_pt > 8'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_00_15_pt12.root -R pt20 eta 'abs(LepGood_eta)<1.479 && LepGood_pt > 12' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_15_24_pt12.root -R pt20 eta 'abs(LepGood_eta)>1.479 && LepGood_pt > 12' ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_00_15_pt17.root -R pt20 eta 'abs(LepGood_eta)<1.5 && LepGood_pt > 17'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_15_24_pt17.root -R pt20 eta 'abs(LepGood_eta)>1.5 && LepGood_pt > 17'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_00_15_pt23.root -R pt20 eta 'abs(LepGood_eta)<1.479 && LepGood_pt > 23' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_15_24_pt23.root -R pt20 eta 'abs(LepGood_eta)>1.479 && LepGood_pt > 23' ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_00_15_pt24.root -R pt20 eta 'abs(LepGood_eta)<1.5 && LepGood_pt > 17'   ${BG} )"
        echo "( $MuFakeVsPt -p TT_red,QCDMu_red -o $PBASE/$what/mu_wp${WP}_a_eta_15_24_pt24.root -R pt20 eta 'abs(LepGood_eta)>1.5 && abs(LepGood_eta)< 2.1 && LepGood_pt > 24'   ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_00_15_pt32.root -R pt20 eta 'abs(LepGood_eta)<1.479 && LepGood_pt > 32' ${BG} )"
        echo "( $ElFakeVsPt -p TT_red,QCDEl_red -o $PBASE/$what/el_wp${WP}_a_eta_15_24_pt32.root -R pt20 eta 'abs(LepGood_eta)>1.479 && abs(LepGood_eta)< 2.1 && LepGood_pt > 32' ${BG} )"
        ;;

esac;
