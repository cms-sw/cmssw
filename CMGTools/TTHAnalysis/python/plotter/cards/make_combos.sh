if [[ "$1" == "inclusive" ]]; then
combineCards.py ss_ee=2lss_ee_nJet.card.txt   \
                ss_mm=2lss_mumu_nJet.card.txt \
                ss_em=2lss_em_nJet.card.txt     > 2lss_nJet.card.txt
combineCards.py ss_ee=2lss_ee_MVA.card.txt   \
                ss_mm=2lss_mumu_MVA.card.txt \
                ss_em=2lss_em_MVA.card.txt     > 2lss_MVA.card.txt
combineCards.py .=2lss_nJet.card.txt tl=3l_nJet.card.txt ql=4l_nJet.card.txt > comb_nJet.card.txt
combineCards.py .=2lss_MVA.card.txt  tl=3l_MVA.card.txt  ql=4l_nJet.card.txt > comb_MVA.card.txt

elif [[ "$1" == "charge" ]]; then
    for X in ee em mumu; do
        #combineCards.py ss_${X/mumu/mm}_pos=ttH_2lss_${X}_nJet_pos.card.txt \
        #                ss_${X/mumu/mm}_neg=ttH_2lss_${X}_nJet_neg.card.txt > ttH_2lss_${X}_QnJet.card.txt
        combineCards.py ss_${X/mumu/mm}_pos=ttH_2lss_${X}_MVA_pos.card.txt \
                        ss_${X/mumu/mm}_neg=ttH_2lss_${X}_MVA_neg.card.txt > ttH_2lss_${X}_QMVA.card.txt
    done
    #combineCards.py .=ttH_2lss_ee_QnJet.card.txt   \
    #                .=ttH_2lss_mumu_QnJet.card.txt \
    #                .=ttH_2lss_em_QnJet.card.txt     > ttH_2lss_QnJet.card.txt
    combineCards.py .=ttH_2lss_ee_QMVA.card.txt   \
                    .=ttH_2lss_mumu_QMVA.card.txt \
                    .=ttH_2lss_em_QMVA.card.txt     > ttH_2lss_QMVA.card.txt

    #combineCards.py tl_pos=ttH_3l_nJet_pos.card.txt \
    #                tl_neg=ttH_3l_nJet_neg.card.txt > ttH_3l_QnJet.card.txt
    combineCards.py tl_pos=ttH_3l_MVA_pos.card.txt \
                    tl_neg=ttH_3l_MVA_neg.card.txt > ttH_3l_QMVA.card.txt
    #combineCards.py .=ttH_2lss_QnJet.card.txt tl=ttH_3l_QnJet.card.txt ql=ttH_4l_nJet.card.txt > comb_QnJet.card.txt
    combineCards.py .=ttH_2lss_QMVA.card.txt  .=ttH_3l_QMVA.card.txt  ql=ttH_4l_nJet.card.txt > comb_QMVA.card.txt
elif [[ "$1" == "chargeBCat" ]]; then
    for X in ee em mumu; do
        #combineCards.py ss_${X/mumu/mm}_pos=ttH_2lss_${X}BCat_nJet_pos.card.txt \
        #                ss_${X/mumu/mm}_neg=ttH_2lss_${X}BCat_nJet_neg.card.txt > ttH_2lss_${X}BCat_QnJet.card.txt
        combineCards.py ss_${X/mumu/mm}_pos=ttH_2lss_${X}BCat_MVA_pos.card.txt \
                        ss_${X/mumu/mm}_neg=ttH_2lss_${X}BCat_MVA_neg.card.txt > ttH_2lss_${X}BCat_QMVA.card.txt
    done
    #combineCards.py .=ttH_2lss_eeBCat_QnJet.card.txt   \
    #                .=ttH_2lss_mumuBCat_QnJet.card.txt \
    #                .=ttH_2lss_emBCat_QnJet.card.txt     > ttH_2lssBCat_QnJet.card.txt
    combineCards.py .=ttH_2lss_eeBCat_QMVA.card.txt   \
                    .=ttH_2lss_mumuBCat_QMVA.card.txt \
                    .=ttH_2lss_emBCat_QMVA.card.txt     > ttH_2lssBCat_QMVA.card.txt

    #combineCards.py tl_pos=ttH_3lBCat_nJet_pos.card.txt \
    #                tl_neg=ttH_3lBCat_nJet_neg.card.txt > ttH_3lBCat_QnJet.card.txt
    combineCards.py tl_pos=ttH_3lBCat_MVA_pos.card.txt \
                    tl_neg=ttH_3lBCat_MVA_neg.card.txt > ttH_3lBCat_QMVA.card.txt
    #combineCards.py .=ttH_2lssBCat_QnJet.card.txt .=ttH_3lBCat_QnJet.card.txt ql=ttH_4l_nJet.card.txt > combBCat_QnJet.card.txt
    combineCards.py .=ttH_2lssBCat_QMVA.card.txt  .=ttH_3lBCat_QMVA.card.txt  ql=ttH_4l_nJet.card.txt > combBCat_QMVA.card.txt
elif [[ "$1" == "BCat" ]]; then
    combineCards.py .=ttH_2lss_eeBCat_nJet.card.txt   \
                    .=ttH_2lss_mumuBCat_nJet.card.txt \
                    .=ttH_2lss_emBCat_nJet.card.txt     > ttH_2lssBCat_nJet.card.txt
    combineCards.py .=ttH_2lss_eeBCat_MVA.card.txt   \
                    .=ttH_2lss_mumuBCat_MVA.card.txt \
                    .=ttH_2lss_emBCat_MVA.card.txt     > ttH_2lssBCat_MVA.card.txt
    combineCards.py .=ttH_2lssBCat_nJet.card.txt tl=ttH_3lBCat_nJet.card.txt ql=ttH_4l_nJet.card.txt > combBCat_nJet.card.txt
    combineCards.py .=ttH_2lssBCat_MVA.card.txt  tl=ttH_3lBCat_MVA.card.txt  ql=ttH_4l_nJet.card.txt > combBCat_MVA.card.txt
elif [[ "$1" == "chargeBCatMuSip4" ]]; then
    for X in ee em mumu; do
        #combineCards.py ss_${X/mumu/mm}_pos=2lss_${X}BCatMuSip4_nJet_pos.card.txt \
        #                ss_${X/mumu/mm}_neg=2lss_${X}BCatMuSip4_nJet_neg.card.txt > 2lss_${X}BCatMuSip4_QnJet.card.txt
        combineCards.py ss_${X/mumu/mm}_pos=2lss_${X}BCatMuSip4_MVA_pos.card.txt \
                        ss_${X/mumu/mm}_neg=2lss_${X}BCatMuSip4_MVA_neg.card.txt > 2lss_${X}BCatMuSip4_QMVA.card.txt
    done
    #combineCards.py .=2lss_eeBCatMuSip4_QnJet.card.txt   \
    #                .=2lss_mumuBCatMuSip4_QnJet.card.txt \
    #                .=2lss_emBCatMuSip4_QnJet.card.txt     > 2lssBCatMuSip4_QnJet.card.txt
    combineCards.py .=2lss_eeBCatMuSip4_QMVA.card.txt   \
                    .=2lss_mumuBCatMuSip4_QMVA.card.txt \
                    .=2lss_emBCatMuSip4_QMVA.card.txt     > 2lssBCatMuSip4_QMVA.card.txt

    #combineCards.py tl_pos=3lBCatMuSip4_nJet_pos.card.txt \
    #                tl_neg=3lBCatMuSip4_nJet_neg.card.txt > 3lBCatMuSip4_QnJet.card.txt
    combineCards.py tl_pos=3lBCatMuSip4_MVA_pos.card.txt \
                    tl_neg=3lBCatMuSip4_MVA_neg.card.txt > 3lBCatMuSip4_QMVA.card.txt
    #combineCards.py .=2lssBCatMuSip4_QnJet.card.txt tl=3lBCatMuSip4_QnJet.card.txt ql=4l_nJet.card.txt > combBCatMuSip4_QnJet.card.txt
    combineCards.py .=2lssBCatMuSip4_QMVA.card.txt  tl=3lBCatMuSip4_QMVA.card.txt  ql=4l_nJet.card.txt > combBCatMuSip4_QMVA.card.txt

    combineCards.py .=2lss_eeBCatMuSip4_MVA.card.txt   \
                    .=2lss_mumuBCatMuSip4_MVA.card.txt \
                    .=2lss_emBCatMuSip4_MVA.card.txt     > 2lssBCatMuSip4_MVA.card.txt
    combineCards.py .=2lssBCatMuSip4_MVA.card.txt  tl=3lBCatMuSip4_MVA.card.txt  ql=4l_nJet.card.txt > combBCatMuSip4_MVA.card.txt
elif [[ "$1" == "chargeSUS13" ]]; then
    for X in ee em mumu; do
        #combineCards.py ss_${X/mumu/mm}_pos=2lss_${X}SUS13_nJet_pos.card.txt \
        #                ss_${X/mumu/mm}_neg=2lss_${X}SUS13_nJet_neg.card.txt > 2lss_${X}SUS13_QnJet.card.txt
        combineCards.py ss_${X/mumu/mm}_pos=2lss_${X}SUS13_MVA_pos.card.txt \
                        ss_${X/mumu/mm}_neg=2lss_${X}SUS13_MVA_neg.card.txt > 2lss_${X}SUS13_QMVA.card.txt
    done
    #combineCards.py .=2lss_eeSUS13_QnJet.card.txt   \
    #                .=2lss_mumuSUS13_QnJet.card.txt \
    #                .=2lss_emSUS13_QnJet.card.txt     > 2lssSUS13_QnJet.card.txt
    combineCards.py .=2lss_eeSUS13_QMVA.card.txt   \
                    .=2lss_mumuSUS13_QMVA.card.txt \
                    .=2lss_emSUS13_QMVA.card.txt     > 2lssSUS13_QMVA.card.txt

    #combineCards.py tl_pos=3lSUS13_nJet_pos.card.txt \
    #                tl_neg=3lSUS13_nJet_neg.card.txt > 3lSUS13_QnJet.card.txt
    combineCards.py tl_pos=3lSUS13_MVA_pos.card.txt \
                    tl_neg=3lSUS13_MVA_neg.card.txt > 3lSUS13_QMVA.card.txt
    #combineCards.py .=2lssSUS13_QnJet.card.txt tl=3lSUS13_QnJet.card.txt ql=4l_nJet.card.txt > combSUS13_QnJet.card.txt
    combineCards.py .=2lssSUS13_QMVA.card.txt  tl=3lSUS13_QMVA.card.txt  ql=4l_nJet.card.txt > combSUS13_QMVA.card.txt

    combineCards.py .=2lss_eeSUS13_MVA.card.txt   \
                    .=2lss_mumuSUS13_MVA.card.txt \
                    .=2lss_emSUS13_MVA.card.txt     > 2lssSUS13_MVA.card.txt
    combineCards.py .=2lssSUS13_MVA.card.txt  tl=3lSUS13_MVA.card.txt  ql=4l_nJet.card.txt > combSUS13_MVA.card.txt

elif [[ "$1" == "consistency" ]]; then
    for X in combBCat_QMVA.card.txt; do
        text2workspace.py $X -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
                --PO 'map=ss_ee.*/ttH.*:r_ee[1,-6,12]' \
                --PO 'map=ss_em.*/ttH.*:r_em[1,-6,12]' \
                --PO 'map=ss_mm.*/ttH.*:r_mm[1,-6,12]' \
                --PO 'map=tl_.*/ttH.*:r_3l[1,-6,12]' \
                --PO 'map=ql_.*/ttH.*:r_4l[1,-6,12]' \
                -o CCC_${X/.card.txt/}.root
    done
elif [[ "$1" == "unconstrained" ]]; then
    combineCards.py tl_Z_pos=3l_MVA_Zpeak_pos.card.txt \
                    tl_Z_neg=3l_MVA_Zpeak_neg.card.txt > 3l_Zpeak_QMVA.card.txt
    #text2workspace.py 3l_Zpeak_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #            --PO 'map=tl_Z_.*/TTZ.*:r[1,-2,6]' \
    #            -o fitTTZ_3l_QMVA.root
    #combineCards.py .=comb_QMVA.card.txt .=3l_Zpeak_QMVA.card.txt > combZ_QMVA.card.txt
    #for X in ee em mumu; do
    #    combineCards.py ss3_${X/mumu/mm}_pos=2lss_${X}_3j_MVA_pos.card.txt \
    #                    ss3_${X/mumu/mm}_neg=2lss_${X}_3j_MVA_neg.card.txt > 2lss_${X}_3j_QMVA.card.txt
    #done
    #combineCards.py .=2lss_ee_3j_QMVA.card.txt   \
    #                .=2lss_mumu_3j_QMVA.card.txt \
    #               .=2lss_em_3j_QMVA.card.txt     > 2lss_3j_QMVA.card.txt
    #text2workspace.py 2lss_3j_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #            --PO 'map=.*/TTW:r[1,-2,6]' \
    #            --PO 'map=.*/TTWW:1' \
    #            -o fitTTW_3j_QMVA.root
    #combineCards.py .=comb_QMVA.card.txt .=2lss_3j_QMVA.card.txt .=3l_Zpeak_QMVA.card.txt > combZ3j_QMVA.card.txt
    text2workspace.py combZ3j_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
               --PO 'map=.*/ttH.*:r[1,-2,6]' \
               --PO 'map=.*/TTW:r_ttV[1,0,6]' \
               --PO 'map=.*/TTWW:1' \
               --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
               --PO 'map=.*_mm.*/FR_data:r_fake_mu[1,0,10]' \
               --PO 'map=.*_ee.*/FR_data:r_fake_el[1,0,10]' \
               --PO 'map=.*_em.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               --PO 'map=.*tl_.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               -o floatS_FR_Z3j_QMVA.root
    #text2workspace.py combZ3j_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #            --PO 'map=.*/ttH.*:r[1,-2,6]' \
    #            --PO 'map=.*/TTW:r_ttW[1,0,6]' \
    #            --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
    #            --PO 'map=.*/TTWW:1' \
    #            -o floatS_Z3j_QMVA.root
    #text2workspace.py combZ_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #           --PO 'map=.*/ttH.*:r[1,-2,6]' \
    #           --PO 'map=.*/TTW:r_ttV[1,0,6]' \
    #           --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
    #           --PO 'map=.*_mm.*/FR_data:r_fake_mu[1,0,10]' \
    #           --PO 'map=.*_ee.*/FR_data:r_fake_el[1,0,10]' \
    #           --PO 'map=.*_em.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
    #           --PO 'map=.*tl_.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
    #           -o floatS_FR_Z_QMVA.root
    #
    #
#    for X in combZ_QMVA.card.txt; do
#        text2workspace.py $X -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
#                --PO 'map=.*/ttH.*:r[1,-2,12]' \
#                --PO 'map=.*/TTZ:r_ttV[1,0,6]' \
#                --PO 'map=.*/TTWW?:r_ttV[1,0,6]' \
#                --PO 'map=.*/(TT$|FR_data):r_fake[1,0,10]' \
#                -o float3_${X/.card.txt/}.root
#        text2workspace.py $X -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
#                --PO 'map=.*/ttH.*:r[1,-2,12]' \
#                --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
#                --PO 'map=.*/TTW:r_ttW[1,0,6]' \
#                --PO 'map=.*/(TT$|FR_data):r_fake[1,0,10]' \
#                -o float4_${X/.card.txt/}.root
#        text2workspace.py $X -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
#                --PO 'map=.*/ttH.*:r[1,-2,12]' \
#                --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
#                --PO 'map=.*/TTW:r_ttW[1,0,6]' \
#                -o floatS_${X/.card.txt/}.root
#    done
elif [[ "$1" == "unconstrainedBCat" ]]; then
    #combineCards.py tl_Z_pos=3lBCat_MVA_Zpeak_pos.card.txt \
    #                tl_Z_neg=3lBCat_MVA_Zpeak_neg.card.txt > 3lBCat_Zpeak_QMVA.card.txt
    #text2workspace.py 3lBCat_Zpeak_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #            --PO 'map=tl_Z_.*/TTZ.*:r[1,0,6]' \
    #            --X-exclude-nuisance QCDscale_ttZ \
    #            -o fitTTZ_3lBCat_QMVA.root
    #combineCards.py .=combBCat_QMVA.card.txt .=3lBCat_Zpeak_QMVA.card.txt > combZBCat_QMVA.card.txt
    #for X in ee em mumu; do
    #    combineCards.py ss3_${X/mumu/mm}_pos=2lss_${X}BCat_3j_MVA_pos.card.txt \
    #                    ss3_${X/mumu/mm}_neg=2lss_${X}BCat_3j_MVA_neg.card.txt > 2lss_${X}BCat_3j_QMVA.card.txt
    #done
    #combineCards.py .=2lss_eeBCat_3j_QMVA.card.txt   \
    #                .=2lss_mumuBCat_3j_QMVA.card.txt \
    #               .=2lss_emBCat_3j_QMVA.card.txt     > 2lssBCat_3j_QMVA.card.txt
    #text2workspace.py 2lssBCat_3j_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #            --X-exclude-nuisance QCDscale_ttW \
    #            --PO 'map=.*/TTW:r[1,0,4]' \
    #            --PO 'map=.*/TTWW:1' \
    #            -o fitTTWBCat_3j_QMVA.root
    #combineCards.py .=combBCat_QMVA.card.txt .=2lssBCat_3j_QMVA.card.txt .=3lBCat_Zpeak_QMVA.card.txt > combZ3jBCat_QMVA.card.txt
    text2workspace.py combZ3jBCat_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
               --X-exclude-nuisance QCDscale_tt[ZW] \
               --X-exclude-nuisance CMS_ttHl_FR[em]_norm \
               --PO 'map=.*/ttH.*:r[1,-2,6]' \
               --PO 'map=.*/TTW:r_ttW[1,0,6]' \
               --PO 'map=.*/TTWW:1' \
               --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
               --PO 'map=.*_mm.*/FR_data:r_fake_mu[1,0,10]' \
               --PO 'map=.*_ee.*/FR_data:r_fake_el[1,0,10]' \
               --PO 'map=.*_em.*/FR_data:r_fake_em=expr::r_fake_em("0.40*@0+0.60*@1",r_fake_mu,r_fake_el)' \
               --PO 'map=.*tl_.*/FR_data:r_fake_3l=expr::r_fake_3l("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               -o floatS_FR_Z3jBCat_QMVA.root
    #text2workspace.py combZ3jBCat_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #           --X-exclude-nuisance QCDscale_tt[ZW] \
    #            --PO 'map=.*/ttH.*:r[1,-2,6]' \
    #            --PO 'map=.*/TTW:r_ttW[1,0,6]' \
    #            --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
    #            --PO 'map=.*/TTWW:1' \
    #            -o floatS_Z3jBCat_QMVA.root
    #text2workspace.py combZBCat_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #           --X-exclude-nuisance QCDscale_tt[ZW] \
    #           --X-exclude-nuisance CMS_ttHl_FR[em]_norm \
    #           --PO 'map=.*/ttH.*:r[1,-2,6]' \
    #           --PO 'map=.*/TTW:r_ttW[1,0,6]' \
    #           --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
    #           --PO 'map=.*/TTWW:1' \
    #           --PO 'map=.*_mm.*/FR_data:r_fake_mu[1,0,10]' \
    #           --PO 'map=.*_ee.*/FR_data:r_fake_el[1,0,10]' \
    #           --PO 'map=.*_em.*/FR_data:r_fake_em=expr;;r_fake_em("0.40*@0+0.60*@1",r_fake_mu,r_fake_el)' \
    #           --PO 'map=.*tl_.*/FR_data:r_fake_3l=expr;;r_fake_3l("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
    #           -o floatS_FR_ZBCat_QMVA.root
    #text2workspace.py combZBCat_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #           --X-exclude-nuisance QCDscale_tt[ZW] \
    #           --PO 'map=.*/ttH.*:r[1,-2,6]' \
    #           --PO 'map=.*/TTW:r_ttW[1,0,6]' \
    #           --PO 'map=.*/TTWW:1' \
    #           --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
    #           -o floatS_ZBCat_QMVA.root
elif [[ "$1" == "teaser" ]]; then
    for X in ee em mumu; do
        combineCards.py ss_${X/mumu/mm}_pos_bl=2lss_${X}_MVA_pos_bl.card.txt  ss_${X/mumu/mm}_pos_bt=2lss_${X}_MVA_pos_bt.card.txt \
                        ss_${X/mumu/mm}_neg_bl=2lss_${X}_MVA_neg_bl.card.txt  ss_${X/mumu/mm}_neg_bt=2lss_${X}_MVA_neg_bt.card.txt > 2lss_${X}_QBMVA.card.txt
        combineCards.py ss3_${X/mumu/mm}_pos_bl=2lss_${X}_3j_MVA_pos_bl.card.txt  ss3_${X/mumu/mm}_pos_bt=2lss_${X}_3j_MVA_pos_bt.card.txt \
                        ss3_${X/mumu/mm}_neg_bl=2lss_${X}_3j_MVA_neg_bl.card.txt  ss3_${X/mumu/mm}_neg_bt=2lss_${X}_3j_MVA_neg_bt.card.txt > 2lss_${X}_3j_QBMVA.card.txt
    done
    combineCards.py .=2lss_ee_QBMVA.card.txt   \
                    .=2lss_mumu_QBMVA.card.txt \
                    .=2lss_em_QBMVA.card.txt     > 2lss_QBMVA.card.txt
    combineCards.py .=2lss_ee_3j_QBMVA.card.txt   \
                    .=2lss_mumu_3j_QBMVA.card.txt \
                    .=2lss_em_3j_QBMVA.card.txt     > 2lss_3j_QBMVA.card.txt
    combineCards.py tl_pos_bl=3l_MVA_pos_bl.card.txt  tl_pos_bt=3l_MVA_pos_bt.card.txt \
                    tl_neg_bl=3l_MVA_neg_bl.card.txt  tl_neg_bt=3l_MVA_neg_bt.card.txt > 3l_QBMVA.card.txt
    combineCards.py tl_Z_pos_bl=3l_MVA_Zpeak_pos_bl.card.txt  tl_Z_pos_bt=3l_MVA_Zpeak_pos_bt.card.txt \
                    tl_Z_neg_bl=3l_MVA_Zpeak_neg_bl.card.txt  tl_Z_neg_bt=3l_MVA_Zpeak_neg_bt.card.txt > 3l_Zpeak_QBMVA.card.txt
    text2workspace.py 3l_Zpeak_QBMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
                --PO 'map=tl_Z_.*/TTZ.*:r[1,-2,6]' \
                -o fitTTZ_3l_QBMVA.root
    text2workspace.py 2lss_3j_QBMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
                --PO 'map=.*/TTW:r[1,-2,6]' \
                --PO 'map=.*/TTWW:1' \
                -o fitTTW_3j_QBMVA.root
    combineCards.py .=2lss_QBMVA.card.txt .=3l_QBMVA.card.txt  ql=4l_nJet.card.txt > comb_QBMVA.card.txt
    combineCards.py .=comb_QBMVA.card.txt .=2lss_3j_QBMVA.card.txt .=3l_Zpeak_QBMVA.card.txt > combZ3j_QBMVA.card.txt
    text2workspace.py combZ3j_QBMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
               --PO 'map=.*/ttH.*:r[1,-2,6]' \
               --PO 'map=.*/TTW:r_ttV[1,0,6]' \
               --PO 'map=.*/TTWW:1' \
               --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
               --PO 'map=.*_mm.*/FR_data:r_fake_mu[1,0,10]' \
               --PO 'map=.*_ee.*/FR_data:r_fake_el[1,0,10]' \
               --PO 'map=.*_em.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               --PO 'map=.*tl_.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               -o floatS_FR_Z3j_QBMVA.root
elif [[ "$1" == "mc" ]]; then
    #for X in ee em mumu; do
    #    combineCards.py ss3_${X/mumu/mm}_pos=2lss_${X}_MC_3j_MVA_pos.card.txt \
    #                    ss3_${X/mumu/mm}_neg=2lss_${X}_MC_3j_MVA_neg.card.txt > 2lss_${X}_MC_3j_QMVA.card.txt
    #    combineCards.py ss4_${X/mumu/mm}_pos=2lss_${X}_MC_MVA_pos.card.txt \
    #                    ss4_${X/mumu/mm}_neg=2lss_${X}_MC_MVA_neg.card.txt > 2lss_${X}_MC_4j_QMVA.card.txt
    #done
    #combineCards.py .=2lss_ee_MC_3j_QMVA.card.txt   \
    #                .=2lss_mumu_MC_3j_QMVA.card.txt \
    #                .=2lss_em_MC_3j_QMVA.card.txt     > 2lss_MC_3j_QMVA.card.txt
    #combineCards.py .=2lss_ee_MC_4j_QMVA.card.txt   \
    #                .=2lss_mumu_MC_4j_QMVA.card.txt \
    #                .=2lss_em_MC_4j_QMVA.card.txt     > 2lss_MC_4j_QMVA.card.txt
    #combineCards.py .=2lss_MC_3j_QMVA.card.txt .=2lss_MC_4j_QMVA.card.txt > 2lss_MC_QMVA.card.txt
    #combineCards.py tl_pos=3l_MC_MVA_pos.card.txt \
    #                tl_neg=3l_MC_MVA_neg.card.txt > 3l_MC_QMVA.card.txt
    #combineCards.py .=2lss_MC_QMVA.card.txt .=3l_MC_QMVA.card.txt .=4l_nJet.card.txt > comb_MC_QMVA.card.txt
    #for X in 2lss_MC_QMVA.card.txt comb_MC_QMVA.card.txt; do
    #text2workspace.py ${X} -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
    #           --PO 'map=.*/ttH.*:r[1,-2,6]' \
    #           --PO 'map=.*_mm.*/Fakes:r_fake_mu[1,0,10]' \
    #           --PO 'map=.*_ee.*/Fakes:r_fake_el[1,0,10]' \
    #           --PO 'map=.*_em.*/Fakes:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
    #           --PO 'map=.*tl_.*/Fakes:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
    #           -o ${X/.card.txt}.root
    #done
    combineCards.py tl_Z_pos=3l_MC_MVA_Zpeak_pos.card.txt \
                    tl_Z_neg=3l_MC_MVA_Zpeak_neg.card.txt > 3l_MC_Zpeak_QMVA.card.txt
    combineCards.py .=comb_MC_QMVA.card.txt .=3l_MC_Zpeak_QMVA.card.txt  > combZ_MC_QMVA.card.txt
    text2workspace.py combZ_MC_QMVA.card.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
               --PO 'map=.*/ttH.*:r[1,-2,6]' \
               --PO 'map=.*/TTW:r_ttV[1,0,6]' \
               --PO 'map=.*/TTZ:r_ttZ[1,0,6]' \
               --PO 'map=.*_mm.*/Fakes:r_fake_mu[1,0,10]' \
               --PO 'map=.*_ee.*/Fakes:r_fake_el[1,0,10]' \
               --PO 'map=.*_em.*/Fakes:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               --PO 'map=.*tl_.*/Fakes:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
               -o floatS_FR_MC_Z_QMVA.root
elif [[ "$1" == "test" ]]; then
        X=2lss_MVA.card.txt
        text2workspace.py $X -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
                --PO 'map=.*/ttH.*:r[1,-2,12]' \
                --PO 'map=.*_mm.*/FR_data:r_fake_mu[1,0,10]' \
                --PO 'map=.*_ee.*/FR_data:r_fake_el[1,0,10]' \
                --PO 'map=.*_em.*/FR_data:r_fake_em=expr;;r_fake_em("0.45*@0+0.55*@1",r_fake_mu,r_fake_el)' \
                -o test.root
fi



##   #combineCards.py 2lss_{ee,em,mumu}_nJet.card.txt > 2lss_nJet.card.txt
##   #combineCards.py 2lss_{ee,em,mumu}_nJet_{pos,neg}_{bl,bt}.card.txt > 2lss_QBnJet.card.txt
##   #combineCards.py 2lss_{ee,em,mumu}_nJet_{pos,neg}_{bl,btFRMC}.card.txt > 2lss_QBnJetFRMC.card.txt
##   combineCards.py 2lss_{ee,em,mumu}_nJet_{pos,neg}.card.txt > 2lss_QnJet.card.txt
##   
##   #combineCards.py 2lss_{ee,em,mumu}_MVA_4j_6v.card.txt > 2lss_MVA.card.txt
##   #combineCards.py 2lss_{ee,em,mumu}_MVA_4j_6v_{pos,neg}_{bl,bt}.card.txt  > 2lss_QBMVA.card.txt
##   #combineCards.py 2lss_{ee,em,mumu}_MVA_4j_6v_{pos,neg}_{bl,btFRMC}.card.txt  > 2lss_QBMVAFRMC.card.txt
##   combineCards.py 2lss_{ee,em,mumu}_MVA_4j_6v_{pos,neg}.card.txt  > 2lss_QMVA.card.txt
##   for X in ee em mumu; do
##       #combineCards.py 2lss_${X}_nJet_{pos,neg}_{bl,btFRMC}.card.txt > 2lss_${X}_QBnJetFRMC.card.txt
##       #combineCards.py 2lss_${X}_MVA_4j_6v_{pos,neg}_{bl,btFRMC}.card.txt  > 2lss_${X}_QBMVAFRMC.card.txt
##       combineCards.py 2lss_${X}_nJet_{pos,neg}.card.txt > 2lss_${X}_QnJet.card.txt
##       combineCards.py 2lss_${X}_MVA_4j_6v_{pos,neg}.card.txt  > 2lss_${X}_QMVA.card.txt
##   done
##   
##   #combineCards.py 3l_nJet_{pos,neg}_{bl,btFRMC}.card.txt  > 3l_QBnJetFRMC.card.txt
##   #combineCards.py 3l_MVA_{pos,neg}_{bl,bt}.card.txt   > 3l_QBMVA.card.txt
##   #combineCards.py 3l_MVA_{pos,neg}_{bl,btFRMC}.card.txt   > 3l_QBMVAFRMC.card.txt
##   combineCards.py 3l_nJet_{pos,neg}.card.txt  > 3l_QnJet.card.txt
##   combineCards.py 3l_MVA_{pos,neg}.card.txt   > 3l_QMVA.card.txt
##   
##   #combineCards.py 4l_nJet_{bl,bt}.card.txt  > 4l_BnJet.card.txt
##   
##   #combineCards.py 2lss_nJet.card.txt 3l_nJet.card.txt 4l_nJet.card.txt > comb_nJet.card.txt
##   #combineCards.py 2lss_QBnJet.card.txt 3l_QBnJet.card.txt 4l_BnJet.card.txt > comb_QBnJet.card.txt
##   #combineCards.py 2lss_QBnJetFRMC.card.txt 3l_QBnJetFRMC.card.txt 4l_BnJet.card.txt > comb_QBnJetFRMC.card.txt
##   #combineCards.py 2lss_MVA.card.txt 3l_MVA.card.txt 4l_nJet.card.txt > comb_MVA.card.txt
##   #combineCards.py 2lss_QBMVA.card.txt 3l_QBMVA.card.txt 4l_BnJet.card.txt > comb_QBMVA.card.txt
##   #combineCards.py 2lss_QBMVAFRMC.card.txt 3l_QBMVAFRMC.card.txt 4l_BnJet.card.txt > comb_QBMVAFRMC.card.txt
##   combineCards.py 2lss_QnJet.card.txt 3l_QnJet.card.txt 4l_BnJet.card.txt > comb_QnJet.card.txt
##   combineCards.py 2lss_QMVA.card.txt 3l_QMVA.card.txt 4l_BnJet.card.txt > comb_QMVA.card.txt
