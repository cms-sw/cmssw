#!/bin/bash
T=/data/gpetrucc/8TeV/ttH/TREES_270213_HADD
#T=/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD
CORE="mcPlots.py -P $T -j 4 --print=png,pdf -l 19.6 -W 'puWeight*Eff_2lep' --showSFitShape -f"
CORE="${CORE} -F newMVA/t $T/0_leptonMVA_v3/lepMVAFriend_{cname}.root"
CORE="${CORE} --sP probeBTag"

ROOT="plots/270213/btagging"

### =============== TOP =================
PROCS="WW,TT,TW,DY,WJets,data"
#PROCS="TT"
TOP="${CORE} mca.txt bins/cr_btag.txt bins/cr_btag_plots.txt  -p '$PROCS' --sp 'TT'  --xf 'DoubleMu.*,DoubleEl.*'"

PT="(abs(LepGood1_charge) != +1)*Jet1_pt + (abs(LepGood1_charge) == +1)*Jet2_pt"
AETA="(abs(LepGood1_charge) != +1)*abs(Jet1_eta) + (abs(LepGood1_charge) == +1)*abs(Jet2_eta)"

PDIR="$ROOT/top_emu/"
echo "python $TOP --pdir $PDIR/all  "
echo "python $TOP --pdir $PDIR/barrel_pt25_40 -A highPt 'bin' '$PT <= 40              && $AETA  < 1.0'"
echo "python $TOP --pdir $PDIR/barrel_pt40_60 -A highPt 'bin' '$PT >  40 && $PT <= 60 && $AETA  < 1.0'"
echo "python $TOP --pdir $PDIR/barrel_pt60_90 -A highPt 'bin' '$PT >  60 && $PT <= 90 && $AETA  < 1.0'"
echo "python $TOP --pdir $PDIR/barrel_pt90_in -A highPt 'bin' '$PT >  90              && $AETA  < 1.0'"
echo "python $TOP --pdir $PDIR/endcap_pt25_40 -A highPt 'bin' '$PT <= 40              && $AETA >= 1.0'"
echo "python $TOP --pdir $PDIR/endcap_pt40_60 -A highPt 'bin' '$PT >  40 && $PT <= 60 && $AETA >= 1.0'"
echo "python $TOP --pdir $PDIR/endcap_pt60_90 -A highPt 'bin' '$PT >  60 && $PT <= 90 && $AETA >= 1.0'"
echo "python $TOP --pdir $PDIR/endcap_pt90_in -A highPt 'bin' '$PT >  90              && $AETA >= 1.0'"

### =============== Z1J =================
PROCS="WW,TT,TW,DY.,WJets,data"
Z1J="${CORE} mca-incl-dyclass.txt bins/cr_untag.txt bins/cr_untag_plots.txt  -p '$PROCS' --sp 'DY.'  --xf 'MuEG.*,DoubleEl.*'"

PT="Jet1_pt"
AETA="abs(Jet1_eta)"

PDIR="$ROOT/zmm_1jet/"
echo "python $Z1J --pdir $PDIR/all  "
echo "python $Z1J --pdir $PDIR/barrel_pt25_40 -A highPt 'bin' '$PT <= 40              && $AETA  < 1.0'"
echo "python $Z1J --pdir $PDIR/barrel_pt40_60 -A highPt 'bin' '$PT >  40 && $PT <= 60 && $AETA  < 1.0'"
echo "python $Z1J --pdir $PDIR/barrel_pt60_90 -A highPt 'bin' '$PT >  60 && $PT <= 90 && $AETA  < 1.0'"
echo "python $Z1J --pdir $PDIR/barrel_pt90_in -A highPt 'bin' '$PT >  90              && $AETA  < 1.0'"
echo "python $Z1J --pdir $PDIR/endcap_pt25_40 -A highPt 'bin' '$PT <= 40              && $AETA >= 1.0'"
echo "python $Z1J --pdir $PDIR/endcap_pt40_60 -A highPt 'bin' '$PT >  40 && $PT <= 60 && $AETA >= 1.0'"
echo "python $Z1J --pdir $PDIR/endcap_pt60_90 -A highPt 'bin' '$PT >  60 && $PT <= 90 && $AETA >= 1.0'"
echo "python $Z1J --pdir $PDIR/endcap_pt90_in -A highPt 'bin' '$PT >  90              && $AETA >= 1.0'"
