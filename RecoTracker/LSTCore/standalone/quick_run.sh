################################################################
# File created by Kasia so she can automate making plots
################################################################

# Make room for new LSTNtuple.root and LSTNumDen.root
rm -r LST*.root

# Compile
lst_make_tracklooper -mcC;

# Takes new_tree.root as input and produces LSTNtuple.root, which includes only the relevant branches + reconstructed tracks
lst -i QCD1800-2400PU0 -l -o LSTNtuple.root;

# Creates LSTNumDen.root, a collection of all the numerator/denominator histograms
createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root;

# Creates plots. The --indivdual tag produces the *total* efficiency plot, --pt_cut changes the label for the pt cut
python3 efficiency/python/lst_plot_performance.py --individual --pt_cut 0.9 LSTNumDen.root -t "myWork";
# python3 efficiency/python/lst_plot_performance.py --pt_cut 2 LSTNumDen.root -t "mywork001";

# Saves relevant plot to my public html page
# cp /mnt/data1/kk829/cmssw/RecoTracker/LSTCore/standalone/performance/mywork001_028aa9D-PU200/mtv/var/TC_base_0_0_eff_rjet.png ~/public_html/recent/TC_base_0_0_pT09_jet-pT100_etacut.png