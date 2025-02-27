# File created by Kasia so she can automate making plots a bit
# Yes, I know about lst_run, but I have it set up how I like, ok?

rm -r LST*.root
lst_make_tracklooper -mcC;
lst -i PU200 -l -o LSTNtuple.root;
createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root;
python3 efficiency/python/lst_plot_performance.py --individual --pt_cut 50 LSTNumDen.root -t "mywork021z";
cp /mnt/data1/kk829/cmssw/RecoTracker/LSTCore/standalone/performance/mywork021z_a3119eD-PU200/mtv/var/TC_base_0_0_eff_rjet.png ~/public_html/all_rjet_1_zoom/TC_base_0_pT50_jet-pT50_etacut.png