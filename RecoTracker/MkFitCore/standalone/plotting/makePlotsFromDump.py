import os.path, glob, sys
import ROOT

arch   = sys.argv[1]
sample = sys.argv[2]
build  = sys.argv[3]
suffix = sys.argv[4]

g = ROOT.TFile("test_"+arch+"_"+sample+"_"+build+"_"+suffix+".root","recreate")

# declare hists: reco only
h_MXNH  = ROOT.TH1F("h_MXNH_"+suffix, "nHits/Track", 35, 0, 35)
h_MXPT  = ROOT.TH1F("h_MXPT_"+suffix, "p_{T}^{mkFit}", 100, 0, 100)
h_MXETA = ROOT.TH1F("h_MXETA_"+suffix, "#eta^{mkFit}", 25, -2.5, 2.5)
h_MXPHI = ROOT.TH1F("h_MXPHI_"+suffix, "#phi^{mkFit}", 32, -3.2, 3.2)

h_MXNH.Sumw2()
h_MXPT.Sumw2()
h_MXETA.Sumw2()
h_MXPHI.Sumw2()

# declare hists: diffs 
h_DCNH  = ROOT.TH1F("h_DCNH_"+suffix, "#DeltanHits(mkFit,CMSSW)", 46, -20.5, 25.5)
h_DCPT  = ROOT.TH1F("h_DCPT_"+suffix, "#Deltap_{T}(mkFit,CMSSW)", 63, -2.5, 2.5)
h_DCETA = ROOT.TH1F("h_DCETA_"+suffix, "#Delta#eta(mkFit,CMSSW)", 45, -0.5, 0.5)
h_DCPHI = ROOT.TH1F("h_DCPHI_"+suffix, "#Delta#phi(mkFit,CMSSW)", 45, -0.5, 0.5)

h_DCNH.Sumw2()
h_DCPT.Sumw2()
h_DCETA.Sumw2()
h_DCPHI.Sumw2()

with open('log_'+arch+'_'+sample+'_'+build+'_'+suffix+'_DumpForPlots.txt') as f :
    for line in f :
        if "MX - found track with chi2" in line :
            lsplit = line.split()

            NH = float(lsplit[8])
            h_MXNH.Fill(NH)

            PT = float(lsplit[10])
            h_MXPT.Fill(PT)

            ETA = float(lsplit[12])
            h_MXETA.Fill(ETA)

            PHI = float(lsplit[14])
            h_MXPHI.Fill(PHI)

            NHC = float(lsplit[24])
            if NHC > 0 :
                h_DCNH.Fill(NH-NHC)

                PTC = float(lsplit[26])
                h_DCPT.Fill(PT-PTC)
                
                ETAC = float(lsplit[28])
                h_DCETA.Fill(ETA-ETAC)
                
                PHIC = float(lsplit[30])
                h_DCPHI.Fill(PHI-PHIC)

g.Write()
g.Close()
