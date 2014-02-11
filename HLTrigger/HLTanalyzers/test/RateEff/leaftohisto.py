import math,ROOT
from ROOT import gStyle, gROOT, TFile, TChain, TTree, TH1F, TH1D, TF1, TCanvas, TLatex, SetOwnership

outfile="NPVtx_new.root"
outf = TFile(outfile,"RECREATE");
outf.cd()

SetOwnership( outf, False )   # tell python not to take ownership
print "Output written to: ", outfile


fname = '/eos/uscms/store/user/ingabu/TMD/FixedNtuples/QCD_Pt-30to50_MuEnrichedPt5_TuneZ2star_13TeV_pythia6_25ns/hltbitsmc_34_1_qFL.root'

f1 = TFile.Open(fname)

tree = f1.Get("HltTree")

NPV = TH1D("NPV","NPV",25, 0., 50.)

tree.Draw("NPUgenBX0>>NPV", '' ,'goff')



#raw_input('...')


outf.cd()
NPV.Write()
outf.Write()
outf.Close()
#f1.Close()

