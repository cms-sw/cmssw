import sys;
import os;
import tarfile;
import shutil;

hfilename = sys.argv[1];
pfilename = sys.argv[2];

from ROOT import *;
gSystem.Load("libFWCoreFWLite.so");
AutoLibraryLoader.enable();

gROOT.Reset();

def plot(file, hist, xLabel, yLabel="Events", Opt=""):
	H = file.Get(hist);
	H.GetXaxis().SetTitle(xLabel);
	H.GetYaxis().SetTitle(yLabel);
	H.Draw(Opt);

# Sum two/three L1 histos and plot
# Use for eg. overall jet Et distribution
def plotSum(file, dirA, dirB, dirC, hist, xLabel, yLabel="Events", Opt=""):
	histA = file.Get(dirA+hist);
	histB = file.Get(dirB+hist);
	histA.Add(histB);
	if (dirC != ""):
		histC = file.Get(dirC+hist);
		histA.Add(histC);
	histA.GetXaxis().SetTitle(xLabel);
  	histA.GetYaxis().SetTitle(yLabel);
	histA.Draw(Opt);

# Sum two/three L1 histos, divide by ref, and plot
# Use for eg. total EM efficiency, rather than iso and non-iso separately
def plotCombEff(file, dirA, dirB, dirC, sumRef, hist, xLabel, yLabel="Events", Opt=""):
	histA = file.Get(dirA+"/L1Candidates/"+hist);
	histB = file.Get(dirB+"/L1Candidates/"+hist);
       	histA.Add(histB);
	histRefA = file.Get(dirA+"/RefCandidates/"+hist);
	if (sumRef):
		histRefB = file.Get(dirB+"/RefCandidates/"+hist);
		histRefA.Add(histRefB);
       	if (dirC != ""):
		histC = file.Get(dirC+"/L1Candidates/"+hist);
		histA.Add(histC);
		if (sumRef):
			histRefC = file.Get(dirC+"/RefCandidates/"+hist);
			histRefA.Add(histC);
	if (histRefA.GetEntries() > 0):
		histA.Divide(histRefA);
	histA.GetXaxis().SetTitle(xLabel);
  	histA.GetYaxis().SetTitle(yLabel);
	histA.Draw(Opt);

# Basic plots
def basicPlots(obj, dir, plotdir):
	plot(hfile,dir+"/L1Candidates/Et","E_{T}","No. of entries");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"Et.png");

       	plot(hfile,dir+"/L1Candidates/Eta","#eta","No. of entries");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"Eta.png");

	plot(hfile,dir+"/L1Candidates/Phi","#phi","No. of entries");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"Phi.png");

	plot(hfile,dir+"/Resolutions/EtRes","(E_{T,L1}-E_{T,Ref})/E_{T,Ref}","No. of entries");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"EtRes.png");

	plot(hfile,dir+"/Resolutions/EtCor","E_{T, Ref} (GeV)","E_{T, L1} (GeV)");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"EtCor.png");

	plot(hfile,dir+"/Resolutions/EtaRes","(#eta_{L1}-#eta_{Ref})/#eta_{Ref}","No. of entries");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"EtaRes.png");

	plot(hfile,dir+"/Resolutions/EtaCor","#eta_{Ref}","#eta_{L1}");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"EtaCor.png");

	plot(hfile,dir+"/Resolutions/PhiRes","(#phi_{L1}-#phi_{Ref})/#phi_{Ref}","No. of entries");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"PhiRes.png");

	plot(hfile,dir+"/Resolutions/PhiCor","#phi_{Ref} (rad)","#phi_{L1} (rad)");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"PhiCor.png");

	plot(hfile,dir+"/Efficiencies/EtEff","E_{T} (GeV)","Efficiency","e");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"EtEff.png");

	plot(hfile,dir+"/Efficiencies/EtaEff","#eta","Efficiency","e");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"EtaEff.png");

	plot(hfile,dir+"/Efficiencies/PhiEff","#phi (rad)","Efficiency","e");
	canvas.Update();
	canvas.Print(plotdir+"/"+obj+"PhiEff.png");


# make output directory structure
plotdirs = ["muon", "isoem", "nonisoem", "relem", "tau", "cenjet", "forjet", "jet", "met"];
for dir in plotdirs:
	try:
		os.mkdir(dir);
	except OSError:
		pass


# open file
hfile = TFile(hfilename);
print "Making plots for "+hfilename;

# canvas
canvas = TCanvas("canvas");


# muon resolutions & efficiencies
#basicPlots("muon", "L1AnalyzerMuonMC", "muon");

# iso EM
basicPlots("isoem", "L1AnalyzerIsoEmMC", "isoem");

# iso EM
basicPlots("nonisoem", "L1AnalyzerIsoEmMC", "nonisoem");

# tau jets
basicPlots("tau", "L1AnalyzerTauJetsMC", "tau");

# cen jets
basicPlots("cenjet", "L1AnalyzerForJetsMC", "cenjet");

# for jets
basicPlots("forjet", "L1AnalyzerForJetsMC", "forjet");

# missing Et
basicPlots("met", "L1AnalyzerMetMC", "met");

# combined iso and non-iso EM efficiencies
plotCombEff(hfile,"L1AnalyzerIsoEmMC","L1AnalyzerNonIsoEmMC","", false, "Et", "#et","Efficiency","e"); 
canvas.Update();
canvas.Print("relem/relemEtEff.png");

plotCombEff(hfile,"L1AnalyzerIsoEmMC","L1AnalyzerNonIsoEmMC","", false, "Eta", "#eta","Efficiency","e"); 
canvas.Update();
canvas.Print("relem/relemEtaEff.png");

plotCombEff(hfile,"L1AnalyzerIsoEmMC","L1AnalyzerNonIsoEmMC","", false, "Phi", "#phi","Efficiency","e"); 
canvas.Update();
canvas.Print("relem/relemPhiEff.png");

# total jet Et
plotSum(hfile, "L1AnalyzerCenJetsMC", "L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC","/L1Candidates/Et", "E_{t}","No. of entries","e");
canvas.Update();
canvas.Print("jet/jetEt.png");

# total jet Et resolution
plotSum(hfile, "L1AnalyzerCenJetsMC", "L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC","/Resolutions/EtRes", "(E_{T,L1}-E_{T,Ref})/E_{T,Ref}","No. of entries","e");
canvas.Update();
canvas.Print("jet/jetEtRes.png");

# total jet Eta resolution
plotSum(hfile, "L1AnalyzerCenJetsMC", "L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC","/Resolutions/EtaRes", "(#eta_{L1}-#eta_{Ref})/#eta_{Ref}","No. of entries","e");
canvas.Update();
canvas.Print("jet/jetEtaRes.png");

# total jet Phi resolution
plotSum(hfile, "L1AnalyzerCenJetsMC", "L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC","/Resolutions/PhiRes", "(#phi_{L1}-#phi_{Ref})/#phi_{Ref}","No. of entries","e");
canvas.Update();
canvas.Print("jet/jetPhiRes.png");

# total jet efficiencies
plotCombEff(hfile,"L1AnalyzerCenJetsMC","L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC", true, "Et", "#et","Efficiency","e"); 
canvas.Update();
canvas.Print("jet/jetEtEff.png");

plotCombEff(hfile,"L1AnalyzerCenJetsMC","L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC", true, "Eta", "#eta","Efficiency","e"); 
canvas.Update();
canvas.Print("jet/jetEtaEff.png");

plotCombEff(hfile,"L1AnalyzerCenJetsMC","L1AnalyzerTauJetsQCDMC","L1AnalyzerForJetsMC", true, "Phi", "#phi","Efficiency","e"); 
canvas.Update();
canvas.Print("jet/jetPhiEff.png");


# make tarfile
t = tarfile.open(name = pfilename+".tgz", mode = 'w:gz')
for dir in plotdirs:
	t.add(dir);
t.close()

# cleanup
for dir in plotdirs:
	shutil.rmtree(dir);
