from __future__ import print_function
# import ROOT in batch mode
import sys
oldargv = sys.argv[:]
#sys.argv = [ '-b-' ]
import ROOT
#ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
events = Events("file:test.root")
clusters, clusterLabel = Handle("l1t::HGCalClusterBxCollection"), "hgcalTriggerPrimitiveDigiProducer:HGCalTriggerSimClusterBestChoice"
genParticles, genParticlesLabel = Handle("vector<reco::GenParticle>"),"genParticles"

verbose=False

hEta=ROOT.TH1D("eta","eta",1000,-20,20)
hPhi=ROOT.TH1D("phi","phi",1000,-3.1416,3.1416)
hPt = ROOT.TH1D("pt","pt",1000,0,1000)
hE = ROOT.TH1D("E","E",1000,0,1000)
## cluster shapes
hN = ROOT.TH1D("N","N",50,0,1000)
hSEE = ROOT.TH1D("SEE","SEE",1000,0,0.02)
hSPP = ROOT.TH1D("SPP","SPP",1000,0,0.02)
hSRR = ROOT.TH1D("SRR","SRR",1000,0,100.)
##
hLE = ROOT.TH1D("LogEoverE","Log(e)/E",1000,-0.5,1.)
hED = ROOT.TH1D("ED","ED",1000,0.,1.)
hSEP = ROOT.TH1D("SEP","SEP",1000,-0.02,0.02)

hMatch={
	"deta":ROOT.TH1D("eta_m","eta",1000,-20,20),
	"dphi":ROOT.TH1D("phi_m","phi",1000,-3.1416,3.1416),
	"pt" : ROOT.TH1D("pt_m","pt",1000,0,1000),
	"ptsum" : ROOT.TH1D("pt_sum_m","pt",1000,0,1000),
	}

import math
def deltaPhi(phi1,phi2):
	pi=3.14159265359
	dphi=phi1-phi2
	while dphi > pi: dphi -= 2*pi
	while dphi < -pi: dphi += 2*pi
	return dphi

def deltaR(eta1,phi1,eta2,phi2):
	return math.sqrt( (eta1-eta2)**2 + deltaPhi(phi1,phi2)**2)

for iev,event in enumerate(events):
	if verbose: 
	   print("\nEvent %d: run %6d, lumi %4d, event %12d" % (iev,event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event()))

	event.getByLabel(clusterLabel, clusters)
	event.getByLabel(genParticlesLabel,genParticles)

	## check that handle is valid
	if not clusters.isValid(): print("Cluster handle is invalid")
	if not genParticles.isValid(): print("GP handle not valid")

	## --
	if verbose: print("-> BX=0")
	if verbose: print("-> clusters=",clusters,"\n","product=",clusters.product())

	print("--------------------------------") 
	print("GP SIZE=",genParticles.product().size())
	for gp in genParticles.product():
		if gp.status() == 1:
			print("-> Found GP",gp.pt(),gp.eta(),gp.phi(),gp.pdgId())
	print("--------------------------------") 

	## get 0 bunch crossing vector
	ptsum = [ROOT.TLorentzVector(),ROOT.TLorentzVector() ]
	for bx,vector in enumerate(clusters.product()):	
		if verbose: print("-> 0 pos",clusters.product()[0])
		if vector == None: 
			print("   cluster product is none")
			continue
		hEta.Fill(vector.eta() ) 
		if vector.eta() <8:
			hE.Fill(vector.energy() ) 
			hPt.Fill(vector.pt() ) 
			hPhi.Fill(vector.phi() ) 
			## cluster shapes
			hN.Fill(vector.shapes.N())
			hSEE.Fill(vector.shapes.SigmaEtaEta())
			hSPP.Fill(vector.shapes.SigmaPhiPhi())
			hSRR.Fill(vector.shapes.SigmaRR())
			hLE.Fill(vector.shapes.LogEoverE() ) 
			hED.Fill(vector.shapes.eD())
			hSEP.Fill(vector.shapes.SigmaEtaPhi() )
		for idx,gp in enumerate(genParticles.product()):
			if gp.status()==1 and deltaR(vector.eta(),vector.phi(),gp.eta(),gp.phi())<0.1:
				hMatch["deta"].Fill(vector.eta()-gp.eta())
				hMatch["dphi"].Fill(deltaPhi(vector.phi(),gp.phi()))
				hMatch["pt"].Fill(vector.pt())
				ptsum[idx] += ROOT.TLorentzVector(vector.px(),vector.py(),vector.pz(),vector.energy())
	hMatch["ptsum"].Fill( ptsum[0].Pt() ) 
	hMatch["ptsum"].Fill( ptsum[1].Pt() ) 



c=ROOT.TCanvas()
c.Divide(2,2)
c.cd(1)
hPt.Draw("HIST")
c.cd(2)
hEta.Draw("HIST")
c.cd(3)
hPhi.Draw("HIST")
c.cd(4)
hE.Draw("HIST")

c2=ROOT.TCanvas("c2","c2")
c2.Divide(2,2)
c2.cd(1)
hN.Draw("HIST")
c2.cd(2)
hSEE.Draw("HIST")
c2.cd(3)
hSPP.Draw("HIST")
c2.cd(4)
hSRR.Draw("HIST")

c4=ROOT.TCanvas("c4","c4")
c4.Divide(2,2)
c4.cd(1)
hMatch["pt"].Draw("HIST")
c4.cd(2)
hMatch["deta"].Draw("HIST")
c4.cd(3)
hMatch["dphi"].Draw("HIST")
c4.cd(4)
hMatch["ptsum"].Draw("HIST")
raw_input("ok?")

