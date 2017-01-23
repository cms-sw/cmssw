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
events = Events("file:step2.root")
# BXVector<l1t::HGCalCluster>           "hgcalTriggerPrimitiveDigiProducer"   "HGCalTriggerSimClusterBestChoice"   "DIGI" 
#clusters, clusterLabel = Handle("BXVector<l1t::HGCalCluster>"), "hgcalTriggerPrimitiveDigiProducer:HGCalTriggerSimClusterBestChoice"
clusters, clusterLabel = Handle("l1t::HGCalClusterBxCollection"), "hgcalTriggerPrimitiveDigiProducer:HGCalTriggerSimClusterBestChoice"

verbose=True

hEta=ROOT.TH1D("eta","eta",1000,-20,20)
hPhi=ROOT.TH1D("phi","phi",1000,-3.1416,3.1416)
hPt = ROOT.TH1D("pt","pt",1000,0,1000)
hE = ROOT.TH1D("E","E",1000,0,1000)

for iev,event in enumerate(events):
	if verbose: 
	   print "\nEvent %d: run %6d, lumi %4d, event %12d" % (iev,event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event())
	   #if iev > 10: break

	event.getByLabel(clusterLabel, clusters)

	## check that handle is valid
	if not clusters.isValid(): print "Cluster handle is invalid"

	## --
	if verbose: print "-> BX=0"
	if verbose: print "-> clusters=",clusters,"\n","product=",clusters.product()

	print "what's saved in clusters?"
	for bx in clusters.product():
		print "bx=",bx

	## get 0 bunch crossing vector
	for bx,vector in enumerate(clusters.product()):	
		if verbose: print "-> 0 pos",clusters.product()[0]
		if vector == None: 
			print "   cluster product is none"
			continue
		print "   pt=",vector.pt(),"eta=",vector.eta(),"phi=",vector.phi()
		hEta.Fill(vector.eta() ) 
		if vector.eta() <8:
			hE.Fill(vector.energy() ) 
			hPt.Fill(vector.pt() ) 
			hPhi.Fill(vector.phi() ) 


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
raw_input("ok?")

