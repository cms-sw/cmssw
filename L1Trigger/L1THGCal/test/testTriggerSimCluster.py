# import ROOT in batch mode
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
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

for i,event in enumerate(events):
	if verbose: 
	   print "\nEvent", i
	   if i > 10: break

	event.getByLabel(clusterLabel, clusters)

	## check that handle is valid
	if not clusters.isValid(): print "Cluster handle is invalid"

	## --
	if verbose: print "-> BX=0"
	if verbose: print "-> clusters=",clusters,"\n","product=",clusters.product()

	print "what's saved in clusters?"
	for bx in clusters.product():
		print "bx=",bx
		print "product", clusters.product()[bx]

	## get 0 bunch crossing vector
	for bx in [0]:	

		if verbose: print "-> 0 pos",clusters.product()[0]

		vector = clusters.product()[bx]
		for idx in range(0,vector.size()):
			print "* accessing position ", idx
			print "   ",vector[idx].pt(),vector[idx].eta()
	
