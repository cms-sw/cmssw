#! /usr/bin/env python3

from __future__ import print_function
import ROOT
import sys
from DataFormats.FWLite import Events, Handle
from math import *

def deltaPhi(a,b) :
 r = a-b
 while r>2*pi : r-=2*pi 
 while r<-2*pi : r+=2*pi 
 return r

def deltaR(a,b) :
  dphi=deltaPhi(a.phi(),b.phi())
  return sqrt(dphi*dphi+(a.eta()-b.eta())**2)  

events = Events (['test.root'])

handleGJ1  = Handle ("std::vector<pat::Jet>")
handleGJ2  = Handle ("std::vector<pat::Jet>")
handleGP  = Handle ("std::vector<reco::GenParticle>")

# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
labelGJ1 = ("slimmedJets","","PAT")
labelGJ2 = ("patJetsAK5PFCHS","","S2")
labelGP = ("prunedGenParticles")

ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.gROOT.SetStyle('Plain') # white background
zptHist = ROOT.TH1F ("zpt", "Z Pt", 50, 0, 500)

# loop over events
count= 0
for event in events:
    count+=1 
    if count % 1000 == 0 :
	print(count)
    event.getByLabel (labelGJ1, handleGJ1)
    event.getByLabel (labelGJ2, handleGJ2)
    event.getByLabel (labelGP, handleGP)
    # get the product
    jets1 = handleGJ1.product()
    jets2 = handleGJ2.product()
    
    for j1,j2 in zip(jets1,jets2)  :
            if abs(j1.eta()) < 2.5 and j1.pt() > 20 and j1.chargedHadronEnergyFraction() > 0.05 :
		if abs(j1.pt()-j2.pt())/(j1.pt()+j2.pt()) >0.05 :
			print("Mismatch at record ", count)
			print("Bad match is : pt %s vs %s, b-tag %s vs %s, MC flavour %s vs %s " %(j1.pt(),j2.pt(),j1.bDiscriminator("combinedSecondaryVertexBJetTags"),j2.bDiscriminator("combinedSecondaryVertexBJetTags"),j1.partonFlavour(),j2.partonFlavour()))
			print("Jet eta and phi" ,j1.eta(),j1.phi())
			print(" alljets ")
   		        for j11,j22 in zip(jets1,jets2)  :
	                        print("  Jet pt %s vs %s, b-tag %s vs %s, MC flavour %s vs %s " % (j11.pt(),j22.pt(),j11.bDiscriminator("combinedSecondaryVertexBJetTags"),j22.bDiscriminator("combinedSecondaryVertexBJetTags"),j11.partonFlavour(),j22.partonFlavour()))
	#		print "gen parts"
	#		genparts = handleGP.product()
	#		for gp in genparts :
	#			if abs(gp.eta()-j1.eta()) < 0.3 :
	#				print deltaR(j1.p4(),gp.p4()),gp.pdgId(),gp.status(),gp.pt(),gp.eta(),gp.phi()
    

