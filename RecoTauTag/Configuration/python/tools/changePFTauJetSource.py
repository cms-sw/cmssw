'''

Change the jets with which the PFTau collections are built.

Author: Evan K. Friis, UC Davis

'''

def changePFTauJetSource(process, jetSrc):
    # Update all the tau producers
    for producer in ['combinatoricRecoTaus', 'shrinkingConePFTauProducer']:
        # Update the pizero producer associated to this tau
        tauProducer = getattr(process, producer)
        tauProducer.jetSrc = jetSrc
        piZeroProdName = getattr(tauProducer, 'piZeroSrc').value()
        piZeroProducer = getattr(process, piZeroProdName)
        piZeroProducer.src = jetSrc
    # Set the PFTauTagInfoProducer jet tracks associator correctly
    process.ak5PFJetTracksAssociatorAtVertex.jets = jetSrc

#if __name__ == "__main__":
    #import FWCore.ParameterSet.Config as cms
    #process = cms.Process("TEST")
    #process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    #changePFTauJetSource(process, "myJets")
    #print process.combinatoricRecoTaus.jetSrc
    #print process.ak5PFJetsLegacyTaNCPiZeros.src
