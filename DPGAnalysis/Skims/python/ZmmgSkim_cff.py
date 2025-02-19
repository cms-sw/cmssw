'''
Defines the selection sequence ZmmgSkimSeq for the Zmmg skim for the 
RAW-RECO event content. It also defines several other modules and sequences
used: 
    ZmmgHLTFilter
    ZmmgTrailingMuons
    ZmmgLeadingMuons
    ZmmgDimuons
    ZmmgDimuonFilter
    ZmmgDimuonSequence
    ZmmgMergedSuperClusters
    ZmmgPhotonCandidates
    ZmmgPhotons
    ZmmgPhotonSequence
    ZmmgCandidates
    ZmmgFilter
    ZmmgSequence

Jan Veverka, Caltech, 5 May 2012
'''

import copy
import FWCore.ParameterSet.Config as cms


###____________________________________________________________________________
###
###  HLT Filter
###____________________________________________________________________________

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ZmmgHLTFilter = copy.deepcopy(hltHighLevel)
ZmmgHLTFilter.throw = cms.bool(False)
ZmmgHLTFilter.HLTPaths = ['HLT_Mu*','HLT_IsoMu*','HLT_DoubleMu*']


###____________________________________________________________________________
###
###  Build the Dimuon Sequence
###____________________________________________________________________________

### Get muons of needed quality for Z -> mumugamma 
ZmmgTrailingMuons = cms.EDFilter('MuonSelector',
    src = cms.InputTag('muons'),
    cut = cms.string('''pt > 10 && 
                        abs(eta) < 2.4 && 
                        isGlobalMuon = 1 && 
                        isTrackerMuon = 1 && 
                        abs(innerTrack().dxy)<2.0'''),
    filter = cms.bool(True)                                
    )

### Require a harder pt cut on the leading leg
ZmmgLeadingMuons = cms.EDFilter('MuonSelector',
    src = cms.InputTag('ZmmgTrailingMuons'),
    cut = cms.string('pt > 20'),
    filter = cms.bool(True)                                
    )

### Build dimuon candidates
ZmmgDimuons = cms.EDProducer('CandViewShallowCloneCombiner',
    decay = cms.string('ZmmgLeadingMuons@+ ZmmgTrailingMuons@-'),
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 30'),
    )

### Require at least one dimuon candidate
ZmmgDimuonFilter = cms.EDFilter('CandViewCountFilter',
    src = cms.InputTag('ZmmgDimuons'),
    minNumber = cms.uint32(1)
    )

### Put together the dimuon sequence
ZmmgDimuonSequence = cms.Sequence(
    ZmmgTrailingMuons *
    ZmmgLeadingMuons *
    ZmmgDimuons *
    ZmmgDimuonFilter
    )
    
    
###____________________________________________________________________________
###
###  Build the Supercluster/Photon Sequence
###____________________________________________________________________________

### Merge the barrel and endcap superclusters
ZmmgMergedSuperClusters =  cms.EDProducer('EgammaSuperClusterMerger',
    src = cms.VInputTag(
        cms.InputTag('correctedHybridSuperClusters'),
        cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')
        )
    )

### Build candidates from all the merged superclusters
ZmmgPhotonCandidates = cms.EDProducer('ConcreteEcalCandidateProducer',
    src = cms.InputTag('ZmmgMergedSuperClusters'),
    particleType = cms.string('gamma')
    )

### Select photon candidates with Et > 5 GeV
ZmmgPhotons = cms.EDFilter('CandViewSelector',
    src = cms.InputTag('ZmmgPhotonCandidates'),
    cut = cms.string('et > 5'),
    filter = cms.bool(True)
    )

### Put together the photon sequence
ZmmgPhotonSequence = cms.Sequence(
    ZmmgMergedSuperClusters *
    ZmmgPhotonCandidates *
    ZmmgPhotons
    )
    
    
###____________________________________________________________________________
###
###  Build the mu-mu-gamma filter sequence
###____________________________________________________________________________

### Combine dimuons and photons to mumugamma candidates requiring
###     1. trailing muon pt + photon et > 20 GeV
###     2. distance between photon and near muon deltaR < 1.5
###     3. sum of invariant masses of the mmg and mm systems < 200 GeV 
###     4. invariant mass of the mmg system > 40 GeV
### dimuon        = daughter(0)
### leading muon  = daughter(0).daughter(0)
### trailing muon = daughter(0).daughter(1)
### photon        = daughter(1)
ZmmgCandidates = cms.EDProducer('CandViewShallowCloneCombiner',
    decay = cms.string('ZmmgDimuons ZmmgPhotons'),
    checkCharge = cms.bool(False),
    cut = cms.string('''
        daughter(0).daughter(1).pt + daughter(1).pt > 20 &
        min(deltaR(daughter(0).daughter(0).eta,
                   daughter(0).daughter(0).phi,
                   daughter(1).eta,
                   daughter(1).phi),
            deltaR(daughter(0).daughter(1).eta,
                   daughter(0).daughter(1).phi,
                   daughter(1).eta,
                   daughter(1).phi)) < 1.5 &
        mass + daughter(0).mass < 200 &
        mass > 40
        '''),
    )
    
### Require at least one mu-mu-gamma candidate passing the cuts
ZmmgFilter = cms.EDFilter('CandViewCountFilter',
    src = cms.InputTag('ZmmgCandidates'),
    minNumber = cms.uint32(1)
    )
    
ZmmgSequence = cms.Sequence(
    ZmmgCandidates *
    ZmmgFilter
    )


###____________________________________________________________________________
###
###  Build the full selection sequence for the ZMuMuGammaSkim
###____________________________________________________________________________

ZmmgSkimSeq = cms.Sequence(
    ZmmgHLTFilter *
    ZmmgDimuonSequence *
    ZmmgPhotonSequence *
    ZmmgSequence
    )

