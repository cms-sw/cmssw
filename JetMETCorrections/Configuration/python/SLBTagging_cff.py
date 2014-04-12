import FWCore.ParameterSet.Config as cms

from RecoBTag.SoftLepton.softPFElectronProducer_cfi import *
from RecoBTag.SoftLepton.softMuonTagInfos_cfi import *
from RecoBTag.SoftLepton.softElectronTagInfos_cfi import *


#
# SOFT ELECTRON TAGGING
#
ak5GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'ak5GenJets')
ak5CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'ak5CaloJets')
ak5PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'ak5PFJets')

ak7GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'ak7GenJets')
ak7CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'ak7CaloJets')
ak7PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'ak7PFJets')

kt4GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'kt4GenJets')
kt4CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'kt4CaloJets')
kt4PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'kt4PFJets')

kt6GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'kt6GenJets')
kt6CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'kt6CaloJets')
kt6PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'kt6PFJets')

sc5GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'sc5GenJets')
sc5CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'sc5CaloJets')
sc5PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'sc5PFJets')

sc7GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'sc7GenJets')
sc7CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'sc7CaloJets')
sc7PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'sc7PFJets')

ic5GenJetsSoftElectronTagInfos  = softElectronTagInfos.clone(jets = 'ic5GenJets')
ic5CaloJetsSoftElectronTagInfos = softElectronTagInfos.clone(jets = 'ic5CaloJets')
ic5PFJetsSoftElectronTagInfos   = softElectronTagInfos.clone(jets = 'ic5PFJets')


#
# SOFT MUON TAGGING
#

softMuonTagInfosGMPT = softMuonTagInfos.clone(
    muonSelection = RecoBTag.SoftLepton.muonSelection.GlobalMuonPromptTight
    )

ak5GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'ak5GenJets')
ak5CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'ak5CaloJets')
ak5PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'ak5PFJets')

ak7GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'ak7GenJets')
ak7CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'ak7CaloJets')
ak7PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'ak7PFJets')

kt4GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'kt4GenJets')
kt4CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'kt4CaloJets')
kt4PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'kt4PFJets')

kt6GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'kt6GenJets')
kt6CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'kt6CaloJets')
kt6PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'kt6PFJets')

sc5GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'sc5GenJets')
sc5CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'sc5CaloJets')
sc5PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'sc5PFJets')

sc7GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'sc7GenJets')
sc7CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'sc7CaloJets')
sc7PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'sc7PFJets')

ic5GenJetsSoftMuonTagInfos  = softMuonTagInfosGMPT.clone(jets = 'ic5GenJets')
ic5CaloJetsSoftMuonTagInfos = softMuonTagInfosGMPT.clone(jets = 'ic5CaloJets')
ic5PFJetsSoftMuonTagInfos   = softMuonTagInfosGMPT.clone(jets = 'ic5PFJets')


#
# SOFT LEPTON TAGGING SEQUENCES PER ALGORITHM
#

# ak5
ak5GenJetsSLBSequence = cms.Sequence(
    ak5GenJetsSoftMuonTagInfos*softPFElectrons*ak5GenJetsSoftElectronTagInfos
    )
ak5CaloJetsSLBSequence = cms.Sequence(
    ak5CaloJetsSoftMuonTagInfos*softPFElectrons*ak5CaloJetsSoftElectronTagInfos
    )
ak5PFJetsSLBSequence = cms.Sequence(
    ak5PFJetsSoftMuonTagInfos*softPFElectrons*ak5PFJetsSoftElectronTagInfos
    )

# ak7
ak7GenJetsSLBSequence = cms.Sequence(
    ak7GenJetsSoftMuonTagInfos*softPFElectrons*ak7GenJetsSoftElectronTagInfos
    )
ak7CaloJetsSLBSequence = cms.Sequence(
    ak7CaloJetsSoftMuonTagInfos*softPFElectrons*ak7CaloJetsSoftElectronTagInfos
    )
ak7PFJetsSLBSequence = cms.Sequence(
    ak7PFJetsSoftMuonTagInfos*softPFElectrons*ak7PFJetsSoftElectronTagInfos
    )

# kt4
kt4GenJetsSLBSequence = cms.Sequence(
    kt4GenJetsSoftMuonTagInfos*softPFElectrons*kt4GenJetsSoftElectronTagInfos
    )
kt4CaloJetsSLBSequence = cms.Sequence(
    kt4CaloJetsSoftMuonTagInfos*softPFElectrons*kt4CaloJetsSoftElectronTagInfos
    )
kt4PFJetsSLBSequence = cms.Sequence(
    kt4PFJetsSoftMuonTagInfos*softPFElectrons*kt4PFJetsSoftElectronTagInfos
    )

# kt6
kt6GenJetsSLBSequence = cms.Sequence(
    kt6GenJetsSoftMuonTagInfos*softPFElectrons*kt6GenJetsSoftElectronTagInfos
    )
kt6CaloJetsSLBSequence = cms.Sequence(
    kt6CaloJetsSoftMuonTagInfos*softPFElectrons*kt6CaloJetsSoftElectronTagInfos
    )
kt6PFJetsSLBSequence = cms.Sequence(
    kt6PFJetsSoftMuonTagInfos*softPFElectrons*kt6PFJetsSoftElectronTagInfos
    )

# sc5
sc5GenJetsSLBSequence = cms.Sequence(
    sc5GenJetsSoftMuonTagInfos*softPFElectrons*sc5GenJetsSoftElectronTagInfos
    )
sc5CaloJetsSLBSequence = cms.Sequence(
    sc5CaloJetsSoftMuonTagInfos*softPFElectrons*sc5CaloJetsSoftElectronTagInfos
    )
sc5PFJetsSLBSequence = cms.Sequence(
    sc5PFJetsSoftMuonTagInfos*softPFElectrons*sc5PFJetsSoftElectronTagInfos
    )

# sc7
sc7GenJetsSLBSequence = cms.Sequence(
    sc7GenJetsSoftMuonTagInfos*softPFElectrons*sc7GenJetsSoftElectronTagInfos
    )
sc7CaloJetsSLBSequence = cms.Sequence(
    sc7CaloJetsSoftMuonTagInfos*softPFElectrons*sc7CaloJetsSoftElectronTagInfos
    )
sc7PFJetsSLBSequence = cms.Sequence(
    sc7PFJetsSoftMuonTagInfos*softPFElectrons*sc7PFJetsSoftElectronTagInfos
    )

# ic5
ic5GenJetsSLBSequence = cms.Sequence(
    ic5GenJetsSoftMuonTagInfos*softPFElectrons*ic5GenJetsSoftElectronTagInfos
    )
ic5CaloJetsSLBSequence = cms.Sequence(
    ic5CaloJetsSoftMuonTagInfos*softPFElectrons*ic5CaloJetsSoftElectronTagInfos
    )
ic5PFJetsSLBSequence = cms.Sequence(
    ic5PFJetsSoftMuonTagInfos*softPFElectrons*ic5PFJetsSoftElectronTagInfos
    )
