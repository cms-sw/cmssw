import FWCore.ParameterSet.Config as cms
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from JetMETCorrections.TauJet.TCTauAlgoParameters_cfi import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

tcRecoTauDiscriminationAgainstHadronicJets = cms.EDProducer("TCRecoTauDiscriminationAgainstHadronicJets",
	tcTauAlgoParameters,
	CaloTauProducer = cms.InputTag('tcRecoTauProducer'),
	Prediscriminants = noPrediscriminants
)
