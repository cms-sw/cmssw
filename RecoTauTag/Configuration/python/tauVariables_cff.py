import FWCore.ParameterSet.Config as cms

##
## Content for dumpTauVariables that can be run on pat::Tau content in the standard an miniAOD
## configuration  
##
slimmedVariables = cms.VPSet(
    cms.PSet(
        tag = cms.untracked.string("mass"),
        quantity = cms.untracked.string("mass")
        ),
    cms.PSet(
        tag = cms.untracked.string("pt"),
        quantity = cms.untracked.string("pt")
        ),
    cms.PSet(
        tag = cms.untracked.string("eta"),
        quantity = cms.untracked.string("eta")
        ),
    cms.PSet(
        tag = cms.untracked.string("phi"),
        quantity = cms.untracked.string("phi")
        ),
    cms.PSet(
        tag = cms.untracked.string("decayMode"),
        quantity = cms.untracked.string("decayMode")
        ),
    cms.PSet(
        tag = cms.untracked.string("decayModeFinding"),
        quantity = cms.untracked.string("tauID('decayModeFinding')")
        ),
    cms.PSet(
        tag = cms.untracked.string("decayModeFindingNewDMs"),
        quantity = cms.untracked.string("tauID('decayModeFindingNewDMs')")
        ),
    cms.PSet(
        tag = cms.untracked.string("chargedIsoPtSum"),
        quantity = cms.untracked.string("tauID('chargedIsoPtSum')")
        ),
    cms.PSet(
        tag = cms.untracked.string("neutralIsoPtSum"),
        quantity = cms.untracked.string("tauID('neutralIsoPtSum')")
        ),
    cms.PSet(
        tag = cms.untracked.string("puCorrPtSum"),
        quantity = cms.untracked.string("tauID('puCorrPtSum')")
        ),
    cms.PSet(
        tag = cms.untracked.string("neutralIsoPtSumWeight"),
        quantity = cms.untracked.string("tauID('neutralIsoPtSumWeight')")
        ),
    cms.PSet(
        tag = cms.untracked.string("footprintCorrection"),
        quantity = cms.untracked.string("tauID('footprintCorrection')")
        ),
    cms.PSet(
        tag = cms.untracked.string("photonPtSumOutsideSignalCone"),
        quantity = cms.untracked.string("tauID('photonPtSumOutsideSignalCone')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstMuonLoose3"),
        quantity = cms.untracked.string("tauID('againstMuonLoose3')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstMuonTight3"),
        quantity = cms.untracked.string("tauID('againstMuonTight3')")
        ),                                        
    cms.PSet(
        tag = cms.untracked.string("byCombinedIsolationDeltaBetaCorrRaw3Hits"),
        quantity = cms.untracked.string("tauID('byCombinedIsolationDeltaBetaCorrRaw3Hits')")
        ),
    cms.PSet(
        tag = cms.untracked.string("byIsolationMVArun2v1DBoldDMwLTraw"),
        quantity = cms.untracked.string("tauID('byIsolationMVArun2v1DBoldDMwLTraw')")
        ),                                
    cms.PSet(
        tag = cms.untracked.string("byIsolationMVArun2v1DBnewDMwLTraw"),
        quantity = cms.untracked.string("tauID('byIsolationMVArun2v1DBnewDMwLTraw')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstElectronMVA6Raw"),
        quantity = cms.untracked.string("tauID('againstElectronMVA6Raw')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstElectronMVA6category"),
        quantity = cms.untracked.string("tauID('againstElectronMVA6category')")
        ),
    )  

##
## Content for dumpTauVariables that can be run on pat::Tau content if maximal event content is
## embedded to the tau candidates (not yet supported)
##
fatVariables = cms.VPSet(
    cms.PSet(
        tag = cms.untracked.string("mass"),
        quantity = cms.untracked.string("mass")
        ),
    cms.PSet(
        tag = cms.untracked.string("pt"),
        quantity = cms.untracked.string("pt")
        ),
    cms.PSet(
        tag = cms.untracked.string("eta"),
        quantity = cms.untracked.string("eta")
        ),
    cms.PSet(
        tag = cms.untracked.string("phi"),
        quantity = cms.untracked.string("phi")
        ),
    cms.PSet(
        tag = cms.untracked.string("decayMode"),
        quantity = cms.untracked.string("decayMode")
        ),
    cms.PSet(
        tag = cms.untracked.string("ecalStripSumEOverPLead"),
        quantity = cms.untracked.string("ecalStripSumEOverPLead")
        ),
    cms.PSet(
        tag = cms.untracked.string("electronPreIDOutput"),
        quantity = cms.untracked.string("electronPreIDOutput")
        ),
    cms.PSet(
        tag = cms.untracked.string("emFraction"),
        quantity = cms.untracked.string("emFraction")
        ),
    cms.PSet(
        tag = cms.untracked.string("etaetaMoment"),
        quantity = cms.untracked.string("etaetaMoment")
        ),
    cms.PSet(
        tag = cms.untracked.string("etaphiMoment"),
        quantity = cms.untracked.string("etaphiMoment")
        ),
    cms.PSet(
        tag = cms.untracked.string("hcal3x3OverPLead"),
        quantity = cms.untracked.string("hcal3x3OverPLead")
        ),
    cms.PSet(
        tag = cms.untracked.string("hcalMaxOverPLead"),
        quantity = cms.untracked.string("hcalMaxOverPLead")
        ),
    cms.PSet(
        tag = cms.untracked.string("hcalTotOverPLead"),
        quantity = cms.untracked.string("hcalTotOverPLead")
        ),
    cms.PSet(
        tag = cms.untracked.string("isolationECALhitsEtSum"),
        quantity = cms.untracked.string("isolationECALhitsEtSum")
        ),
    cms.PSet(
        tag = cms.untracked.string("isolationPFChargedHadrCandsPtSum"),
        quantity = cms.untracked.string("isolationPFChargedHadrCandsPtSum")
        ),
    cms.PSet(
        tag = cms.untracked.string("isolationPFGammaCandsEtSum"),
        quantity = cms.untracked.string("isolationPFGammaCandsEtSum")
        ),
    cms.PSet(
        tag = cms.untracked.string("isolationTracksPtSum"),
        quantity = cms.untracked.string("isolationTracksPtSum")
        ),
    cms.PSet(
        tag = cms.untracked.string("leadPFChargedHadrCandsignedSipt"),
        quantity = cms.untracked.string("leadPFChargedHadrCandsignedSipt")
        ),
    cms.PSet(
        tag = cms.untracked.string("leadTrackHCAL3x3hitsEtSum"),
        quantity = cms.untracked.string("leadTrackHCAL3x3hitsEtSum")
        ),
    cms.PSet(
        tag = cms.untracked.string("leadTrackHCAL3x3hottesthitDEta"),
        quantity = cms.untracked.string("leadTrackHCAL3x3hottesthitDEta")
        ),
    cms.PSet(
        tag = cms.untracked.string("leadTracksignedSipt"),
        quantity = cms.untracked.string("leadTracksignedSipt")
        ),
    cms.PSet(
        tag = cms.untracked.string("maximumHCALhitEt"),
        quantity = cms.untracked.string("maximumHCALhitEt")
        ),
    cms.PSet(
        tag = cms.untracked.string("maximumHCALPFClusterEt"),
        quantity = cms.untracked.string("maximumHCALPFClusterEt")
        ),
    cms.PSet(
        tag = cms.untracked.string("muonDecision"),
        quantity = cms.untracked.string("muonDecision")
        ),
    cms.PSet(
        tag = cms.untracked.string("phiphiMoment"),
        quantity = cms.untracked.string("phiphiMoment")
        ),
    cms.PSet(
        tag = cms.untracked.string("segComp"),
        quantity = cms.untracked.string("segComp")
        ),
    cms.PSet(
        tag = cms.untracked.string("signalTracksInvariantMass"),
        quantity = cms.untracked.string("signalTracksInvariantMass")
        ),                                
    cms.PSet(
        tag = cms.untracked.string("TracksInvariantMass"),
        quantity = cms.untracked.string("TracksInvariantMass")
        ),
    cms.PSet(
        tag = cms.untracked.string("decayModeFinding"),
        quantity = cms.untracked.string("tauID('decayModeFinding')")
        ),
    cms.PSet(
        tag = cms.untracked.string("decayModeFindingNewDMs"),
        quantity = cms.untracked.string("tauID('decayModeFindingNewDMs')")
        ),
    cms.PSet(
        tag = cms.untracked.string("chargedIsoPtSum"),
        quantity = cms.untracked.string("tauID('chargedIsoPtSum')")
        ),
    cms.PSet(
        tag = cms.untracked.string("neutralIsoPtSum"),
        quantity = cms.untracked.string("tauID('neutralIsoPtSum')")
        ),
    cms.PSet(
        tag = cms.untracked.string("puCorrPtSum"),
        quantity = cms.untracked.string("tauID('puCorrPtSum')")
        ),
    cms.PSet(
        tag = cms.untracked.string("neutralIsoPtSumWeight"),
        quantity = cms.untracked.string("tauID('neutralIsoPtSumWeight')")
        ),
    cms.PSet(
        tag = cms.untracked.string("footprintCorrection"),
        quantity = cms.untracked.string("tauID('footprintCorrection')")
        ),
    cms.PSet(
        tag = cms.untracked.string("photonPtSumOutsideSignalCone"),
        quantity = cms.untracked.string("tauID('photonPtSumOutsideSignalCone')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstMuonLoose3"),
        quantity = cms.untracked.string("tauID('againstMuonLoose3')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstMuonTight3"),
        quantity = cms.untracked.string("tauID('againstMuonTight3')")
        ),                                        
    cms.PSet(
        tag = cms.untracked.string("byCombinedIsolationDeltaBetaCorrRaw3Hits"),
        quantity = cms.untracked.string("tauID('byCombinedIsolationDeltaBetaCorrRaw3Hits')")
        ),
    cms.PSet(
        tag = cms.untracked.string("byIsolationMVArun2v1DBoldDMwLTraw"),
        quantity = cms.untracked.string("tauID('byIsolationMVArun2v1DBoldDMwLTraw')")
        ),                                
    cms.PSet(
        tag = cms.untracked.string("byIsolationMVArun2v1DBnewDMwLTraw"),
        quantity = cms.untracked.string("tauID('byIsolationMVArun2v1DBnewDMwLTraw')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstElectronMVA6Raw"),
        quantity = cms.untracked.string("tauID('againstElectronMVA6Raw')")
        ),
    cms.PSet(
        tag = cms.untracked.string("againstElectronMVA6category"),
        quantity = cms.untracked.string("tauID('againstElectronMVA6category')")
        ),
    )  
