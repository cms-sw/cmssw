import FWCore.ParameterSet.Config as cms

process = cms.Process("produceAntiMuonDiscrMVATrainingNtuple")

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
#process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string('START53_V15::All')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        ##'/store/user/veelken/CMSSW_5_3_x/skims/SVfitStudies/AccLowPtThresholds/ZplusJets_mutau/genTauLeptonPairSkim_ZplusJets_mutau_1_1_aWJ.root'
        ##'file:/store/user/veelken/CMSSW_5_3_x/skims/simQCDmuEnrichedPt470to600_AOD_1_1_A8V.root'
        'file:/data1/veelken/CMSSW_5_3_x/skims/96E96DDB-61D3-E111-BEFB-001E67397D05.root'
    ),
    ##eventsToProcess = cms.untracked.VEventRange(
    ##    '1:104427:41737415'
    ##),
    ##skipEvents = cms.untracked.uint32(539)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#--------------------------------------------------------------------------------
# define configuration parameter default values

##type = 'SignalMC'
type = 'BackgroundMC'
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# define "hooks" for replacing configuration parameters
# in case running jobs on the CERN batch system/grid
#
#__type = #type#
#
isMC = None
if type == 'SignalMC' or type == 'BackgroundMC':
    isMC = True
else:
    isMC = False
#--------------------------------------------------------------------------------
   
process.produceAntiMuonDiscrMVATrainingNtupleSequence = cms.Sequence()

#--------------------------------------------------------------------------------
# rerun tau reconstruction with latest tags

process.load("RecoTauTag/Configuration/RecoPFTauTag_cff")
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.PFTau

##process.ak5PFJetsRecoTauChargedHadrons.verbosity = cms.int32(1)
##process.ak5PFJetsRecoTauChargedHadrons.builders[0].verbosity = cms.int32(1)

##process.combinatoricRecoTaus.modifiers[2].verbosity = cms.int32(1)

##process.hpsPFTauProducerSansRefs.verbosity = cms.int32(1)

##process.hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.verbosity = cms.int32(1)

##process.load("TauAnalysis/RecoTools/recoVertexSelection_cff")
##process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.selectPrimaryVertex

##process.dumpPFTaus = cms.EDAnalyzer("DumpPFTaus",
##    src = cms.InputTag('hpsPFTauProducer'),
##    srcTauRef = cms.InputTag('hpsPFTauProducer'),
##    srcTauRefDiscriminators = cms.VInputTag('hpsPFTauDiscriminationByDecayModeFinding'),
##    srcVertex = cms.InputTag('selectedPrimaryVertexHighestPtTrackSum')
##)
##process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.dumpPFTaus

process.tausForAntiMuonDiscrMVATraining = cms.EDFilter("PFTauSelector",
    src = cms.InputTag('hpsPFTauProducer'),
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs'),
            selectionCut = cms.double(0.5)
        )
    ),
    cut = cms.string("pt > 18. & abs(eta) < 2.4")
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.tausForAntiMuonDiscrMVATraining

process.tausForAntiMuonDiscrMVATrainingDiscriminationByDecayModeFinding = process.hpsPFTauDiscriminationByDecayModeFindingNewDMs.clone(
    PFTauProducer = cms.InputTag('tausForAntiMuonDiscrMVATraining')
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.tausForAntiMuonDiscrMVATrainingDiscriminationByDecayModeFinding

process.tausForAntiMuonDiscrMVATrainingPrimaryVertexProducer = process.hpsPFTauPrimaryVertexProducer.clone(
    PFTauTag = cms.InputTag("tausForAntiMuonDiscrMVATraining"),
    discriminators = cms.VPSet(),
    cut = cms.string('')
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.tausForAntiMuonDiscrMVATrainingPrimaryVertexProducer
process.tausForAntiMuonDiscrMVATrainingSecondaryVertexProducer = process.hpsPFTauSecondaryVertexProducer.clone(
    PFTauTag = cms.InputTag("tausForAntiMuonDiscrMVATraining")
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.tausForAntiMuonDiscrMVATrainingSecondaryVertexProducer
process.tausForAntiMuonDiscrMVATrainingTransverseImpactParameters = process.hpsPFTauTransverseImpactParameters.clone(
    PFTauTag =  cms.InputTag("tausForAntiMuonDiscrMVATraining"),
    PFTauPVATag = cms.InputTag("tausForAntiMuonDiscrMVATrainingPrimaryVertexProducer"),
    PFTauSVATag = cms.InputTag("tausForAntiMuonDiscrMVATrainingSecondaryVertexProducer")
)    
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.tausForAntiMuonDiscrMVATrainingTransverseImpactParameters

tauIdDiscriminatorsToReRun = [
    "hpsPFTauDiscriminationByDecayModeFindingNewDMs",
    "hpsPFTauDiscriminationByDecayModeFindingOldDMs",
    "hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr",
    "hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr",
    "hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr",
    "hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits",
    "hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits",
    "hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits",
    "hpsPFTauDiscriminationByIsolationMVAraw",
    "hpsPFTauDiscriminationByIsolationMVA2raw",
    "hpsPFTauMVA3IsolationChargedIsoPtSum",
    "hpsPFTauMVA3IsolationNeutralIsoPtSum",
    "hpsPFTauMVA3IsolationPUcorrPtSum",
    "hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw",
    "hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT",
    "hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT",
    "hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT",
    "hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT",
    "hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT",
    "hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT",    
    "hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw",
    "hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT",
    "hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT",
    "hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT",
    "hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT",
    "hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT",
    "hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT",
    "hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw",
    "hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT",
    "hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT",
    "hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT",
    "hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT",
    "hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT",
    "hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT",    
    "hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw",
    "hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT",
    "hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT",
    "hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT",
    "hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT",
    "hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT",
    "hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT",    
    "hpsPFTauDiscriminationByMVA3rawElectronRejection",
    "hpsPFTauDiscriminationByMVA3LooseElectronRejection",
    "hpsPFTauDiscriminationByMVA3MediumElectronRejection",
    "hpsPFTauDiscriminationByMVA3TightElectronRejection",
    "hpsPFTauDiscriminationByMVA3VTightElectronRejection",
    "hpsPFTauDiscriminationByMVA3rawElectronRejection",
    "hpsPFTauDiscriminationByMVA3LooseElectronRejection",
    "hpsPFTauDiscriminationByMVA3MediumElectronRejection",
    "hpsPFTauDiscriminationByMVA3TightElectronRejection",
    "hpsPFTauDiscriminationByMVA3VTightElectronRejection",
    "hpsPFTauDiscriminationByDeadECALElectronRejection",
    "hpsPFTauDiscriminationByLooseMuonRejection",
    "hpsPFTauDiscriminationByMediumMuonRejection",
    "hpsPFTauDiscriminationByTightMuonRejection",
    "hpsPFTauDiscriminationByLooseMuonRejection2",
    "hpsPFTauDiscriminationByMediumMuonRejection2",
    "hpsPFTauDiscriminationByTightMuonRejection2",
    "hpsPFTauDiscriminationByLooseMuonRejection3",
    "hpsPFTauDiscriminationByTightMuonRejection3",
    "hpsPFTauDiscriminationByMVArawMuonRejection",
    "hpsPFTauDiscriminationByMVALooseMuonRejection",
    "hpsPFTauDiscriminationByMVAMediumMuonRejection",
    "hpsPFTauDiscriminationByMVATightMuonRejection",
]
for tauIdDiscriminator in tauIdDiscriminatorsToReRun:
    moduleToClone = getattr(process, tauIdDiscriminator)
    module = moduleToClone.clone(
        PFTauProducer = cms.InputTag('tausForAntiMuonDiscrMVATraining')
    )
    if hasattr(module, "Prediscriminants") and hasattr(module.Prediscriminants, "decayMode"):
        module.Prediscriminants.decayMode.Producer = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByDecayModeFinding')
    # CV: handle MVA working-point discriminators that are based on cutting on the raw MVA output stored in another discriminator
    if hasattr(module, "key"):
        module.key = cms.InputTag(module.key.getModuleLabel().replace("hpsPFTau", "tausForAntiMuonDiscrMVATraining"),  module.key.getProductInstanceLabel())
    if hasattr(module, "toMultiplex"):
        module.toMultiplex = cms.InputTag(module.toMultiplex.getModuleLabel().replace("hpsPFTau", "tausForAntiMuonDiscrMVATraining"), module.toMultiplex.getProductInstanceLabel())
    if hasattr(module, "srcTauTransverseImpactParameters"):
        module.srcTauTransverseImpactParameters = cms.InputTag('tausForAntiMuonDiscrMVATrainingTransverseImpactParameters')
    for srcIsolation in [ "srcChargedIsoPtSum", "srcNeutralIsoPtSum", "srcPUcorrPtSum" ]:
        if hasattr(module, srcIsolation):
            moduleAttr = getattr(module, srcIsolation)
            moduleAttr = cms.InputTag(moduleAttr.getModuleLabel().replace("hpsPFTau", "tausForAntiMuonDiscrMVATraining"), moduleAttr.getProductInstanceLabel())
            setattr(module, srcIsolation, moduleAttr)    
    moduleName = tauIdDiscriminator.replace("hpsPFTau", "tausForAntiMuonDiscrMVATraining")
    setattr(process, moduleName, module)
    process.produceAntiMuonDiscrMVATrainingNtupleSequence += module
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# select generator level hadronic tau decays,
# veto jets overlapping with electrons or muons

process.prePFTauSequence = cms.Sequence()

if type == 'SignalMC' or type == 'BackgroundMC':
    process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
    process.prePFTauSequence += process.tauGenJets
    process.load("PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi")
    process.tauGenJetsSelectorAllHadrons.select = cms.vstring(
        'oneProng0Pi0', 
        'oneProng1Pi0', 
        'oneProng2Pi0', 
        'oneProngOther',
        'threeProng0Pi0', 
        'threeProng1Pi0', 
        'threeProngOther', 
        'rare'
    )
    process.prePFTauSequence += process.tauGenJetsSelectorAllHadrons

    jetCollection = None
    if type == 'SignalMC':
        process.tauGenJetsSelectorAllHadrons.filter = cms.bool(True)
                
        process.genTauMatchedPFJets = cms.EDFilter("PFJetAntiOverlapSelector",
            src = cms.InputTag('ak5PFJets'),
            srcNotToBeFiltered = cms.VInputTag(
                'tauGenJetsSelectorAllHadrons'
            ),
            dRmin = cms.double(0.3),
            invert = cms.bool(True),
            filter = cms.bool(False)                                                          
        )
        process.prePFTauSequence += process.genTauMatchedPFJets
        jetCollection = "genTauMatchedPFJets"
    elif type == 'BackgroundMC':
        process.genMuons = cms.EDFilter("GenParticleSelector",
            src = cms.InputTag("genParticles"),
            cut = cms.string('abs(pdgId) = 13 & pt > 2.'),
            stableOnly = cms.bool(True),
            filter = cms.bool(True)
        )
        process.prePFTauSequence += process.genMuons
        
        process.genMuonMatchedPFJets = cms.EDFilter("PFJetAntiOverlapSelector",
            src = cms.InputTag('ak5PFJets'),
            srcNotToBeFiltered = cms.VInputTag(
                'genMuons'
            ),
            dRmin = cms.double(0.3),
            invert = cms.bool(True),
            filter = cms.bool(False)                                                          
        )
        process.prePFTauSequence += process.genMuonMatchedPFJets

        ##process.dumpGenMuonMatchedPFJets = cms.EDAnalyzer("DumpPFJets",
        ##    src = cms.InputTag('genMuonMatchedPFJets'),
        ##    minPt = cms.double(10.)
        ##)
        ##process.prePFTauSequence += process.dumpGenMuonMatchedPFJets

        jetCollection = "genMuonMatchedPFJets"
    if not jetCollection:
        raise ValueError("Invalid Parameter 'jetCollection' = None !!")    
    process.ak5PFJetTracksAssociatorAtVertex.jets = cms.InputTag(jetCollection)
    process.ak5PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag(jetCollection)
    process.recoTauAK5PFJets08Region.src = cms.InputTag(jetCollection)
    process.ak5PFJetsRecoTauChargedHadrons.jetSrc = cms.InputTag(jetCollection)
    process.combinatoricRecoTaus.jetSrc = cms.InputTag(jetCollection)

process.produceAntiMuonDiscrMVATrainingNtupleSequence.replace(process.PFTau, process.prePFTauSequence + process.PFTau)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# select "good" reconstructed vertices
#
# CV: cut on ndof >= 4 if using 'offlinePrimaryVertices',
#                 >= 7 if using 'offlinePrimaryVerticesWithBS' as input
#
process.selectedOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
    filter = cms.bool(False)                                          
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.selectedOfflinePrimaryVertices

process.selectedOfflinePrimaryVerticesWithBS = process.selectedOfflinePrimaryVertices.clone(
    src = cms.InputTag('offlinePrimaryVerticesWithBS'),
    cut = cms.string("isValid & ndof >= 7 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2.")
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.selectedOfflinePrimaryVerticesWithBS
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# compute event weights for pile-up reweighting
# (Summer'12 MC to 2012 run ABCD data)

srcWeights = []
if isMC:
    from TauAnalysis.RecoTools.vertexMultiplicityReweight_cfi import vertexMultiplicityReweight
    process.vertexMultiplicityReweight3d2012RunABCD = vertexMultiplicityReweight.clone(
        inputFileName = cms.FileInPath("TauAnalysis/RecoTools/data/expPUpoissonMean_runs190456to208686_Mu17_Mu8.root"),
        type = cms.string("gen3d"),
        mcPeriod = cms.string("Summer12_S10")
    )
    process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.vertexMultiplicityReweight3d2012RunABCD
    srcWeights.extend([ 'vertexMultiplicityReweight3d2012RunABCD' ])
#--------------------------------------------------------------------------------

process.antiMuonDiscrMVATrainingNtupleProducer = cms.EDProducer("AntiMuonDiscrMVATrainingNtupleProducer",
    srcRecTaus = cms.InputTag('tausForAntiMuonDiscrMVATraining'),
    srcMuons = cms.InputTag('muons'),
    dRmuonMatch = cms.double(0.3),
    srcGenTauJets = cms.InputTag('tauGenJetsSelectorAllHadrons'),
    srcGenParticles = cms.InputTag('genParticles'),        
    minGenVisPt = cms.double(10.),                                          
    dRgenParticleMatch = cms.double(0.3),
    tauIdDiscriminators = cms.PSet(
        decayModeFindingNewDMs = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByDecayModeFindingNewDMs'),
        decayModeFindingOldDMs = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByDecayModeFindingOldDMs'),
        byLooseCombinedIsolationDeltaBetaCorr8Hits = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseCombinedIsolationDBSumPtCorr'),
        byMediumCombinedIsolationDeltaBetaCorr8Hits = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumCombinedIsolationDBSumPtCorr'),
        byTightCombinedIsolationDeltaBetaCorr8Hits = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightCombinedIsolationDBSumPtCorr'),
        byLooseCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits'),
        byMediumCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits'),
        byTightCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits'),
        byIsolationMVAraw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByIsolationMVAraw'),
        byIsolationMVA2raw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByIsolationMVA2raw'),
        byIsolationMVA3oldDMwoLTraw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByIsolationMVA3oldDMwoLTraw'),
        byVLooseIsolationMVA3oldDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVLooseIsolationMVA3oldDMwoLT'),
        byLooseIsolationMVA3oldDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseIsolationMVA3oldDMwoLT'),
        byMediumIsolationMVA3oldDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumIsolationMVA3oldDMwoLT'),
        byTightIsolationMVA3oldDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightIsolationMVA3oldDMwoLT'),
        byVTightIsolationMVA3oldDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVTightIsolationMVA3oldDMwoLT'),
        byVVTightIsolationMVA3oldDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVVTightIsolationMVA3oldDMwoLT'),                                                                    
        byIsolationMVA3oldDMwLTraw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByIsolationMVA3oldDMwLTraw'),
        byVLooseIsolationMVA3oldDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVLooseIsolationMVA3oldDMwLT'),
        byLooseIsolationMVA3oldDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseIsolationMVA3oldDMwLT'),
        byMediumIsolationMVA3oldDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumIsolationMVA3oldDMwLT'),
        byTightIsolationMVA3oldDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightIsolationMVA3oldDMwLT'),
        byVTightIsolationMVA3oldDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVTightIsolationMVA3oldDMwLT'),
        byVVTightIsolationMVA3oldDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVVTightIsolationMVA3oldDMwLT'),                                                                    
        byIsolationMVA3newDMwoLTraw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByIsolationMVA3newDMwoLTraw'),
        byVLooseIsolationMVA3newDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVLooseIsolationMVA3newDMwoLT'),
        byLooseIsolationMVA3newDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseIsolationMVA3newDMwoLT'),
        byMediumIsolationMVA3newDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumIsolationMVA3newDMwoLT'),
        byTightIsolationMVA3newDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightIsolationMVA3newDMwoLT'),
        byVTightIsolationMVA3newDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVTightIsolationMVA3newDMwoLT'),
        byVVTightIsolationMVA3newDMwoLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVVTightIsolationMVA3newDMwoLT'),                                                                    
        byIsolationMVA3newDMwLTraw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByIsolationMVA3newDMwLTraw'),
        byVLooseIsolationMVA3newDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVLooseIsolationMVA3newDMwLT'),
        byLooseIsolationMVA3newDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseIsolationMVA3newDMwLT'),
        byMediumIsolationMVA3newDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumIsolationMVA3newDMwLT'),
        byTightIsolationMVA3newDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightIsolationMVA3newDMwLT'),
        byVTightIsolationMVA3newDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVTightIsolationMVA3newDMwLT'),
        byVVTightIsolationMVA3newDMwLT = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByVVTightIsolationMVA3newDMwLT'),                                                                    
        againstElectronLooseMVA3 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVA3LooseElectronRejection'),
        againstElectronMediumMVA3 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVA3MediumElectronRejection'),
        againstElectronTightMVA3 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVA3TightElectronRejection'),
        againstElectronVTightMVA3 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVA3VTightElectronRejection'),
        againstElectronDeadECAL = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByDeadECALElectronRejection'),
        againstMuonLoose = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseMuonRejection'),
        againstMuonMedium = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumMuonRejection'),
        againstMuonTight = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightMuonRejection'),                                                            
        againstMuonLoose2 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseMuonRejection2'),
        againstMuonMedium2 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMediumMuonRejection2'),
        againstMuonTight2 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightMuonRejection2'),
        againstMuonLoose3 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByLooseMuonRejection3'),
        againstMuonTight3 = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByTightMuonRejection3'),                                                                    
        againstMuonMVAraw = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVArawMuonRejection'),                                                            
        againstMuonLooseMVA = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVALooseMuonRejection'),
        againstMuonMediumMVA = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVAMediumMuonRejection'),
        againstMuonTightMVA = cms.InputTag('tausForAntiMuonDiscrMVATrainingDiscriminationByMVATightMuonRejection'),                                            
    ),
    vertexCollections = cms.PSet(
        offlinePrimaryVertices = cms.InputTag('offlinePrimaryVertices'),
        offlinePrimaryVerticesWithBS = cms.InputTag('offlinePrimaryVerticesWithBS'),
        selectedOfflinePrimaryVertices = cms.InputTag('selectedOfflinePrimaryVertices'),
        selectedOfflinePrimaryVerticesWithBS = cms.InputTag('selectedOfflinePrimaryVerticesWithBS')
    ),                                                            
    isMC = cms.bool(True),                                                            
    srcWeights = cms.VInputTag(srcWeights)
)
process.produceAntiMuonDiscrMVATrainingNtupleSequence += process.antiMuonDiscrMVATrainingNtupleProducer

process.p = cms.Path(process.produceAntiMuonDiscrMVATrainingNtupleSequence)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("antiMuonDiscrMVATrainingNtuple.root")
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

processDumpFile = open('produceAntiMuonDiscrMVATrainingNtuple.dump', 'w')
print >> processDumpFile, process.dumpPython()




