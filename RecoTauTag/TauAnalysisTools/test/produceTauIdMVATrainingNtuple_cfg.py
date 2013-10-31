import FWCore.ParameterSet.Config as cms

process = cms.Process("produceTauIdMVATrainingNtuple")

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
        'file:/data1/veelken/CMSSW_5_3_x/skims/96E96DDB-61D3-E111-BEFB-001E67397D05.root'
        ##'file:/data1/veelken/CMSSW_5_3_x/skims/QCD_Pt-470to600_MuEnrichedPt5_TuneZ2star_8TeV_pythia6_AOD.root'
        ##'file:/data1/veelken/CMSSW_5_3_x/skims/selEvents_pfCandCaloEnNan_AOD.root'
        ##'/store/user/veelken/CMSSW_5_3_x/skims/simQCDmuEnrichedPt470to600_AOD_1_1_A8V.root'
        ##'/store/user/veelken/CMSSW_5_3_x/skims/simQCD_Pt-600to800_AOD_1_1_25m.root'
    ),
    ##eventsToProcess = cms.untracked.VEventRange(
    ##    '1:917:1719279',
    ##    '1:1022:1915188'
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
   
process.produceTauIdMVATrainingNtupleSequence = cms.Sequence()

#--------------------------------------------------------------------------------
# rerun tau reconstruction with latest tags

process.load("RecoTauTag/Configuration/RecoPFTauTag_cff")
# CV: hpsPFTauPrimaryVertexProducer module takes more than 5 seconds per event for high Pt QCD background samples
#    --> disable tau lifetime reconstruction for 'hpsPFTauProducer' collection for now and run it on 'tausForTauIdMVATraining' collection only !!
##process.produceAndDiscriminateHPSPFTaus.remove(process.hpsPFTauVertexAndImpactParametersSeq)
process.produceTauIdMVATrainingNtupleSequence += process.PFTau

##process.combinatoricRecoTaus.modifiers[2].verbosity = cms.int32(1)

##process.hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.verbosity = cms.int32(1)

##process.printEventContent = cms.EDAnalyzer("EventContentAnalyzer")
##process.PFTau.replace(process.combinatoricRecoTaus, process.printEventContent + process.combinatoricRecoTaus)

process.tausForTauIdMVATraining = cms.EDFilter("PFTauSelector",
    src = cms.InputTag('hpsPFTauProducer'),
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs'),
            selectionCut = cms.double(0.5)
        )
    ),
    cut = cms.string("pt > 18. & abs(eta) < 2.4")
)
process.produceTauIdMVATrainingNtupleSequence += process.tausForTauIdMVATraining

process.tausForTauIdMVATrainingDiscriminationByDecayModeFinding = process.hpsPFTauDiscriminationByDecayModeFindingNewDMs.clone(
    PFTauProducer = cms.InputTag('tausForTauIdMVATraining')
)
process.produceTauIdMVATrainingNtupleSequence += process.tausForTauIdMVATrainingDiscriminationByDecayModeFinding

dRisoCones = [ 0.4, 0.5, 0.8 ]
ptThresholds = [ 'Loose8Hits', 'Medium8Hits', 'Tight8Hits', 'Loose3Hits', 'Medium3Hits', 'Tight3Hits' ]
moduleNames_tauIsoPtSum = {} # key = (dRisoCone, 'loose'/'medium'/'tight', 'chargedIsoPtSum'/'neutralIsoPtSum'/'puCorrPtSum')
for dRisoCone in dRisoCones:
    moduleNames_tauIsoPtSum[dRisoCone] = {}
    for ptThreshold in ptThresholds:
        moduleNames_tauIsoPtSum[dRisoCone][ptThreshold] = {}

        moduleToClone = None
        if ptThreshold == 'Loose8Hits':
            moduleToClone = process.hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr
        elif ptThreshold == 'Medium8Hits':
            moduleToClone = process.hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr
        elif ptThreshold == 'Tight8Hits':
            moduleToClone = process.hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr
        elif ptThreshold == 'Loose3Hits':
            moduleToClone = process.hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits
        elif ptThreshold == 'Medium3Hits':
            moduleToClone = process.hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits
        elif ptThreshold == 'Tight3Hits':
            moduleToClone = process.hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits
        else:
            raise ValueError("Invalid Parameter 'ptThreshold' = %s !!" % ptThreshold)
    
        moduleChargedIsoPtSum = moduleToClone.clone(
            PFTauProducer = cms.InputTag('tausForTauIdMVATraining'),
            ApplyDiscriminationByECALIsolation = cms.bool(False),
            ApplyDiscriminationByTrackerIsolation = cms.bool(True),
            applySumPtCut = cms.bool(False),
            applyDeltaBetaCorrection = cms.bool(False),
            storeRawSumPt = cms.bool(True),
            storeRawPUsumPt = cms.bool(False),
            customOuterCone = cms.double(dRisoCone),
            isoConeSizeForDeltaBeta = cms.double(0.8*(dRisoCone/0.5)),
            verbosity = cms.int32(0)
        )
        moduleChargedIsoPtSum.Prediscriminants.decayMode.Producer = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDecayModeFinding')
        moduleNameChargedIsoPtSum = "tauChargedIsoPtSumDeltaR%02.0fPtThresholds%s" % (dRisoCone*10., ptThreshold)
        setattr(process, moduleNameChargedIsoPtSum, moduleChargedIsoPtSum)
        process.produceTauIdMVATrainingNtupleSequence += moduleChargedIsoPtSum
        moduleNames_tauIsoPtSum[dRisoCone][ptThreshold]['chargedIsoPtSum'] = moduleNameChargedIsoPtSum

        moduleNeutralIsoPtSum = moduleToClone.clone(
            PFTauProducer = cms.InputTag('tausForTauIdMVATraining'),
            ApplyDiscriminationByECALIsolation = cms.bool(True),
            ApplyDiscriminationByTrackerIsolation = cms.bool(False),
            applySumPtCut = cms.bool(False),
            applyDeltaBetaCorrection = cms.bool(False),
            storeRawSumPt = cms.bool(True),
            storeRawPUsumPt = cms.bool(False),
            customOuterCone = cms.double(dRisoCone),
            isoConeSizeForDeltaBeta = cms.double(0.8*(dRisoCone/0.5)),
            verbosity = cms.int32(0)
        )
        moduleNeutralIsoPtSum.Prediscriminants.decayMode.Producer = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDecayModeFinding')
        moduleNameNeutralIsoPtSum = "tauNeutralIsoPtSumDeltaR%02.0fPtThresholds%s" % (dRisoCone*10., ptThreshold)
        setattr(process, moduleNameNeutralIsoPtSum, moduleNeutralIsoPtSum)
        process.produceTauIdMVATrainingNtupleSequence += moduleNeutralIsoPtSum
        moduleNames_tauIsoPtSum[dRisoCone][ptThreshold]['neutralIsoPtSum'] = moduleNameNeutralIsoPtSum

        modulePUcorrPtSum = moduleToClone.clone(
            PFTauProducer = cms.InputTag('tausForTauIdMVATraining'),
            ApplyDiscriminationByECALIsolation = cms.bool(False),
            ApplyDiscriminationByTrackerIsolation = cms.bool(False),
            applySumPtCut = cms.bool(False),
            applyDeltaBetaCorrection = cms.bool(True),
            storeRawSumPt = cms.bool(False),
            storeRawPUsumPt = cms.bool(True),
            customOuterCone = cms.double(dRisoCone),
            isoConeSizeForDeltaBeta = cms.double(0.8*(dRisoCone/0.5)),
            verbosity = cms.int32(0)
        )
        modulePUcorrPtSum.Prediscriminants.decayMode.Producer = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDecayModeFinding')
        moduleNamePUcorrPtSum = "tauPUcorrPtSumDeltaR%02.0fPtThresholds%s" % (dRisoCone*10., ptThreshold)
        setattr(process, moduleNamePUcorrPtSum, modulePUcorrPtSum)
        process.produceTauIdMVATrainingNtupleSequence += modulePUcorrPtSum
        moduleNames_tauIsoPtSum[dRisoCone][ptThreshold]['puCorrPtSum'] = moduleNamePUcorrPtSum

process.tausForTauIdMVATrainingPrimaryVertexProducer = process.hpsPFTauPrimaryVertexProducer.clone(
    PFTauTag = cms.InputTag("tausForTauIdMVATraining"),
    discriminators = cms.VPSet(),
    cut = cms.string('')
)
process.produceTauIdMVATrainingNtupleSequence += process.tausForTauIdMVATrainingPrimaryVertexProducer
process.tausForTauIdMVATrainingSecondaryVertexProducer = process.hpsPFTauSecondaryVertexProducer.clone(
    PFTauTag = cms.InputTag("tausForTauIdMVATraining")
)
process.produceTauIdMVATrainingNtupleSequence += process.tausForTauIdMVATrainingSecondaryVertexProducer
process.tausForTauIdMVATrainingTransverseImpactParameters = process.hpsPFTauTransverseImpactParameters.clone(
    PFTauTag =  cms.InputTag("tausForTauIdMVATraining"),
    PFTauPVATag = cms.InputTag("tausForTauIdMVATrainingPrimaryVertexProducer"),
    PFTauSVATag = cms.InputTag("tausForTauIdMVATrainingSecondaryVertexProducer")
)    
process.produceTauIdMVATrainingNtupleSequence += process.tausForTauIdMVATrainingTransverseImpactParameters

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
        PFTauProducer = cms.InputTag('tausForTauIdMVATraining')
    )
    if hasattr(module, "Prediscriminants") and hasattr(module.Prediscriminants, "decayMode"):
        module.Prediscriminants.decayMode.Producer = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDecayModeFinding')
    # CV: handle MVA working-point discriminators that are based on cutting on the raw MVA output stored in another discriminator
    if hasattr(module, "key"):
        module.key = cms.InputTag(module.key.getModuleLabel().replace("hpsPFTau", "tausForTauIdMVATraining"),  module.key.getProductInstanceLabel())
    if hasattr(module, "toMultiplex"):
        module.toMultiplex = cms.InputTag(module.toMultiplex.getModuleLabel().replace("hpsPFTau", "tausForTauIdMVATraining"), module.toMultiplex.getProductInstanceLabel())
    if hasattr(module, "srcTauTransverseImpactParameters"):
        module.srcTauTransverseImpactParameters = cms.InputTag('tausForTauIdMVATrainingTransverseImpactParameters')
    for srcIsolation in [ "srcChargedIsoPtSum", "srcNeutralIsoPtSum", "srcPUcorrPtSum" ]:
        if hasattr(module, srcIsolation):
            moduleAttr = getattr(module, srcIsolation)
            moduleAttr = cms.InputTag(moduleAttr.getModuleLabel().replace("hpsPFTau", "tausForTauIdMVATraining"), moduleAttr.getProductInstanceLabel())
            setattr(module, srcIsolation, moduleAttr)
    moduleName = tauIdDiscriminator.replace("hpsPFTau", "tausForTauIdMVATraining")
    setattr(process, moduleName, module)
    process.produceTauIdMVATrainingNtupleSequence += module


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
        process.genElectrons = cms.EDFilter("GenParticleSelector",
            src = cms.InputTag("genParticles"),
            cut = cms.string('abs(pdgId) = 11 & pt > 10.'),
            stableOnly = cms.bool(True),
            filter = cms.bool(False)
        )
        process.prePFTauSequence += process.genElectrons

        process.genMuons = process.genElectrons.clone(
            cut = cms.string('abs(pdgId) = 13 & pt > 10.')
        )
        process.prePFTauSequence += process.genMuons
        
        process.pfJetsAntiOverlapWithLeptonsVeto = cms.EDFilter("PFJetAntiOverlapSelector",
            src = cms.InputTag('ak5PFJets'),
            srcNotToBeFiltered = cms.VInputTag(
                'genElectrons',
                'genMuons',
                'tauGenJetsSelectorAllHadrons'
            ),
            dRmin = cms.double(0.5),
            invert = cms.bool(False),
            filter = cms.bool(False)                                                          
        )
        process.prePFTauSequence += process.pfJetsAntiOverlapWithLeptonsVeto
        jetCollection = "pfJetsAntiOverlapWithLeptonsVeto"
    if not jetCollection:
        raise ValueError("Invalid Parameter 'jetCollection' = None !!")    
    process.ak5PFJetTracksAssociatorAtVertex.jets = cms.InputTag(jetCollection)
    process.ak5PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag(jetCollection)
    process.recoTauAK5PFJets08Region.src = cms.InputTag(jetCollection)
    process.ak5PFJetsRecoTauChargedHadrons.jetSrc = cms.InputTag(jetCollection)
    process.combinatoricRecoTaus.jetSrc = cms.InputTag(jetCollection)

process.produceTauIdMVATrainingNtupleSequence.replace(process.PFTau, process.prePFTauSequence + process.PFTau)
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
process.produceTauIdMVATrainingNtupleSequence += process.selectedOfflinePrimaryVertices

process.selectedOfflinePrimaryVerticesWithBS = process.selectedOfflinePrimaryVertices.clone(
    src = cms.InputTag('offlinePrimaryVerticesWithBS'),
    cut = cms.string("isValid & ndof >= 7 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2.")
)
process.produceTauIdMVATrainingNtupleSequence += process.selectedOfflinePrimaryVerticesWithBS
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# compute event weights for pile-up reweighting
# (Summer'12 MC to 2012 run ABCD data)

srcWeights = []
inputFileNameLumiCalc = None
if isMC:
    from TauAnalysis.RecoTools.vertexMultiplicityReweight_cfi import vertexMultiplicityReweight
    process.vertexMultiplicityReweight3d2012RunABCD = vertexMultiplicityReweight.clone(
        inputFileName = cms.FileInPath("TauAnalysis/RecoTools/data/expPUpoissonMean_runs190456to208686_Mu17_Mu8.root"),
        type = cms.string("gen3d"),
        mcPeriod = cms.string("Summer12_S10")
    )
    process.produceTauIdMVATrainingNtupleSequence += process.vertexMultiplicityReweight3d2012RunABCD
    srcWeights.extend([ 'vertexMultiplicityReweight3d2012RunABCD' ])
    inputFileNameLumiCalc = 'TauAnalysis/RecoTools/data/dummy.txt'
else:
    inputFileNameLumiCalc = 'TauAnalysis/RecoTools/data_nocrab/lumiCalc_2012RunABCD_byLS.out'
#--------------------------------------------------------------------------------

process.tauIdMVATrainingNtupleProducer = cms.EDProducer("TauIdMVATrainingNtupleProducer",
    srcRecTaus = cms.InputTag('tausForTauIdMVATraining'),
    srcRecTauTransverseImpactParameters = cms.InputTag('tausForTauIdMVATrainingTransverseImpactParameters'),                                            
    srcGenTauJets = cms.InputTag('tauGenJetsSelectorAllHadrons'),
    srcGenParticles = cms.InputTag('genParticles'),        
    minGenVisPt = cms.double(10.),                                          
    dRmatch = cms.double(0.3),
    tauIdDiscriminators = cms.PSet(
        decayModeFindingNewDMs = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDecayModeFindingNewDMs'),
        decayModeFindingOldDMs = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDecayModeFindingOldDMs'),
        byLooseCombinedIsolationDeltaBetaCorr8Hits = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseCombinedIsolationDBSumPtCorr'),
        byMediumCombinedIsolationDeltaBetaCorr8Hits = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumCombinedIsolationDBSumPtCorr'),
        byTightCombinedIsolationDeltaBetaCorr8Hits = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightCombinedIsolationDBSumPtCorr'),
        byLooseCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits'),
        byMediumCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits'),
        byTightCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits'),
        byIsolationMVAraw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByIsolationMVAraw'),
        byIsolationMVA2raw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByIsolationMVA2raw'),
        byIsolationMVA3oldDMwoLTraw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByIsolationMVA3oldDMwoLTraw'),
        byVLooseIsolationMVA3oldDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVLooseIsolationMVA3oldDMwoLT'),
        byLooseIsolationMVA3oldDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseIsolationMVA3oldDMwoLT'),
        byMediumIsolationMVA3oldDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumIsolationMVA3oldDMwoLT'),
        byTightIsolationMVA3oldDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightIsolationMVA3oldDMwoLT'),
        byVTightIsolationMVA3oldDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVTightIsolationMVA3oldDMwoLT'),
        byVVTightIsolationMVA3oldDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVVTightIsolationMVA3oldDMwoLT'),                                                            
        byIsolationMVA3oldDMwLTraw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByIsolationMVA3oldDMwLTraw'),
        byVLooseIsolationMVA3oldDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVLooseIsolationMVA3oldDMwLT'),
        byLooseIsolationMVA3oldDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseIsolationMVA3oldDMwLT'),
        byMediumIsolationMVA3oldDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumIsolationMVA3oldDMwLT'),
        byTightIsolationMVA3oldDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightIsolationMVA3oldDMwLT'),
        byVTightIsolationMVA3oldDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVTightIsolationMVA3oldDMwLT'),
        byVVTightIsolationMVA3oldDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVVTightIsolationMVA3oldDMwLT'),                                                            
        byIsolationMVA3newDMwoLTraw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByIsolationMVA3newDMwoLTraw'),
        byVLooseIsolationMVA3newDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVLooseIsolationMVA3newDMwoLT'),
        byLooseIsolationMVA3newDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseIsolationMVA3newDMwoLT'),
        byMediumIsolationMVA3newDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumIsolationMVA3newDMwoLT'),
        byTightIsolationMVA3newDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightIsolationMVA3newDMwoLT'),
        byVTightIsolationMVA3newDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVTightIsolationMVA3newDMwoLT'),
        byVVTightIsolationMVA3newDMwoLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVVTightIsolationMVA3newDMwoLT'),                                                            
        byIsolationMVA3newDMwLTraw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByIsolationMVA3newDMwLTraw'),
        byVLooseIsolationMVA3newDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVLooseIsolationMVA3newDMwLT'),
        byLooseIsolationMVA3newDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseIsolationMVA3newDMwLT'),
        byMediumIsolationMVA3newDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumIsolationMVA3newDMwLT'),
        byTightIsolationMVA3newDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightIsolationMVA3newDMwLT'),
        byVTightIsolationMVA3newDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVTightIsolationMVA3newDMwLT'),
        byVVTightIsolationMVA3newDMwLT = cms.InputTag('tausForTauIdMVATrainingDiscriminationByVVTightIsolationMVA3newDMwLT'),                                                            
        againstElectronLooseMVA3 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVA3LooseElectronRejection'),
        againstElectronMediumMVA3 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVA3MediumElectronRejection'),
        againstElectronTightMVA3 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVA3TightElectronRejection'),
        againstElectronVTightMVA3 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVA3VTightElectronRejection'),
        againstElectronDeadECAL = cms.InputTag('tausForTauIdMVATrainingDiscriminationByDeadECALElectronRejection'),
        againstMuonLoose = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseMuonRejection'),
        againstMuonMedium = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumMuonRejection'),
        againstMuonTight = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightMuonRejection'),
        againstMuonLoose2 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseMuonRejection2'),
        againstMuonMedium2 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMediumMuonRejection2'),
        againstMuonTight2 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightMuonRejection2'),
        againstMuonLoose3 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByLooseMuonRejection3'),
        againstMuonTight3 = cms.InputTag('tausForTauIdMVATrainingDiscriminationByTightMuonRejection3'),
        againstMuonMVAraw = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVArawMuonRejection'),                                                            
        againstMuonLooseMVA = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVALooseMuonRejection'),
        againstMuonMediumMVA = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVAMediumMuonRejection'),
        againstMuonTightMVA = cms.InputTag('tausForTauIdMVATrainingDiscriminationByMVATightMuonRejection')                                         
    ),
    isolationPtSums = cms.PSet(),
    vertexCollections = cms.PSet(
        offlinePrimaryVertices = cms.InputTag('offlinePrimaryVertices'),
        offlinePrimaryVerticesWithBS = cms.InputTag('offlinePrimaryVerticesWithBS'),
        selectedOfflinePrimaryVertices = cms.InputTag('selectedOfflinePrimaryVertices'),
        selectedOfflinePrimaryVerticesWithBS = cms.InputTag('selectedOfflinePrimaryVerticesWithBS')
    ),
    #--------------------------------------------------------
    # CV: pile-up information for Monte Carlo and data                                                               
    srcGenPileUpSummary = cms.InputTag('addPileupInfo'),
    inputFileNameLumiCalc = cms.FileInPath(inputFileNameLumiCalc),
    isMC = cms.bool(isMC),
    #--------------------------------------------------------                                                                       
    srcWeights = cms.VInputTag(srcWeights),
    verbosity = cms.int32(0)
)
for dRisoCone in dRisoCones:
    for ptThreshold in ptThresholds:
        pset = cms.PSet(
            chargedIsoPtSum = cms.InputTag(moduleNames_tauIsoPtSum[dRisoCone][ptThreshold]['chargedIsoPtSum']),
            neutralIsoPtSum = cms.InputTag(moduleNames_tauIsoPtSum[dRisoCone][ptThreshold]['neutralIsoPtSum']),
            puCorrPtSum = cms.InputTag(moduleNames_tauIsoPtSum[dRisoCone][ptThreshold]['puCorrPtSum']) 
        )
        psetName = "tauIsoDeltaR%02.0fPtThresholds%s" % (dRisoCone*10., ptThreshold)    
        setattr(process.tauIdMVATrainingNtupleProducer.isolationPtSums, psetName, pset)
process.produceTauIdMVATrainingNtupleSequence += process.tauIdMVATrainingNtupleProducer

process.p = cms.Path(process.produceTauIdMVATrainingNtupleSequence)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("tauIdMVATrainingNtuple.root")
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

processDumpFile = open('produceTauIdMVATrainingNtuple.dump', 'w')
print >> processDumpFile, process.dumpPython()




