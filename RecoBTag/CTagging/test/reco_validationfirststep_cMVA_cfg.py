import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Core.DQM_cfg")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.cerr.threshold = 'ERROR'

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff") 
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.GlobalTag.globaltag = cms.string("MCRUN2_74_V7::All") #contains latest-greatest CSVv2+IVF training

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

#DB FILE WITH ALL RECORDS +  NEW cMVA RECORD!!!
#process.load("CondCore.DBCommon.CondDBSetup_cfi")
#process.BTauMVAJetTagComputerRecord = cms.ESSource("PoolDBESSource",
#	process.CondDBSetup,
#	timetype = cms.string('runnumber'),
#	toGet = cms.VPSet(cms.PSet(
#		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
#                tag = cms.string('MVAJetTags')
#	)),
#	connect = cms.string("sqlite_file:MVAJetTags_newCMVA_SE_bugFixed.db"), 
	#connect = cms.string('frontier://FrontierDev/CMS_COND_BTAU'),
#	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
#)
#process.es_prefer_BTauMVAJetTagComputerRecord = cms.ESPrefer("PoolDBESSource","BTauMVAJetTagComputerRecord")


# DQM include
process.load("Configuration.EventContent.EventContent_cff")
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# write DQM file
process.DQMoutput = cms.OutputModule("PoolOutputModule",
  splitLevel = cms.untracked.int32(0),
  outputCommands = process.DQMEventContent.outputCommands,
  fileName = cms.untracked.string('DQMfile.root'),
  #fileName = cms.untracked.string('DQMfile.root'),
  dataset = cms.untracked.PSet(
    filterName = cms.untracked.string(''),
    dataTier = cms.untracked.string('')
  )
)

#JTA for your jets
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
process.myak4JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
                                                  j2tParametersVX,
                                                  jets = cms.InputTag("ak4PFJetsCHS") 
                                                  )
#new input for impactParameterTagInfos
from RecoBTag.Configuration.RecoBTag_cff import *
process.impactParameterTagInfos.jetTracks = cms.InputTag("myak4JetTracksAssociatorAtVertex")

#select good primary vertex
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
process.goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
    )

#for Inclusive Vertex Finder
process.load('RecoVertex/AdaptiveVertexFinder/inclusiveVertexing_cff')
process.load('RecoBTag/SecondaryVertex/inclusiveSecondaryVertexFinderTagInfos_cfi')
process.inclusiveVertexFinder.primaryVertices = cms.InputTag("goodOfflinePrimaryVertices")
process.trackVertexArbitrator.primaryVertices = cms.InputTag("goodOfflinePrimaryVertices")


#input for softLeptonTagInfos
from RecoBTag.SoftLepton.softPFElectronTagInfos_cfi import *
process.softPFElectronsTagInfos.primaryVertex = cms.InputTag('goodOfflinePrimaryVertices')
process.softPFElectronsTagInfos.jets = cms.InputTag("ak4PFJetsCHS")
process.softPFElectronsTagInfos.genParticles = cms.InputTag("genParticles")
process.softPFElectronsTagInfos.useMCpromptElectronFilter = cms.bool(True) ###to implement
from RecoBTag.SoftLepton.softPFMuonTagInfos_cfi import *
process.softPFMuonsTagInfos.primaryVertex = cms.InputTag('goodOfflinePrimaryVertices')
process.softPFMuonsTagInfos.jets = cms.InputTag("ak4PFJetsCHS")
process.softPFMuonsTagInfos.genParticles = cms.InputTag("genParticles")
process.softPFMuonsTagInfos.useMCpromptMuonFilter = cms.bool(True) ###to implement


# for the PU ID
# Select GenJets with Pt>8 GeV
process.ak4GenJetsMCPUJetID = cms.EDFilter("GenJetSelector",
    src    = cms.InputTag("ak4GenJets"),
    cut    = cms.string('pt > 8.0'),
    filter = cms.bool(False)             # in case no GenJets pass the selection, do not filter events, just produce an empty GenJet collection
)

# Match selected GenJets to RecoJets
process.ak4PFJetsGenJetMatchMCPUJetID = cms.EDProducer("GenJetMatcher",  # cut on deltaR; pick best by deltaR
    src                   = cms.InputTag("ak4PFJetsCHS"),           # RECO jets (any View<Jet> is ok)
    matched               = cms.InputTag("ak4GenJetsMCPUJetID"), # GEN jets  (must be GenJetCollection)
    mcPdgId               = cms.vint32(),                        # N/A
    mcStatus              = cms.vint32(),                        # N/A
    checkCharge           = cms.bool(False),                     # N/A
    maxDeltaR             = cms.double(0.25),                    # Minimum deltaR for the match
    resolveAmbiguities    = cms.bool(True),                      # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),
)

#for the flavour matching
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone()

from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
process.jetFlavourInfosAK4PFJets = ak4JetFlavourInfos.clone()
process.jetFlavourInfosAK4PFJets.jets = cms.InputTag("ak4PFJetsCHS")

from RecoBTag.Configuration.RecoBTag_cff import *

process.combinedMVAComputer.jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('jetProbabilityComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('combinedSecondaryVertexV2Computer') #default is combinedSecondaryVertexComputer
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFMuonComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFElectronComputer')
		)
)
process.candidateCombinedMVAComputer.jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('candidateJetProbabilityComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer') #default is candidateCombinedSecondaryVertexComputer
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFMuonComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFElectronComputer')
		)
)

process.combinedMVAComputerNEW = combinedMVAComputer.clone(
	calibrationRecord = cms.string('CombinedMVAnew'), 
	jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('jetProbabilityComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('combinedSecondaryVertexV2Computer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFMuonComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFElectronComputer')
		)
	)
)
process.combinedMVABJetTagsNEW = combinedMVABJetTags.clone(
	jetTagComputer = cms.string('combinedMVAComputerNEW')
)
process.candidateCombinedMVAComputerNEW = candidateCombinedMVAComputer.clone(
	calibrationRecord = cms.string('CombinedMVAnew'), 
	jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('candidateJetProbabilityComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFMuonComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFElectronComputer')
		)
	)
)
process.pfCombinedMVABJetTagsNEW =pfCombinedMVABJetTags.clone(
	jetTagComputer = cms.string('candidateCombinedMVAComputerNEW')
)

#standard validation tools
from DQMOffline.RecoB.bTagCommon_cff import*
process.load("DQMOffline.RecoB.bTagCommon_cff")

from Validation.RecoB.bTagAnalysis_cfi import *
process.load("Validation.RecoB.bTagAnalysis_cfi")
process.bTagValidation.jetMCSrc = 'jetFlavourInfosAK4PFJets'
process.bTagValidation.genJetsMatched = 'ak4PFJetsGenJetMatchMCPUJetID'
#process.bTagValidation.allHistograms = True 
#process.bTagValidation.fastMC = True
process.bTagValidation.ptRanges = cms.vdouble(0.0,40.0,60.0,90.0, 150.0,400.0,600.0,3000.0)
process.bTagValidation.etaRanges = cms.vdouble(0.0, 1.2, 2.1, 2.4)
process.bTagValidation.doPUid = cms.bool(True)
process.bTagValidation.doJEC = cms.bool(False)
process.bTagValidation.flavPlots = cms.string("alldusg")

process.bTagValidation.tagConfig += cms.VPSet(
#		cms.PSet(
#	          cTagGenericAnalysisBlock,
#            label = cms.InputTag("combinedInclusiveSecondaryVertexV2BJetTags"),
#            folder = cms.string("CSVv2_tkOnly")
#		),
#                cms.PSet(
#	    parameters = cms.PSet(
#                                discriminatorStart = cms.double(-0.01),
#                                discriminatorEnd = cms.double(1.011),
#                                nBinEffPur = cms.int32(200),
#                                effBConst = cms.double(0.5),
#                                endEffPur = cms.double(1.005),
#                                startEffPur = cms.double(0.005),
#                                doCTagPlots = cms.bool(True)
#                                ),
#            label = cms.InputTag("pfCombinedCvsLJetTags"),
#            folder = cms.string("charmtagger_CvsL")
#                ),
	)
	
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)


process.btagDQM = cms.Path(
process.ak4GenJetsMCPUJetID *
process.ak4PFJetsGenJetMatchMCPUJetID *
process.goodOfflinePrimaryVertices * 
process.inclusiveVertexing * 
#process.inclusiveMergedVerticesFiltered * 
#process.bToCharmDecayVertexMerged * 
process.myak4JetTracksAssociatorAtVertex * 
process.impactParameterTagInfos * 
process.inclusiveSecondaryVertexFinderTagInfos *
process.softPFMuonsTagInfos *
process.softPFElectronsTagInfos *
process.selectedHadronsAndPartons *
process.jetFlavourInfosAK4PFJets *
btagging * 											#rerun all taggers (legacy and pf-based)
#process.combinedMVABJetTagsNEW * 				
#process.pfCombinedMVABJetTagsNEW *
process.bTagValidation
)

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(
  process.btagDQM,
  process.endjob_step,
  process.DQMoutput_step
)

process.PoolSource.fileNames = [
'root://xrootd.unl.edu//store/relval/CMSSW_7_6_2/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v12-v1/00000/7075D514-AC9C-E511-B595-0CC47A4D7650.root',
'root://xrootd.unl.edu//store/relval/CMSSW_7_6_2/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v12-v1/00000/BA0EA887-AB9C-E511-8B56-003048FFD744.root',
'root://xrootd.unl.edu//store/relval/CMSSW_7_6_2/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v12-v1/00000/E6FB9740-A89C-E511-A5A0-0CC47A74527A.root'
#'root://xrootd.unl.edu//store/relval/CMSSW_7_6_0_pre5/RelValTTbar_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v1-v1/00000/12321B4A-3B5C-E511-8FB9-0025905A6092.root'
#'root://cms-xrd-global.cern.ch//store/relval/CMSSW_7_5_0_pre5/RelValProdTTbar_13/AODSIM/MCRUN2_75_V5-v1/00000/5C03C5DF-310B-E511-829C-0025905A606A.root',
#'root://cms-xrd-global.cern.ch//store/relval/CMSSW_7_5_0_pre5/RelValProdTTbar_13/AODSIM/MCRUN2_75_V5-v1/00000/E8F4AA13-310B-E511-AA77-0025905A606A.root'
#'file:/data/scratchLocal/RelValTTbar_13_CMSSW_7_5_0_pre5-MCRUN2_75_V5-v1_GEN-SIM-RECO/383E39B7-BA0B-E511-A509-0025905B8582.root',
#      'file:/data/scratchLocal/RelValTTbar_13_CMSSW_7_5_0_pre5-MCRUN2_75_V5-v1_GEN-SIM-RECO/4AE0AEB0-BA0B-E511-86F9-002618943857.root',
#      'file:/data/scratchLocal/RelValTTbar_13_CMSSW_7_5_0_pre5-MCRUN2_75_V5-v1_GEN-SIM-RECO/F0BEEE9E-120B-E511-88DD-0025905A60C6.root'
#'file:/data/scratchLocal/RelValTTbar_13_CMSSW_7_6_0_pre5-76X_mcRun2_asymptotic_v1-v1_GEN-SIM-RECO/12321B4A-3B5C-E511-8FB9-0025905A6092.root'
##'root://cms-xrd-global.cern.ch//store/mc/Phys14DR/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/AODSIM/PU20bx25_PHYS14_25_V1-v1/00000/000470E0-3B75-E411-8B90-00266CFFA604.root'
##'root://cms-xrd-global.cern.ch//store/mc/RunIISpring15DR74/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v2/00000/06B5178E-F008-E511-A2CF-00261894390B.root'
]

#open('pydump.py','w').write(process.dumpPython())
