import FWCore.ParameterSet.Config as cms

## from CondCore.DBCommon.CondDBSetup_cfi import *
from CondCore.CondDB.CondDB_cfi import *
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from JetMETCorrections.Configuration.JetCorrectors_cff import *
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

def L1NtupleCustomReco(process):


####  Custom Jet reco ####

    # load JEC from SQLite file
    process.load("CondCore.CondDB.CondDB_cfi")

    # re-apply JEC for AK4 CHS PF jets
    process.load('JetMETCorrections.Configuration.JetCorrectors_cff')

    process.load('L1Trigger.L1TNtuples.l1JetRecoTree_cfi')

    addJetCollection(
        process,
        labelName = "CorrectedPuppiJets",
        jetSource = process.l1JetRecoTree.puppiJetToken,
        jetCorrections = ('AK4PFPuppi', ['L1FastJet', 'L2Relative', 'L3Absolute','L2L3Residual'], 'None'),
        pfCandidates = cms.InputTag("particleFlow"),
        algo= 'AK', rParam = 0.4,
        getJetMCFlavour=False
    )
    delattr(process, 'patJetGenJetMatchCorrectedPuppiJets')
    delattr(process, 'patJetPartonMatchCorrectedPuppiJets')



####  Custom Met Filter reco

    # load hbhe noise filter result producer
    process.load('CommonTools/RecoAlgos/HBHENoiseFilterResultProducer_cfi')

    # Type-1 pf MET correction
    process.load("JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff")
    process.load("JetMETCorrections.Type1MET.correctedMet_cff")

    # Bad PF Muon filter for MET & HT
    process.load('RecoMET.METFilters.BadPFMuonFilter_cfi')
    process.BadPFMuonFilter.muons = cms.InputTag("muons")
    process.BadPFMuonFilter.PFCandidates = cms.InputTag("particleFlow")

    # bad charged candidate filter
    process.load('RecoMET.METFilters.BadChargedCandidateFilter_cfi')
    process.BadChargedCandidateFilter.muons = cms.InputTag("muons")
    process.BadChargedCandidateFilter.PFCandidates = cms.InputTag("particleFlow")


####  Custom E/Gamma reco ####

    # turn on VID producer, indicate data format  to be
    # DataFormat.AOD or DataFormat.MiniAOD, as appropriate 
    dataFormat = DataFormat.AOD
    switchOnVIDElectronIdProducer(process, dataFormat)
    process.load("RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cfi")
    process.egmGsfElectronIDSequence = cms.Sequence(process.egmGsfElectronIDs)
    # define which IDs we want to produce
    idmod = 'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff'  
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)




    process.l1CustomReco = cms.Path(
        process.ak4PFCHSL1FastL2L3ResidualCorrectorChain
        +process.ak4CaloL1FastL2L3ResidualCorrectorChain
        +process.HBHENoiseFilterResultProducer
        +process.correctionTermsPfMetType1Type2
        +process.pfMetT1
        +process.egmGsfElectronIDSequence
        +process.BadPFMuonFilter
        +process.BadChargedCandidateFilter
        )
    
    process.schedule.append(process.l1CustomReco)

    return process


def getJECFromSQLite(process):

    process.load("CondCore.CondDB.CondDB_cfi")

    process.jec = cms.ESSource(
        "PoolDBESSource",
        DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(0)
            ),
        timetype = cms.string('runnumber'),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('JetCorrectionsRecord'),
                # for data
                tag    = cms.string('JetCorrectorParametersCollection_Summer15_25nsV6_DATA_AK4PFchs'),
                # for MC
                #tag    = cms.string('JetCorrectorParametersCollection_Fall15_25nsV2_MC_AK4PFchs'),
                label  = cms.untracked.string('AK4PFCHS')
                ),
            ), 
        connect = cms.string('sqlite:Summer15_25nsV6_DATA.db')
        # uncomment above tag lines and this comment to use MC JEC
        # connect = cms.string('sqlite:Fall15_25nsV2_MC.db')
        )
    
    process.es_prefer_jec = cms.ESPrefer('PoolDBESSource','jec')
    
    return process
