import FWCore.ParameterSet.Config as cms

## from CondCore.DBCommon.CondDBSetup_cfi import *
from CondCore.CondDB.CondDB_cfi import *
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from JetMETCorrections.Configuration.JetCorrectors_cff import *

def L1NtupleCustomReco(process):


####  Custom Jet reco ####

    # load JEC from SQLite file
    process.load("CondCore.DBCommon.CondDBCommon_cfi")

    # re-apply JEC for AK4 CHS PF jets
    process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
    #process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')
    #process.load('JetMETCorrections.Configuration.CorrectedJetProducers_cff')

    #process.ak4PFCHSJetsL1FastL2L3Residual = process.ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL1FastL2L3ResidualCorrector'])

####  Custom Met Filter reco

    # load hbhe noise filter result producer
    process.load('CommonTools/RecoAlgos/HBHENoiseFilterResultProducer_cfi')

    # Type-1 pf MET correction
    process.load("JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff")
    process.load("JetMETCorrections.Type1MET.correctedMet_cff")


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
        +process.HBHENoiseFilterResultProducer
        +process.correctionTermsPfMetType1Type2
        +process.pfMetT1
        +process.egmGsfElectronIDSequence
        )
    
    process.schedule.append(process.l1CustomReco)

    return process



def getJECFromSQLite(process):

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
