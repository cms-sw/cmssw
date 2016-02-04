# PYTHON configuration file for class: JetIdExample
# Description:  Example showing how to conveniently select collection
# of jets in the event which pass the jet id criteria ("loose",
# "medium", "tight",..etc) using a simple one-line plugin.
#
# Author: Kalanand Mishra
# Date:  18 - January - 2011

import FWCore.ParameterSet.Config as cms

##  ____        _                       __  __  ____ 
## |  _ \  __ _| |_ __ _    ___  _ __  |  \/  |/ ___|
## | | | |/ _` | __/ _` |  / _ \| '__| | |\/| | |    
## | |_| | (_| | || (_| | | (_) | |    | |  | | |___ 
## |____/ \__,_|\__\__,_|  \___/|_|    |_|  |_|\____|
                                                  
isMC = True


##   ____             __ _                       _     _           
##  / ___|___  _ __  / _(_) __ _ _   _ _ __ __ _| |__ | | ___  ___ 
## | |   / _ \| '_ \| |_| |/ _` | | | | '__/ _` | '_ \| |/ _ \/ __|
## | |__| (_) | | | |  _| | (_| | |_| | | | (_| | |_) | |  __/\__ \
##  \____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|_.__/|_|\___||___/
##                         |___/                                   

NJetsToKeep = 2
GLOBAL_TAG = 'GR_R_38X_V15::All'
inputFile = 'file:/uscms_data/d2/kalanand/dijet-Run2010A-JetMET-Nov4ReReco-9667events.root'

if isMC:
    GLOBAL_TAG = 'START38_V14::All'
    inputFile ='/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root'


##   _            _           _           
## (_)_ __   ___| |_   _  __| | ___  ___ 
## | | '_ \ / __| | | | |/ _` |/ _ \/ __|
## | | | | | (__| | |_| | (_| |  __/\__ \
## |_|_| |_|\___|_|\__,_|\__,_|\___||___/
                                      
   
process = cms.Process("Ana")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.GlobalTag.globaltag = GLOBAL_TAG

#############   Format MessageLogger #################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10


##  ____             _ ____                           
## |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
## | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
## |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
## |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
                                                   

#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inputFile)
)
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")


##      _      _     ___    _ 
##     | | ___| |_  |_ _|__| |
##  _  | |/ _ \ __|  | |/ _` |
## | |_| |  __/ |_   | | (_| |
##  \___/ \___|\__| |___\__,_|
                           

#############   JetID: Calo Jets  ###########################
process.load("RecoJets.JetProducers.ak5JetID_cfi")
process.CaloJetsLooseId = cms.EDProducer("CaloJetIdSelector",
    src     = cms.InputTag( "ak5CaloJets" ),                                     
    idLevel = cms.string("LOOSE"),                            
    jetIDMap = cms.untracked.InputTag("ak5JetID")
)

process.CaloJetsTightId = cms.EDProducer("CaloJetIdSelector",
    src     = cms.InputTag( "ak5CaloJets" ),                                             
    idLevel = cms.string("TIGHT"),                            
    jetIDMap = cms.untracked.InputTag("ak5JetID")
)
#############   JetID: PF Jets    ###########################
process.PFJetsLooseId = cms.EDProducer("PFJetIdSelector",
    src     = cms.InputTag( "ak5PFJets" ),                                     
    idLevel = cms.string("LOOSE")
)

process.PFJetsTightId = cms.EDProducer("PFJetIdSelector",
    src     = cms.InputTag( "ak5PFJets" ),                                             
    idLevel = cms.string("TIGHT")
)


##  ____  _       _       
## |  _ \| | ___ | |_ ___ 
## | |_) | |/ _ \| __/ __|
## |  __/| | (_) | |_\__ \
## |_|   |_|\___/ \__|___/

#######################################################
#############   Analysis: Calo Jets  ##################
process.caloJetAnalysisLooseId = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('CaloJetsLooseId'),
    HistoFileName = cms.string('CaloJetPlotsExample_LooseId.root'),
    NJets         = cms.int32(NJetsToKeep)
)
process.caloJetAnalysisTightId = process.caloJetAnalysisLooseId.clone()
process.caloJetAnalysisTightId.JetAlgorithm = cms.string('CaloJetsTightId')
process.caloJetAnalysisTightId.HistoFileName = cms.string('CaloJetPlotsExample_TightId.root')

#############    Analysis: PF Jets  ###################
process.pfJetAnalysisLooseId = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string('PFJetsLooseId'),
    HistoFileName = cms.string('PFJetPlotsExample_LooseId.root'),
    NJets         = cms.int32(NJetsToKeep)
)
process.pfJetAnalysisTightId = process.pfJetAnalysisLooseId.clone()
process.pfJetAnalysisTightId.JetAlgorithm = cms.string('PFJetsTightId')
process.pfJetAnalysisTightId.HistoFileName = cms.string('PFJetPlotsExample_TightId.root')



##  ____       _   _     
## |  _ \ __ _| |_| |__  
## | |_) / _` | __| '_ \ 
## |  __/ (_| | |_| | | |
## |_|   \__,_|\__|_| |_|

## #############   Path       ###########################
process.p = cms.Path( process.ak5JetID +
                      process.CaloJetsLooseId +
                      process.CaloJetsTightId +
                      process.PFJetsLooseId +
                      process.PFJetsTightId +
                      process.caloJetAnalysisLooseId +
                      process.caloJetAnalysisTightId +                      
                      process.pfJetAnalysisLooseId +
                      process.pfJetAnalysisTightId 
                      )
