# PYTHON configuration file.
# Description:  Example of applying default (L2+L3) jet corrections.
# Author: K. Kousouris
# Date:  02 - September - 2009
# Date:  22 - November - 2009: Kalanand Mishra: Modified for 3.3.X (re-Reco) corrections
# Date:  14 - January - 2011: Kalanand Mishra: Modified for 3.8.X 

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
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = GLOBAL_TAG
process.load("FWCore.MessageService.MessageLogger_cfi")


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



##      _ _____ ____ 
##     | | ____/ ___|
##  _  | |  _|| |    
## | |_| | |__| |___ 
##  \___/|_____\____|
                  

#############   Include the jet corrections ##########
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
ak5CaloJetsCor = cms.InputTag("ak5CaloJetsL2L3")
ak5PFJetsCor = cms.InputTag("ak5PFJetsL2L3")
ak5JPTJetsCor = cms.InputTag("ak5JPTJetsL2L3")
if not isMC:
    ak5CaloJetsCor = cms.InputTag("ak5CaloJetsL2L3Residual")
    ak5PFJetsCor = cms.InputTag("ak5PFJetsL2L3Residual")
    ak5JPTJetsCor = cms.InputTag("ak5JPTJetsL2L3Residual")


##  ____       _           _   _             
## / ___|  ___| | ___  ___| |_(_) ___  _ __  
## \___ \ / _ \ |/ _ \/ __| __| |/ _ \| '_ \ 
##  ___) |  __/ |  __/ (__| |_| | (_) | | | |
## |____/ \___|_|\___|\___|\__|_|\___/|_| |_|
                                          

#############   Apply selection cuts ##
process.ak5CaloJetsSel = cms.EDFilter("CaloJetSelector",  
    src = ak5CaloJetsCor,
    cut = cms.string('pt > 20.0 && eta<3.0 && eta>-3.0')
)
process.ak5PFJetsSel = cms.EDFilter("PFJetSelector",  
    src = ak5PFJetsCor,
    cut = cms.string('pt > 20.0 && eta<3.0 && eta>-3.0')
)

process.ak5JPTJetsSel = cms.EDFilter("JPTJetSelector",  
    src = ak5JPTJetsCor,
    cut = cms.string('pt > 20.0 && eta<3.0 && eta>-3.0')
)

##  ____  _       _       
## |  _ \| | ___ | |_ ___ 
## | |_) | |/ _ \| __/ __|
## |  __/| | (_) | |_\__ \
## |_|   |_|\___/ \__|___/

####################################################### 
#############   User analyzer (corrected calo jets) ##
process.correctedAK5Calo = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string("ak5CaloJetsSel"),
    HistoFileName = cms.string('CorJetHisto_AK5Calo.root'),
    NJets         = cms.int32(NJetsToKeep)
    )
#############   User analyzer (corrected pf jets) ##
process.correctedAK5PF = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string("ak5PFJetsSel"),
    HistoFileName = cms.string('CorJetHisto_AK5PF.root'),
    NJets         = cms.int32(NJetsToKeep)
    )
#############   User analyzer (corrected jpt jets) #####
process.correctedAK5JPT = cms.EDAnalyzer("JPTJetPlotsExample",
    JetAlgorithm    = cms.string("ak5JPTJetsSel"),
    HistoFileName   = cms.string('CorJetHisto_AK5JPT.root'),
    NJets           = cms.int32(NJetsToKeep)
    )



##  ____       _   _     
## |  _ \ __ _| |_| |__  
## | |_) / _` | __| '_ \ 
## |  __/ (_| | |_| | | |
## |_|   \__,_|\__|_| |_|


#############   Path       ###########################
process.p = cms.Path( process.ak5CaloJetsL2L3 +
                      process.ak5CaloJetsL2L3Residual + 
                      process.ak5CaloJetsSel +
                      process.correctedAK5Calo +                      
                      process.ak5PFJetsL2L3 +
                      process.ak5PFJetsL2L3Residual +
                      process.ak5PFJetsSel +
                      process.correctedAK5PF  +                      
                      process.ak5JPTJetsL2L3 +
                      process.ak5JPTJetsL2L3Residual +                      
                      process.ak5JPTJetsSel +
                      process.correctedAK5JPT
                      )
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

