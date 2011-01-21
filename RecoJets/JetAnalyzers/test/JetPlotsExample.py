# PYTHON configuration file for class: JetPlotsExample
# Description:  Example of simple EDAnalyzer for jets.
# Author: K. Kousouris
# Date:  25 - August - 2008
# Modified: Kalanand Mishra
# Date:  11 - January - 2011 (for CMS Data Analysis School jet exercise)


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

inputFile = 'file:/uscms_data/d2/kalanand/dijet-Run2010A-JetMET-Nov4ReReco-9667events.root'
if isMC:
  inputFile ='/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root'
  

##   _            _           _           
## (_)_ __   ___| |_   _  __| | ___  ___ 
## | | '_ \ / __| | | | |/ _` |/ _ \/ __|
## | | | | | (__| | |_| | (_| |  __/\__ \
## |_|_| |_|\___|_|\__,_|\__,_|\___||___/
    
process = cms.Process("Ana")
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


##  ____  _       _       
## |  _ \| | ___ | |_ ___ 
## | |_) | |/ _ \| __/ __|
## |  __/| | (_) | |_\__ \
## |_|   |_|\___/ \__|___/

#############   Calo Jets  ###########################
process.calo = cms.EDAnalyzer("CaloJetPlotsExample",
    JetAlgorithm  = cms.string('ak5CaloJets'),
    HistoFileName = cms.string('CaloJetPlotsExample.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   PF Jets    ###########################
process.pf = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string('ak5PFJets'),
    HistoFileName = cms.string('PFJetPlotsExample.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   JPT Jets    ###########################
process.jpt = cms.EDAnalyzer("JPTJetPlotsExample",
    JetAlgorithm  = cms.string('JetPlusTrackZSPCorJetAntiKt5'),
    HistoFileName = cms.string('JPTJetPlotsExample.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   Gen Jets   ###########################
if isMC:
    process.gen = cms.EDAnalyzer("GenJetPlotsExample",
        JetAlgorithm  = cms.string('ak5GenJets'),
        HistoFileName = cms.string('GenJetPlotsExample.root'),
        NJets         = cms.int32(NJetsToKeep)
    )

#############   Path       ###########################

##  ____       _   _     
## |  _ \ __ _| |_| |__  
## | |_) / _` | __| '_ \ 
## |  __/ (_| | |_| | | |
## |_|   \__,_|\__|_| |_|

    
if isMC:    
    process.p = cms.Path(process.calo*process.pf*process.jpt*process.gen)
else:
    process.p = cms.Path(process.calo*process.pf*process.jpt)    


