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
            
##isMC = False
isMC = True

##   ____             __ _                       _     _           
##  / ___|___  _ __  / _(_) __ _ _   _ _ __ __ _| |__ | | ___  ___ 
## | |   / _ \| '_ \| |_| |/ _` | | | | '__/ _` | '_ \| |/ _ \/ __|
## | |__| (_) | | | |  _| | (_| | |_| | | | (_| | |_) | |  __/\__ \
##  \____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|_.__/|_|\___||___/
##                         |___/                                   

NJetsToKeep = 2
CaloJetCollection = 'ak5CaloJets'
#PFJetCollection   = 'ak5PFJets'
JPTJetCollection  = 'JetPlusTrackZSPCorJetAntiKt5'
#GenJetCollection  = 'ak5GenJets'
PFJetCollection   = "goodPatJetsPFlow"
CAJetCollection   = "goodPatJetsCA8PrunedPF"
GenJetCollection  = "caPrunedGen"


PlotSuffix = "_Data"
if isMC:
  PlotSuffix = "_MC"

## inputFile = 'file:/uscms_data/d2/kalanand/dijet-Run2010A-JetMET-Nov4ReReco-9667events.root'
## if isMC:
##   inputFile ='/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root'


#inputFile = '/store/data/Run2011A/SingleMu/AOD/PromptReco-v6/000/173/692/EE90CCE0-E9CF-E011-B4C7-BCAEC54DB5D6.root'
#inputFile = '/store/user/lpctlbsm/srappocc/Jet/ttbsm_v8_Run2011-May10ReReco/0d3d9a54f3a29af186ad87df2a0c3ce1/ttbsm_42x_data_9_1_BzR.root'

inputFile = '/store/user/lpctlbsm/vasquez/Jet/ttbsm_v9_Run2011A-May10ReReco/f8e845a0332c56398831da6c30999af1/ttbsm_42x_data_60_1_EbE.root'

if isMC:
  #inputFile ='/store/mc/Spring11/QCD_Pt-15_TuneZ2_7TeV-pythia6/AODSIM/PU_S2_START311_V2-v1/0007/0AD300BD-5659-E011-B47A-002618943836.root'
  inputFile ='/store/user/lpctlbsm/srappocc/QCD_Pt-15to3000_TuneZ2_Flat_7TeV_pythia6/ttbsm_v8_Summer11-PU_S3_-START42_V11-v2/d870fa9b0dd695e8eb649b7e725d070f/ttbsm_42x_mc_86_2_fG3.root'
  

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
    input = cms.untracked.int32(1000)
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
    JetAlgorithm  = cms.string(CaloJetCollection),
    HistoFileName = cms.string('CaloJetPlotsExample'+PlotSuffix+'.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   PF Jets    ###########################
process.pf = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string(PFJetCollection),
    HistoFileName = cms.string('PFJetPlotsExample'+PlotSuffix+'.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   JPT Jets    ###########################
process.jpt = cms.EDAnalyzer("JPTJetPlotsExample",
    JetAlgorithm  = cms.string(JPTJetCollection),
    HistoFileName = cms.string('JPTJetPlotsExample'+PlotSuffix+'.root'),
    NJets         = cms.int32(NJetsToKeep)
)
#############   Gen Jets   ###########################
process.gen = cms.EDAnalyzer("GenJetPlotsExample",
     JetAlgorithm  = cms.string(GenJetCollection),
     HistoFileName = cms.string('GenJetPlotsExample'+PlotSuffix+'.root'),
     NJets         = cms.int32(NJetsToKeep)
)

#############   Cambridge-Aachen Jets R=0.8 ###########################
process.ca = cms.EDAnalyzer("PFJetPlotsExample",
    JetAlgorithm  = cms.string(CAJetCollection),
    HistoFileName = cms.string('CAJetPlotsExample'+PlotSuffix+'.root'),
    NJets         = cms.int32(NJetsToKeep)
)




#############   Path       ###########################
##  ____       _   _     
## |  _ \ __ _| |_| |__  
## | |_) / _` | __| '_ \ 
## |  __/ (_| | |_| | | |
## |_|   \__,_|\__|_| |_|

##process.myseq = cms.Sequence(process.calo*process.pf*process.jpt*process.gen)
process.myseq = cms.Sequence(process.ca * process.gen)
  
if not isMC: 
  process.myseq.remove ( process.gen )

process.p = cms.Path( process.myseq)

