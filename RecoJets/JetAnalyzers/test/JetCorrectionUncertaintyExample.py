# PYTHON configuration file.
# Description:  Example of plotting jet correction uncertainty.
# Author: Kalanand Mishra, Fermilab
# Date:  14 - January - 2011

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
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

##  ____             _ ____                           
## |  _ \ ___   ___ | / ___|  ___  _   _ _ __ ___ ___ 
## | |_) / _ \ / _ \| \___ \ / _ \| | | | '__/ __/ _ \
## |  __/ (_) | (_) | |___) | (_) | |_| | | | (_|  __/
## |_|   \___/ \___/|_|____/ \___/ \__,_|_|  \___\___|
                                                   

#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inputFile)
)



##  ____  _       _       
## |  _ \| | ___ | |_ ___ 
## | |_) | |/ _ \| __/ __|
## |  __/| | (_) | |_\__ \
## |_|   |_|\___/ \__|___/

####################################################### 
#############   calo jets ##
process.ak5calol2l3  = cms.EDAnalyzer('JetCorrectorDemo',
    JetCorrectionService     = cms.string('ak5CaloL2L3Residual'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string(''),
    PayloadName              = cms.string('AK5Calo'),
    NHistoPoints             = cms.int32(10000),
    NGraphPoints             = cms.int32(500),
    EtaMin                   = cms.double(-5),
    EtaMax                   = cms.double(5),
    PtMin                    = cms.double(10),
    PtMax                    = cms.double(1000),
    #--- eta values for JEC vs pt plots ----
    VEta                     = cms.vdouble(0.0,1.0,2.0,3.0,4.0),
    #--- corrected pt values for JEC vs eta plots ----
    VPt                      = cms.vdouble(20,30,50,100,200),
    Debug                    = cms.untracked.bool(False),
    UseCondDB                = cms.untracked.bool(True)
)


#############   pf jets ##
process.ak5pfl2l3  = cms.EDAnalyzer('JetCorrectorDemo',
    JetCorrectionService     = cms.string('ak5PFL2L3Residual'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string(''),
    PayloadName              = cms.string('AK5PF'),
    NHistoPoints             = cms.int32(10000),
    NGraphPoints             = cms.int32(500),
    EtaMin                   = cms.double(-5),
    EtaMax                   = cms.double(5),
    PtMin                    = cms.double(10),
    PtMax                    = cms.double(1000),
    #--- eta values for JEC vs pt plots ----
    VEta                     = cms.vdouble(0.0,1.0,2.0,3.0,4.0),
    #--- corrected pt values for JEC vs eta plots ----
    VPt                      = cms.vdouble(20,30,50,100,200),
    Debug                    = cms.untracked.bool(False),
    UseCondDB                = cms.untracked.bool(True)
)


#############   jpt jets ##
process.ak5jptl2l3  = cms.EDAnalyzer('JetCorrectorDemo',
    JetCorrectionService     = cms.string('ak5JPTL2L3Residual'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string(''),
    PayloadName              = cms.string('AK5JPT'),
    NHistoPoints             = cms.int32(10000),
    NGraphPoints             = cms.int32(500),
    EtaMin                   = cms.double(-5),
    EtaMax                   = cms.double(5),
    PtMin                    = cms.double(10),
    PtMax                    = cms.double(1000),
    #--- eta values for JEC vs pt plots ----
    VEta                     = cms.vdouble(0.0,1.0,2.0,3.0,4.0),
    #--- corrected pt values for JEC vs eta plots ----
    VPt                      = cms.vdouble(20,30,50,100,200),
    Debug                    = cms.untracked.bool(False),
    UseCondDB                = cms.untracked.bool(True)
)


##  ____       _   _     
## |  _ \ __ _| |_| |__  
## | |_) / _` | __| '_ \ 
## |  __/ (_| | |_| | | |
## |_|   \__,_|\__|_| |_|


#############   Path       ###########################
process.p = cms.Path(
    process.ak5calol2l3 +
    process.ak5pfl2l3 +
    process.ak5jptl2l3 
    )

