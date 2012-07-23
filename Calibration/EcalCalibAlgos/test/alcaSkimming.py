import FWCore.ParameterSet.Config as cms
import os
import sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing
#sys.path(".")
#
#    _____             __ _                        _   _
#   / ____|           / _(_)                      | | (_)
#   | |     ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __
#   | |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \
#   | |___| (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | |
#    \_____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
#                             __/ |
#                            |___/

### setup 'sandbox'  options
options = VarParsing.VarParsing('standard')
# sandbox, sandboxRereco, alcaAOD, alcaMCAOD
# sandbox -> create sandbox (ALCARAW)
# sandboxRereco -> make the rereco (ALCARECO)
options.register ('type',
                  "sandbox",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: sandbox, sandboxRereco, ALCARECO, ALCARECOSIM")
options.register ('isCrab',
                  1, # default value True
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "bool: is a crab job")
options.register ('tagFile',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "path of the file with the reReco tags")
options.register('skim',
                 "", 
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "type of skim: ZSkim, WSkim, EleSkim (at least one electron), ''")
                
### setup any defaults you want
#options.outputFile = '/uscms/home/cplager/nobackup/outputFiles/try_3.root'
options.output="alcaSkimSandbox.root"
#options.files= "file:///tmp/shervin/RAW-RECO.root"
options.files= "root://eoscms//eos/cms/store/group/alca_ecalcalib/sandbox/RAW-RECO.root"
#options.files= "file:///tmp/shervin/MC-AODSIM.root"
#'file1.root', 'file2.root'
options.maxEvents = -1 # -1 means all events
### get and parse the command line arguments
options.parseArguments()

print options
# Use the options

# Do you want to filter events? 
HLTFilter = False
ZSkim = False
WSkim = False

if(options.skim=="ZSkim"):
    ZSkim=True
elif(options.skim=="WSkim"):
    WSkim=True
else:
    if(options.type=="sandbox"):
        print "[ERROR] no skim selected"
#        sys.exit(-1)


MC = False  # please specify it if starting from AOD
if(options.type == "sandbox"):
    processName = 'ALCASKIM'
#    ZSkim = True
#    WSkim = True
elif(options.type == "sandboxRereco"):
    processName = 'ALCARERECO'
elif(options.type == "ALCARECOSIM"):
    processName = 'ALCARECO'
    MC = True
    ZSkim=True
    WSkim=False
elif(options.type == "ALCARECO"):
    processName = 'ALCARECO'
    MC = False
    ZSkim=True
    WSkim=False
else:
    print "[ERROR] wrong type defined"
    sys.exit(-1)
    
isCrab=options.isCrab




#    _____  __             _             _         _
#   / ____|/ _|           | |           | |       | |
#   | |    | |_ __ _   ___| |_ __ _ _ __| |_ ___  | |__   ___ _ __ ___
#   | |    |  _/ _` | / __| __/ _` | '__| __/ __| | '_ \ / _ \ '__/ _ \
#   | |____| || (_| | \__ \ || (_| | |  | |_\__ \ | | | |  __/ | |  __/
#    \_____|_| \__, | |___/\__\__,_|_|   \__|___/ |_| |_|\___|_|  \___|
#               __/ |
#              |___/



process = cms.Process(processName)
#process.prescaler = cms.EDFilter("Prescaler",
#                                    prescaleFactor = cms.int32(prescale),
#                                    prescaleOffset = cms.int32(0)
#                                    )
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff')

process.load('Configuration.EventContent.EventContent_cff')

# added by Shervin for ES recHits (saved as in AOD): large window 15x3 (strip x row)
process.load('RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi')



#process.MessageLogger.cerr.FwkReport.reportEvery = 500
process.MessageLogger.cerr = cms.untracked.PSet(
    optionalPSet = cms.untracked.bool(True),
    INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    ),
    noTimeStamps = cms.untracked.bool(False),
    FwkReport = cms.untracked.PSet(
    optionalPSet = cms.untracked.bool(True),
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
    ),
    default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    ),
    Root_NoDictionary = cms.untracked.PSet(
                 optionalPSet = cms.untracked.bool(True),
                 limit = cms.untracked.int32(0)
                 ),
    FwkJob = cms.untracked.PSet(
    optionalPSet = cms.untracked.bool(True),
    limit = cms.untracked.int32(0)
    ),
    FwkSummary = cms.untracked.PSet(
    optionalPSet = cms.untracked.bool(True),
    reportEvery = cms.untracked.int32(1),
    limit = cms.untracked.int32(10000000)
    ),
    threshold = cms.untracked.string('INFO')
    )
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

#lcg-cp -v  --vo cms \
#srm://cmsrm-se01.roma1.infn.it:8443/srm/managerv2?SFN=/pnfs/roma1.infn.it/data/cms/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/FEA9F397-DFE1-E011-BA6C-0026B94D1B09.root \
#/tmp/shervin/RAW-RECO.root

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.files),
                            secondaryFileNames = cms.untracked.vstring(options.secondaryFiles)
                            )

# try to drop as much as possible to reduce the running time
# process.source.inputCommands = cms.untracked.vstring("keep *",
#                                                      "drop recoPFTaus*_*_*_*", 
#                                                      "drop recoPFTauDiscriminator*_*_*_*",
#                                                      "drop *_tevMuons*_*_*",
# #                                                     "drop *muon*_*_*_*",
# #                                                     "keep *Electron*_*_*_",
# #                                                     "keep *electron*_*_*_*"
#                                                      )

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Other statements
#

CMSSW_VERSION=os.getenv("CMSSW_VERSION")

if(len(options.tagFile)>0):
    execfile(options.tagFile) # load the GT 
    process.GlobalTag = RerecoGlobalTag 
else:
    if(options.type=="sandboxRereco"):
        print "******************************"
        print "[ERROR] no file with tags specified, but rereco requested"
        sys.exit(1)
        
    if(re.match("CMSSW_4_2_.*",CMSSW_VERSION)):
        if (MC):
            print "[INFO] Using GT START42_V12::All"
            process.GlobalTag.globaltag = 'START42_V12::All'
        else:
            print "[INFO] Using GT GR_P_V22::All"
            process.GlobalTag.globaltag = 'GR_P_V22::All' #GR_R_42_V21B::All' # rereco30Nov
    elif(re.match("CMSSW_5_2_.*",CMSSW_VERSION)):
        if(MC):
            print "[INFO] Using GT START52_V9::All"
            process.GlobalTag.globaltag = 'START52_V9::All'
        else:
            process.GlobalTag.globaltag = 'GR_P_V32::All' # 5_2_0 Prompt
            #            process.GlobalTag.globaltag = 'GR_R_52_V7::All' # 5_2_0
    elif(re.match("CMSSW_5_3_.*",CMSSW_VERSION)):
        if(MC):
            print "[INFO] Using GT START52_V9::All"
            process.GlobalTag.globaltag = 'START52_V9::All'
        else:
            process.GlobalTag.globaltag = 'GR_P_V40::All' # 5_2_0 Prompt
            #            process.GlobalTag.globaltag = 'GR_R_52_V7::All' # 5_2_0
    else:
        print "[ERROR]::Global Tag not set for CMSSW_VERSION: ", CMSSW_VERSION
    



        
###############################
# Event filter sequence: process.filterSeq
# sanbox sequence: process.sandboxSeq
# sandbox rereco sequence: process.sandboxRerecoSeq
# alcareco event reduction: process.alcarecoSeq
#
    
    
################################# FILTERING EVENTS
process.filterSeq = cms.Sequence()
#process.load('calibration.SANDBOX.trackerDrivenFinder_cff')

if (ZSkim):
    process.load('Calibration.EcalCalibAlgos.ZElectronSkimSandbox_cff')
    process.filterSeq *= process.ZeeFilterSeq
elif (WSkim):
    process.load("DPGAnalysis.Skims.WElectronSkim_cff")
    process.filterSeq *= process.elecMetSeq
elif(options.skim=="EleSkim"):
    process.MinEleNumberFilter = cms.EDFilter("CandViewCountFilter",
                                              src = cms.InputTag("gsfElectrons"),
                                              minNumber = cms.uint32(1)
                                              )
    process.filterSeq *= process.MinEleNumberFilter
                

if (HLTFilter):
    import copy
    from HLTrigger.HLTfilters.hltHighLevel_cfi import *
    process.ZEEHltFilter = copy.deepcopy(hltHighLevel)
    process.ZEEHltFilter.throw = cms.bool(False)
    process.ZEEHltFilter.HLTPaths = ["HLT_Ele*"]
    process.filterSeq *= process.ZEEHltFilter




###############################
# ECAL Recalibration
###############################

if (options.type=="sandbox"):
    process.load('Calibration.EcalCalibAlgos.sandboxSeq_cff')
    # this module provides:
    #process.sandboxSeq  = uncalibRecHitSeq



if(not options.type=="sandbox"):
    # I want to reduce the recHit collections to save space
    process.load('Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi')
    #============================== TO BE CHECKED FOR PRESHOWER
    process.load("RecoEcal.EgammaClusterProducers.reducedRecHitsSequence_cff")
    process.reducedEcalRecHitsES.scEtThreshold = cms.double(0.)
    #if(not runFromALCA):
    process.reducedEcalRecHitsES.EcalRecHitCollectionES = cms.InputTag('ecalPreshowerRecHit','EcalRecHitsES')
    process.reducedEcalRecHitsES.noFlag = cms.bool(True)
    process.reducedEcalRecHitsES.OutputLabel_ES = cms.string('alCaRecHitsES')
    
    process.alcarecoSeq = cms.Sequence(process.alCaIsolatedElectrons + process.reducedEcalRecHitsES)

    #==============================

    
if(options.type=="sandboxRereco"):
    process.load('Calibration.EcalCalibAlgos.sandboxRerecoSeq_cff')
    # this module provides:
    # process.electronRecoSeq
    # process.electronClusteringSeq # with ele-SC reassociation
    # process.sandboxRerecoSeq = (electronRecoSeq * electronClusteringSeq)
    process.ecalRecHit.EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB","ALCASKIM")
    process.ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE","ALCASKIM")

    process.correctedHybridSuperClusters.corectedSuperClusterCollection = 'recalibSC'
    process.correctedMulti5x5SuperClustersWithPreshower.corectedSuperClusterCollection = 'endcapRecalibSC'
    if(re.match("CMSSW_5_.*",CMSSW_VERSION)):
        process.multi5x5PreshowerClusterShape.endcapSClusterProducer = "correctedMulti5x5SuperClustersWithPreshower:endcapRecalibSC"

    # in sandboxRereco
    process.reducedEcalRecHitsES.EndcapSuperClusterCollection= cms.InputTag('correctedMulti5x5SuperClustersWithPreshower','endcapRecalibSC',processName)

    process.alCaIsolatedElectrons.electronLabel = cms.InputTag("electronRecalibSCAssociator")
    process.alCaIsolatedElectrons.ebRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    process.alCaIsolatedElectrons.eeRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")

if((not options.type=="sandboxRereco") ):
    process.load('RecoJets.Configuration.RecoPFJets_cff')
    process.kt6PFJetsForRhoCorrection = process.kt6PFJets.clone(doRhoFastjet = True)
    process.kt6PFJetsForRhoCorrection.Rho_EtaMax = cms.double(2.5)
    process.rhoFastJetSeq = cms.Sequence(process.kt6PFJetsForRhoCorrection) 
    
if(options.type=="sandbox"):
    process.ZPath = cms.Path( process.filterSeq * process.trackerDrivenOnlyElectrons * process.rhoFastJetSeq *
                              process.sandboxSeq)
elif(options.type=="sandboxRereco"):
    process.ZPath = cms.Path( process.sandboxRerecoSeq * process.alcarecoSeq)
elif(options.type == "ALCARECO"):
    process.ZPath = cms.Path( process.filterSeq * process.rhoFastJetSeq * process.alcarecoSeq)
elif(options.type == "ALCARECOSIM"):
    process.ZPath = cms.Path( process.filterSeq * process.rhoFastJetSeq * process.alcarecoSeq)


process.load('Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff')

if(options.type=="sandbox"):
    from Calibration.EcalCalibAlgos.sandboxOutput_cff import *
    process.OutALCARECOEcalCalElectron.outputCommands +=  sandboxOutputCommands
    
if(options.type == "sandboxRereco"):
    from Calibration.EcalCalibAlgos.sandboxRerecoOutput_cff import *
    process.OutALCARECOEcalCalElectron.outputCommands += sandboxRerecoOutputCommands 


if(isCrab):
    fileName = cms.untracked.string(options.output)
else:
    fileName = cms.untracked.string('output/'+options.output)

process.output = cms.OutputModule("PoolOutputModule",
                                  maxSize = cms.untracked.int32(3072000),
                                  outputCommands = process.OutALCARECOEcalCalElectron.outputCommands,
                                          fileName = fileName,
#                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('zFilterPath')),
                                          dataset = cms.untracked.PSet(
    filterName = cms.untracked.string(''),
    dataTier = cms.untracked.string('ALCARECO')
    )
                                          )                                          
print "OUTPUTCOMMANDS"
print process.output.outputCommands
 
process.ALCARECOoutput_step = cms.EndPath(process.output)

#process.schedule = cms.Schedule(process.zFilterPath,process.ALCARECOoutput_step)
