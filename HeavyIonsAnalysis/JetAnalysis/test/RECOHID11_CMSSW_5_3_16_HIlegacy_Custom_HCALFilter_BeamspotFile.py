# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s RAW2DIGI,L1Reco,RECO,USER:EventFilter/HcalRawToDigi/hcallaserhbhehffilter2012_cff.hcallLaser2012Filter,DQM --customise Configuration/DataProcessing/RecoTLR.customisePromptHI --inline_custom --process reRECO --data --eventcontent RECO,DQM --scenario HeavyIons --datatier RECO,DQM --repacked --conditions FT_R_53_LV6::All --no_exec --filein=/store/hidata/HIRun2011/HIMinBiasUPC/RAW/v1/000/182/124/FE6DE36B-8F13-E111-B683-002481E0DEC6.root --fileout file:step2.root -n 5 --python RECOHID11_CMSSW_5_3_16_HIlegacy_Custom_HCALFilter.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('reRECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('EventFilter.HcalRawToDigi.hcallaserhbhehffilter2012_cff')
process.load('DQMOffline.Configuration.DQMOfflineHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/hidata/HIRun2011/HIMinBiasUPC/RAW/v1/000/182/124/FE6DE36B-8F13-E111-B683-002481E0DEC6.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('step2 nevts:5'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('file:step2.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('RECO')
    ),
                                      SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('filter_step')
    )
                                      )
# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'FT_R_53_LV6::All', '')

process.load("FWCore.Modules.preScaler_cfi")
process.preScaler.prescaleFactor = 20

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.preScaler*process.RawToDigi)
process.L1Reco_step = cms.Path(process.preScaler*process.L1Reco)
process.reconstruction_step = cms.Path(process.preScaler*process.reconstructionHeavyIons)
process.user_step = cms.Path(process.preScaler*process.hcallLaser2012Filter)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)
process.filter_step = cms.Path(process.preScaler)

# Schedule definition
process.schedule = cms.Schedule(process.filter_step,process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.user_step,process.endjob_step,process.RECOoutput_step)

from Configuration.PyReleaseValidation.ConfigBuilder import MassReplaceInputTag
MassReplaceInputTag(process)

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.RecoTLR

#gone with the fact that there is no difference between production and development sequence
#def customiseCommon(process):
#    return (process)


##############################################################################
def customisePPData(process):
    #deprecated process= customiseCommon(process)
    ##all customisation for data are now deprecated to Reconstruction_Data_cff
    #left as a place holder to alter production sequences in case of emergencies
    return process


##############################################################################
def customisePPMC(process):
    #deprecated process=customiseCommon(process)
    #left as a place holder to alter production sequences in case of emergencies    
    return process

##############################################################################
def customiseCosmicData(process):

    return process

##############################################################################
def customiseCosmicMC(process):
    
    return process
        
##############################################################################
def customiseVALSKIM(process):
    print "WARNING"
    print "this method is outdated, please use RecoTLR.customisePPData"
    process= customisePPData(process)
    return process
                
##############################################################################
def customiseExpress(process):
    process= customisePPData(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePrompt(process):
    process= customisePPData(process)

    #add the lumi producer in the prompt reco only configuration
    process.reconstruction_step+=process.lumiProducer
    return process

##############################################################################
##############################################################################

#gone with the fact that there is no difference between production and development sequence
#def customiseCommonHI(process):
#    return process

##############################################################################
def customiseExpressHI(process):
    #deprecated process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePromptHI(process):
    #deprecated process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################

def planBTracking(process):

    # stuff from LowPtTripletStep_cff
    process.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin=0.3

    # stuff from PixelLessStep_cff
    process.pixelLessStepClusters.oldClusterRemovalInfo=cms.InputTag("tobTecStepClusters")
    process.pixelLessStepClusters.trajectories= cms.InputTag("tobTecStepTracks")
    process.pixelLessStepClusters.overrideTrkQuals=cms.InputTag('tobTecStepSelector','tobTecStep')
    process.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.7
    process.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.5

    # stuff from PixelPairStep_cff
    process.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6

    # stuff from TobTecStep_cff
    process.tobTecStepClusters.oldClusterRemovalInfo=cms.InputTag("detachedTripletStepClusters")
    process.tobTecStepClusters.trajectories= cms.InputTag("detachedTripletStepTracks")
    process.tobTecStepClusters.overrideTrkQuals=cms.InputTag('detachedTripletStep')
    process.tobTecStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 5.0

    # stuff from DetachedTripletStep_cff
    process.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin=0.35

    # stuff from iterativeTk_cff
    process.iterTracking = cms.Sequence(process.InitialStep*
                                        process.LowPtTripletStep*
                                        process.PixelPairStep*
                                        process.DetachedTripletStep*
                                        process.TobTecStep*
                                        process.PixelLessStep*
                                        process.generalTracks*
                                        process.ConvStep*
                                        process.conversionStepTracks
                                        )
    
    
    # stuff from RecoTracker_cff
    process.newCombinedSeeds.seedCollections=cms.VInputTag(
        cms.InputTag('initialStepSeeds'),
        cms.InputTag('pixelPairStepSeeds'),
    #    cms.InputTag('mixedTripletStepSeeds'),
        cms.InputTag('pixelLessStepSeeds')
        )

    # stuff from Kevin's fragment
    process.generalTracks.TrackProducers = (cms.InputTag('initialStepTracks'),
                                            cms.InputTag('lowPtTripletStepTracks'),
                                            cms.InputTag('pixelPairStepTracks'),
                                            cms.InputTag('detachedTripletStepTracks'),
                                            cms.InputTag('pixelLessStepTracks'),
                                            cms.InputTag('tobTecStepTracks'))
    process.generalTracks.hasSelector=cms.vint32(1,1,1,1,1,1)
    process.generalTracks.selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                                             cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                                             cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                                             cms.InputTag("detachedTripletStep"),
                                                             cms.InputTag("pixelLessStepSelector","pixelLessStep"),
                                                             cms.InputTag("tobTecStepSelector","tobTecStep")
                                                             )
    process.generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True) ) )


    if hasattr(process,'dqmoffline_step'):
        process.dqmoffline_step.remove(process.TrackMonStep4)
        #process.dqmoffline_step.remove(process.TrackMonStep5)
        
    return process

#call to customisation function customisePromptHI imported from Configuration.DataProcessing.RecoTLR
process = customisePromptHI(process)

# End of customisation functions

# Load Beamspot
from CondCore.DBCommon.CondDBSetup_cfi import *
process.beamspot = cms.ESSource("PoolDBESSource",CondDBSetup,
                                toGet = cms.VPSet(cms.PSet( record = cms.string('BeamSpotObjectsRcd'),
                                                            tag= cms.string('BeamSpotObjects_2009_LumiBased_SigmaZ_v27_offline')
                                                            )),
                                connect =cms.string('sqlite_file:BeamSpotObjects_2009_LumiBased_SigmaZ_v27_offline.db')
                                )
process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","beamspot")

