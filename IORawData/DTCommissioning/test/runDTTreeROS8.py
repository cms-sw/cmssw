import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDPGAnalyis")


##process.load("UserCode.DTDPGAnalysis.dt_dpganalysis_common_cff_cosmics_miniDAQ")
##process.load("UserCode.DTDPGAnalysis.testLocalDAQ")
process.load("UserCode.DTDPGAnalysis.testLocalDAQ_ROS8")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("EmptySource",
      firstRun= cms.untracked.uint32(3163),
      numberEventsInLuminosityBlock = cms.untracked.uint32(200),
      numberEventsInRun       = cms.untracked.uint32(0)
 )

process.rawDataCollector = cms.EDProducer('DTNewROS8FileReader',
         isRaw = cms.untracked.bool(True),
         numberOfHeaderWords = cms.untracked.int32(10),  ## No esta incluido en el default del codigo de la ROS8
         eventsAnalysisRate = cms.untracked.int32(1),

         fileName = cms.untracked.string(
        'file:./run27082015_3163'
         )
    
)                                                


from  CondCore.CondDB.CondDB_cfi import *

###### MAP ############################################################################
###process.mappingsource = cms.ESSource("PoolDBESSource",
###    CondDBSetup,
###    timetype = cms.string('runnumber'),
###    toGet = cms.VPSet(cms.PSet(record = cms.string('DTReadOutMappingRcd'),
###                               tag = cms.string('DTReadoutMapping_Rob3Mb1')
######                               tag = cms.string('DTReadOutMapping_compact_V04')
###                               )
###                      ),
###    connect = cms.string('frontier://Frontier/CMS_COND_31X_DT'),
###    authenticationMethod = cms.untracked.uint32(1)
###    )
###
###process.es_prefer_mappingsource = cms.ESPrefer('PoolDBESSource','mappingsource')
###### END MAP ########################################################################

###### tTrig  #########################################################################
process.ttrigsource = cms.ESSource("PoolDBESSource", 
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(record = cms.string('DTTtrigRcd'),
                               label = cms.untracked.string('cosmics'),  ## ONLY if using cosmic reconstruction  
                               tag = cms.string('ttrig')
                               )
                      ),
    ## connect = cms.string('frontier://Frontier/CMS_COND_31X_DT'),
    ## connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DT/OfflineCode/GIF2015/LocalDataBases/ttrig_ROS8_1.db'),
    ## connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DT/OfflineCode/GIF2015/LocalDataBases/ttrig_ROS8_1_p520.db'),
    ## connect = cms.string('sqlite_file:ttrig_ROS8_1_p530.db'),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DT/OfflineCode/GIF2015/LocalDataBases/ttrig_ROS8_Tbox410.db'),
    authenticationMethod = cms.untracked.uint32(0)
    )

process.es_prefer_ttrigsource = cms.ESPrefer('PoolDBESSource','ttrigsource')
###### END tTRIG  #####################################################################



###process.MessageLogger = cms.Service("MessageLogger",
###                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')),
###                                    destinations = cms.untracked.vstring('cout')
###                                    )

#------------------
# DT Analyisis
#------------------

# MAGNETIC FIELD
#### B = 0 Tesla ###############################################################
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
####process.SteppingHelixPropagator.useInTeslaFromMagField = True
####process.SteppingHelixPropagator.SetVBFPointer = True
#### B = 3.8 Tesla #############################################################
##process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#--------------------------------------------------------

from UserCode.DTDPGAnalysis.DTOfflineAnalyzer_cfi import *
process.DTOfflineAnalyzer.DTLocalTriggerLabel = 'dtunpacker'
process.DTOfflineAnalyzer.doSA = False
##process.DTOfflineAnalyzer.doWheelm2 = False
##process.DTOfflineAnalyzer.doWheelm1 = False
##process.DTOfflineAnalyzer.doWheel0 = False
##process.DTOfflineAnalyzer.doWheel1 = False
##process.DTOfflineAnalyzer.doWheel2 = False

process.DTOfflineAnalyzer.doTBox = False
process.DTOfflineAnalyzer.doTBoxWhm2 = False
process.DTOfflineAnalyzer.doTBoxWhm1 = False
process.DTOfflineAnalyzer.doTBoxWh0 = False
process.DTOfflineAnalyzer.doTBoxWh1 = False
process.DTOfflineAnalyzer.doTBoxWh2 = False
process.DTOfflineAnalyzer.doTBoxSector  = 0 ## =0 => All Sectors, =N => Sector N
process.DTOfflineAnalyzer.doTBoxChamber = 3 ## =0 => All Chambers,=N => Chamber N
process.DTOfflineAnalyzer.doTBoxSuperLayer = 0 ## =0 => All SuperLayers,=N => SuperLayer N
process.DTOfflineAnalyzer.doTBoxLayer = 0 ## =0 => All Layers,=N => Layer N


process.load("DQM.DTMonitorModule.dtTriggerTask_cfi")
process.dtTriggerMonitor.process_dcc = True
process.dtTriggerMonitor.dcc_label   = 'dttfunpacker'
process.dtTriggerMonitor.process_seg = True

#--------------------------------------------------------
##process.UpdaterService = cms.Service("UpdaterService")  ###  Only needed for STA reco
#--------------------------------------------------------


process.load("UserCode/DTDPGAnalysis/DTTTreGenerator_cfi")
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.myDTNtuple.outputFile = "DTNtuple_LocalRun.root"
##process.myDTNtuple.dtTrigSimDCCLabel = cms.InputTag("dtTriggerPrimitiveDigis")
process.myDTNtuple.dtDigiLabel = cms.InputTag("dtunpacker")
process.myDTNtuple.localDTmuons = cms.untracked.bool(True)
process.myDTNtuple.localDTmuonsWithoutTrigger = cms.untracked.bool(True)  ## This avoid the "standard DCC & DDU triggers  if uncommented
                                                                          ## the program will probably crash
process.myDTNtuple.localDTmuonsWithROS8Trigger = cms.untracked.bool(True) ## << To include the ROS8 Trigger data in the root tree
                                                                          ## Can be put as true even if there is not this triggers but
                                                                          ## only in case the ROS8 unpacker is used 
## Uncomment and change the next parameters in case you need to increase the 
##total number of digis/segments... per event allowed in the root size
##process.myDTNtuple.dtDigiSize = cms.int32(300),
##process.myDTNtuple.dtSegmentSize = cms.int32(50),
##process.myDTNtuple.dtTrigROS8Size = cms.int32(50),


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *', 
                                                                      'keep *_MEtoEDMConverter_*_*'),
                               fileName = cms.untracked.string('DQMOfflineDTDPG.root')
                               )

##process.p = cms.Path( process.dtunpacker )
process.p = cms.Path( process.rawDataCollector+process.dtunpacker * process.reco  + process.sources + process.MEtoEDMConverter  + process.DTOfflineAnalyzer +process.myDTNtuple) 

process.ep = cms.EndPath( process.out )

