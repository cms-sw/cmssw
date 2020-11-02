import FWCore.ParameterSet.Config as cms

process = cms.Process("HVTKMapsCreator")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.destinations.append('infos')
process.MessageLogger.infos = cms.untracked.PSet(
    placeholder = cms.untracked.bool(False),
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(10000)
    )
)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")


process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(77777),
                            numberEventsInRun = cms.untracked.uint32(1)
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("DQM.SiStripCommon.TkHistoMap_cff")
# load TrackerTopology (needed for TkDetMap and TkHistoMap)
process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.trackerTopology = cms.ESProducer("TrackerTopologyEP")

#This is where we configure the input and output files for the LV/HV TkMaps we want to create
#LV/HVStatusFile is a file that contains a list of all the 15148 Tracker DetIDs and for each of them a number 0=OFF 1=ON
#LV/HVTkMapName is the file name for the png image of the TkMap to be produced
#TODO: could add a parameter for the root file used to store the plot directly... currently root file is being overwritten
process.TkVoltageMapCreator = cms.EDAnalyzer('TkVoltageMapCreator',
	LVStatusFile = cms.string("LV_FROM_Thu_Aug__5_21_54_19_2010_TO_Thu_Aug__5_23_01_47_2010.log"),
        #/afs/cern.ch/user/g/gbenelli/public/LV_FROM_Thu_Aug__5_05_15_05_2010_TO_Thu_Aug__5_05_17_32_2010.log"),
	LVTkMapName = cms.string("LV_FROM_Thu_Aug__5_21_54_19_2010_TO_Thu_Aug__5_23_01_47_2010.png"),
	HVStatusFile = cms.string("HV_FROM_Thu_Aug__5_21_54_19_2010_TO_Thu_Aug__5_23_01_47_2010.log"),
        #/afs/cern.ch/user/g/gbenelli/public/HV_FROM_Thu_Aug__5_05_15_05_2010_TO_Thu_Aug__5_05_17_32_2010.log"),
	HVTkMapName = cms.string("HV_FROM_Thu_Aug__5_21_54_19_2010_TO_Thu_Aug__5_23_01_47_2010.png")
)

process.p0 = cms.Path(process.TkVoltageMapCreator)
