import FWCore.ParameterSet.Config as cms

process = cms.Process("LaserDQM")
#    service = MonitorDaemon
#    {
#	# if true, will automatically start DQM thread in background
#	untracked bool AutoInstantiate=false	
#	# if >=0, upon a connection problem, the source will automatically 
#	# attempt to reconnect with a time delay (secs) specified here 
#	# (default: 5)
#	untracked int32 reconnect_delay = 5
#	# collector hostname; examples: localhost(default),lxcmse2.cern.ch, etc
#	untracked string DestinationAddress = "localhost"
#	# port for communicating with collector (default: 9090)
#	untracked int32 SendPort = 9090
#	# monitoring period in ms (i.e. how often monitoring elements 
#	# are shipped to the collector; default: 1000)
#	untracked int32 UpdateDelay = 1000
#	# name of DQM source (default: DQMSource)
#	untracked string NameAsSource = "FU0"	
#    }
# Ideal geometry producer
process.load("Geometry.TrackerRecoData.trackerRecoGeometryXML_cfi")

# Interface to ideal geometry producer
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# Tracker Geometry
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Alignment.LaserDQM.LaserDQM_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('simevent.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.p1 = cms.Path(process.mon)

