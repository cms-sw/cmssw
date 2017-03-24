import FWCore.ParameterSet.Config as cms
import string

process = cms.Process('RECODQM')

process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(-1)
)

process.verbosity = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"


#process.source = cms.Source('PoolSource',
    #fileNames = cms.untracked.vstring(
        #'root://eoscms.cern.ch:1094//eos/totem/data/ctpps/run284036.root',
    #),
#)

#process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(100000)
#)

# diamonds mapping
process.totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TimingDiamond"),
  configuration = cms.VPSet(
    # before diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 283819:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # after diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("283820:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    )
  )
)
    
    
## raw data source
import glob
import os
#filepath='/afs/cern.ch/user/n/nminafra/Work/CMSSW/setesami/CMSSW_8_1_0_pre8/src/dqm_diamonds/outputs/'
filepath='/afs/cern.ch/user/n/nminafra/Eos/cms/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/284/042/00000/'
filenames_root= sorted(glob.glob(filepath+'*.root'))
for i,filename_root in enumerate(filenames_root):
    filenames_root[i] ='root://' + filename_root
    #print filename_root

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 
    #'/store/data/Run2016H/DoubleEG/RAW/v1/000/283/876/00000/988E223A-719B-E611-BFF4-02163E0142EA.root',
'/store/data/Run2016H/HLTPhysics/RAW/v1/000/283/885/00000/008F3821-3C9D-E611-BD53-FA163E866791.root',
    #'/store/data/Run2016H/DoubleEG/RAW/v1/000/284/006/00000/621A743E-C29E-E611-92C6-02163E011DFA.root',
    #'/store/data/Run2016H/DoubleEG/RAW/v1/000/284/040/00000/4C7689B5-5D9F-E611-9885-02163E011D42.root',
    #
    #'/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/284/043/00000/FEF4F3A0-CB9B-E611-AC16-02163E0142ED.root'
    #filenames_root 
    #'root://eostotem.cern.ch//eos/totem/data/ctpps/run283896.root'
    )
    #fileNames = cms.untracked.vstring( 'root://eostotem.cern.ch//eos/totem/data/ctpps/run282007.root' )
)


# from lumi to lumi
start_lumi=1;
stop_lumi=800;
filename_beg='/store/t0streamer/Data/Express/000/284/036/run284036_ls';
filename_end='_streamExpress_StorageManager.dat';
filenames_array = [];
for i in range(start_lumi,stop_lumi):
  filenames_array.append(filename_beg + str(i).zfill(4) + filename_end);
#for filename in filenames_array:
  #print filename



# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi')

# local tracks fitter
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTracks_cfi')
process.ctppsDiamondLocalTracks.trackingAlgorithmParams.threshold = cms.double(1.5)
process.ctppsDiamondLocalTracks.trackingAlgorithmParams.sigma = cms.double(0.25)
process.ctppsDiamondLocalTracks.trackingAlgorithmParams.resolution = cms.double(0.025) # in mm
process.ctppsDiamondLocalTracks.trackingAlgorithmParams.pixel_efficiency_function = cms.string("(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4")


# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.ctppsDiamondDQMSource.excludeMultipleHits = cms.bool(True);

process.path = cms.Path(
  process.ctppsRawToDigi *
  process.recoCTPPS *
  process.ctppsDiamondRawToDigi *
  process.ctppsDiamondRecHits *
  process.ctppsDiamondLocalTracks *
  process.ctppsDQM 
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)

## output configuration
#process.output = cms.OutputModule("PoolOutputModule",
  #fileName = cms.untracked.string("file:./reco_digi.root"),
  #outputCommands = cms.untracked.vstring(
    #'drop *',
    #'keep *_*RawToDigi_*_*',
  #)
#)

#process.outpath = cms.EndPath(process.output)


