import FWCore.ParameterSet.Config as cms
import string

process = cms.Process('RECODQM')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring( *(
    # '''/store/data/Commissioning2018/ZeroBias/RAW/v1/000/314/816/00000/0062E91C-2245-E811-8DCB-FA163EE59A93.root',
    # '''/store/data/Commissioning2018/ZeroBias/RAW/v1/000/314/816/00000/007EABA4-5745-E811-8573-FA163EB3E1C0.root',
# '''/store/data/Commissioning2018/MinimumBias/RAW/v1/000/314/276/00000/B45F5174-6040-E811-BCDA-FA163EB38F53.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/04F64F14-A53F-E811-A60D-FA163ED486E3.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/14CD5D1F-A53F-E811-93C2-02163E013A66.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/28B61E20-A53F-E811-991E-FA163E02DCE9.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/30A5B519-A53F-E811-97FD-FA163E908B47.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/40A1AC78-A73F-E811-A05F-FA163E756520.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/6A2F43D9-A73F-E811-BC51-FA163E19119E.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/82512A24-A53F-E811-8C5D-FA163E8B5DA0.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/A6F81F18-A53F-E811-AAF3-FA163E7947B1.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/AEA372A1-A73F-E811-B554-02163E01A087.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/B808481E-A53F-E811-9E76-FA163E2721BD.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/B8FE1014-A53F-E811-A96F-FA163EE087A1.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/BC28D611-A53F-E811-B0FA-02163E019F09.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/BC6EC94A-A63F-E811-8439-02163E013C9E.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/CA87631C-A53F-E811-9259-FA163EAFECF2.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/CE9D570E-A53F-E811-A8BE-FA163E7CFF74.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/DCF672B7-A63F-E811-9255-02163E00B5C7.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/DE088F24-A53F-E811-B4D5-02163E01A154.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/E0052C15-A53F-E811-AC12-02163E017689.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/F4731F0E-A53F-E811-8240-FA163E8DA7AA.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/F4E1BC16-A53F-E811-BE65-FA163EA4AD14.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/046125EF-A73F-E811-BCB3-FA163EBC0A12.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/04E62BDE-A53F-E811-87A8-02163E00BBEA.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/0E092A13-A53F-E811-B15F-FA163E80B14B.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/1A460F21-A53F-E811-B0C9-02163E019ED6.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/22E0E0B6-A53F-E811-BF11-02163E00C372.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/286A20DE-A43F-E811-B37A-FA163E366887.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/2A555F8A-A43F-E811-BB9B-FA163E756520.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/2CAA1FDC-A43F-E811-B081-FA163EC67281.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/4E38F8E4-A43F-E811-9CC7-FA163E836BF1.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/50A86073-A53F-E811-B671-02163E015019.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/6CB0F276-A53F-E811-AC75-FA163E6D9DFD.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/7257A002-A63F-E811-8667-FA163EB6EC5B.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/841EDF91-A43F-E811-9B44-02163E015F06.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/90322CC5-A43F-E811-8B94-FA163E20FAF0.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/D213809E-A43F-E811-92E3-FA163E5A18F0.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/E89674AB-A53F-E811-98E8-FA163E19119E.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/EA96BCD5-A43F-E811-8B1B-FA163EF3CCDB.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/EAE58A83-A73F-E811-A3E7-02163E00BB45.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/F0CFFDCF-A73F-E811-BE32-FA163EFC15F4.root',
'/store/data/Commissioning2018/ZeroBias2/RAW/v1/000/314/277/00000/FA4BA7CB-A43F-E811-9C57-FA163E57FDA9.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/04B22B0E-A53F-E811-88AE-FA163E6FAA82.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/089D8411-A53F-E811-99CD-FA163ECC03D8.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/161AA678-A53F-E811-9288-FA163EFCF031.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/28678F89-A63F-E811-B503-02163E015240.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/3CE2C015-A53F-E811-9EDF-FA163E821418.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/462A6F15-A53F-E811-A79F-FA163E6C430C.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/4A57E315-A53F-E811-A9F2-FA163E554F4D.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/505B452C-A53F-E811-9531-FA163EC4D207.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/6CD83C6E-A53F-E811-8C10-FA163EE103BD.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/7859BD4C-A53F-E811-9965-FA163EF2704B.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/80F08BA5-A73F-E811-8296-FA163EC4EAD7.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/86181F2D-A53F-E811-B7CE-FA163EC63B78.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/92BFB123-A53F-E811-88C8-02163E00C363.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/98746FBE-A53F-E811-B353-FA163E33C0D2.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/9C253017-A53F-E811-913E-FA163E355512.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/AEBD4456-A73F-E811-B375-02163E013A66.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/D84FE415-A53F-E811-A364-FA163EED3BED.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/DA144F11-A53F-E811-A847-FA163E398BE5.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/DE279FBB-A73F-E811-A6BD-FA163E4F72F2.root',
'/store/data/Commissioning2018/ZeroBias3/RAW/v1/000/314/277/00000/F2372C67-A53F-E811-A3EB-FA163EC4EAD7.root',
    )
    )
)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# load local geometry to avoid GT
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff')

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
    process.ctppsRawToDigi *
    process.recoCTPPS *
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
