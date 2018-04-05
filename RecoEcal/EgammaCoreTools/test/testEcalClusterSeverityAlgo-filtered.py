from FWCore.ParameterSet.Config import *

process = Process("analysis")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

from CondCore.DBCommon.CondDBSetup_cfi import *
process.ecalConditions = cms.ESSource("PoolDBESSource",
                                      CondDBSetup,
                                      siteLocalConfig = cms.untracked.bool(True),
                                      toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalChannelStatusRcd'),
    tag = cms.string('EcalChannelStatus_may2009_mc')
    )
                                                        ),
                                      
                                      messagelevel = cms.untracked.uint32(0),
                                      timetype = cms.string('runnumber'),
                                      connect = cms.string('frontier://FrontierProd/CMS_COND_31X_ECAL'), ##cms_conditions_data/CMS_COND_ECAL"
                                      
                                      authenticationMethod = cms.untracked.uint32(1)
                                      )

# process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# process.GlobalTag.globaltag = 'MC_31X_V6::All'

#input_files.append( "/store/relval/2008/4/17/RelVal-RelValTTbar-1208465820/0000/0ABDA540-EE0C-DD11-BA9F-000423D94990.root" )
input_files = vstring('rfio:///castor/cern.ch/user/m/meridian/ZeeFiltProbSC31X/meridian/Zee/ZeeFiltProbSC/0330624041aaf504f2f5a3cbc2ef3a16/filteredSC_30.root'

#file:///tmp/meridian/filteredSC.root',
#                      'file:///tmp/meridian/filteredSC_39.root'
                      );


process.source = Source("PoolSource",
    fileNames = untracked( input_files )
)

process.maxEvents = untracked.PSet( input = untracked.int32( -1 ) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


process.testEcalClusterSeverityAlgo = EDAnalyzer("testEcalClusterSeverityAlgo",
                                                 reducedBarrelRecHitCollection = InputTag("reducedEcalRecHitsEB"),
                                                 reducedEndcapRecHitCollection = InputTag("reducedEcalRecHitsEE"),
                                                 barrelClusterCollection = InputTag("correctedHybridSuperClusters"),
                                                 endcapClusterCollection = InputTag("correctedMulti5x5SuperClustersWithPreshower"),
                                                 mcTruthCollection = cms.InputTag("VtxSmeared"),
                                                 outputFile = cms.string('filteredSCtree.root')
                                                 )

process.p1 = Path( process.testEcalClusterSeverityAlgo )
                   
