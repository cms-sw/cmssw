from FWCore.ParameterSet.Config import *

process = Process("test")

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
input_files = vstring('/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V3-v1/0007/123C78F5-9078-DE11-8BAD-001D09F23A61.root',
                      '/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V3-v1/0006/E6C7ED95-4878-DE11-B082-000423D98BE8.root'
                      );


process.source = Source("PoolSource",
    fileNames = untracked( input_files )
)

process.maxEvents = untracked.PSet( input = untracked.int32( 1000 ) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.output = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string('filteredSC.root'),
                                  SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p1')
                                  )


process.filterProbCluster = EDFilter("ProbClustersFilter",
                                     maxDistance = int32(100),
                                     maxGoodFraction = double(1.),
                                     reducedBarrelRecHitCollection = InputTag("reducedEcalRecHitsEB"),
                                     reducedEndcapRecHitCollection = InputTag("reducedEcalRecHitsEE"),
                                     barrelClusterCollection = InputTag("correctedHybridSuperClusters"),
                                     endcapClusterCollection = InputTag("correctedMulti5x5SuperClustersWithPreshower")
                                     
)

process.testEcalClusterSeverityAlgo = EDAnalyzer("testEcalClusterSeverityAlgo",
    reducedBarrelRecHitCollection = InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = InputTag("reducedEcalRecHitsEE"),
    barrelClusterCollection = InputTag("correctedHybridSuperClusters"),
    endcapClusterCollection = InputTag("correctedMulti5x5SuperClustersWithPreshower")
)

process.p1 = Path( process.filterProbCluster)
process.e = cms.EndPath( process.output )
                   
