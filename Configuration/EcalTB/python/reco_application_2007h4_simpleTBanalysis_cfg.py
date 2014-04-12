import FWCore.ParameterSet.Config as cms

process = cms.Process("simpleAnalysis")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#Take 2007H4TB geometry
#process.load("Configuration.EcalTB.2007TBH4GeometryXML_cfi");
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi");
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
#process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.CaloGeometryBuilder.SelectedCalos = ['EcalEndcap']

process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(int(10000)))
process.source = cms.Source("PoolSource",
    #	untracked vstring fileNames = {'rfio:/castor/cern.ch/cms/archive/ecal/h4tb.pool-cmssw-SM12/h4b.00015217.A.0.0.root'}
    fileNames = cms.untracked.vstring('file:hits.root')
)

process.simple2007H4TBanalysis = cms.EDAnalyzer("EcalSimple2007H4TBAnalyzer",
    rootfile = cms.untracked.string('ecalSimple2007H4TBAnalysis.root'),
    eventHeaderProducer = cms.string('ecalTBunpack'),
    hitProducer = cms.string('ecal2007TBWeightUncalibRecHit'),
    digiCollection = cms.string('eeDigis'),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    digiProducer = cms.string('ecalTBunpack'),
    hitCollection = cms.string('EcalUncalibRecHitsEE'),
    hodoRecInfoProducer = cms.string('ecal2006TBHodoscopeReconstructor'),
    eventHeaderCollection = cms.string(''),
    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    tdcRecInfoProducer = cms.string('ecal2007H4TBTDCReconstructor')
)

process.p = cms.Path(process.simple2007H4TBanalysis)

