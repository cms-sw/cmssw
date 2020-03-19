import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond
 
process = cms.Process("HGCalParametersTest")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = autoCond['run1_mc']
 
process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PHGCalParametersRcd'),
                                              tag = cms.string('HGCALParameters_Geometry_Test01'),
                                              connect = cms.string("sqlite_file:./myfile.db")
                                              )
                                     )
 
process.source = cms.Source("EmptySource")
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
 
process.test = cms.EDAnalyzer("HGCalParametersAnalyzer")
 
process.p1 = cms.Path(process.test)
