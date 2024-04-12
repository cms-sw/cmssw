import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('python')
options.register('modules','Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt',mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells','Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells','Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.parseArguments()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#ESSources/Producers for the logical mapping
#indexers
process.load('Geometry.HGCalMapping.hgCalMappingESProducer_cfi')
process.hgCalMappingESProducer.modules = cms.FileInPath(options.modules)
process.hgCalMappingESProducer.si = cms.FileInPath(options.sicells)
process.hgCalMappingESProducer.sipm = cms.FileInPath(options.sipmcells)

#cells and modules info
process.load('Configuration.StandardSequences.Accelerators_cff')
process.hgCalMappingCellESProducer = cms.ESProducer('hgcal::HGCalMappingCellESProducer@alpaka',
                                                      filelist=cms.vstring(options.sicells,options.sipmcells),
                                                      cellindexer=cms.ESInputTag('') )
process.hgCalMappingModuleESProducer = cms.ESProducer('hgcal::HGCalMappingModuleESProducer@alpaka',
                                                      filename=cms.FileInPath(options.modules),
                                                      moduleindexer=cms.ESInputTag('') )

#tester
process.tester = cms.EDAnalyzer('HGCalMappingESSourceTester')

process.p = cms.Path(process.tester)
