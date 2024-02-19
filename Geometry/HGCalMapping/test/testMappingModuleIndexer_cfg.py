import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('python')
options.register('modules','Geometry/HGCalMapping/ModuleMaps/modulelocator_test.txt',mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells','Geometry/HGCalMapping/CellMaps/WaferCellMapTraces.txt',mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells','Geometry/HGCalMapping/CellMaps/channels_sipmontile.hgcal.txt',mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring("HGCalMapping*"),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG'),
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    )
)

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#ESSources/Producers for the logical mapping
process.load('Geometry.HGCalMapping.hgCalMappingIndexESSource_cfi')
process.hgCalMappingIndexESSource.modules = options.modules
process.hgCalMappingIndexESSource.si = options.sicells
process.hgCalMappingIndexESSource.sipm = options.sipmcells

process.load('Configuration.StandardSequences.Accelerators_cff')
#process.hgcalMappingModuleIndexerESRecord = cms.ESSource('EmptyESSource',
#                                                          recordName=cms.string('HGCalMappingModuleIndexerRcd'),
#                                                          iovIsRunNotTime=cms.bool(True),
#                                                          firstValid=cms.vuint32(1) )
#process.hgcalMappingModuleESRecord = cms.ESSource('EmptyESSource',
#                                                   recordName=cms.string('HGCalMappingModuleRcd'),
#                                                   iovIsRunNotTime=cms.bool(True),
#                                                   firstValid=cms.vuint32(1) )
process.hgCalMappingModuleESProducer = cms.ESProducer('hgcal::HGCalMappingModuleESProducer@alpaka',
                                                      filename=cms.string(options.modules),
                                                      moduleindexer=cms.ESInputTag('') )
process.hgCalMappingCellESProducer = cms.ESProducer('hgcal::HGCalMappingCellESProducer@alpaka',
                                                      filelist=cms.vstring(options.sicells,options.sipmcells),
                                                      cellindexer=cms.ESInputTag('') )

#tester
process.tester = cms.EDProducer('HGCalMappingESSourceTester@alpaka')

process.p = cms.Path(process.tester)
