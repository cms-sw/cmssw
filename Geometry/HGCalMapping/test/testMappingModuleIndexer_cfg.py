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

#electronics mapping
from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
process = customise_hgcalmapper(process,
                                modules=options.modules,
                                sicells=options.sicells,
                                sipmcells=options.sipmcells)

#Geometry
process.load('Configuration.Geometry.GeometryExtended2026D99Reco_cff')

#tester
process.tester = cms.EDAnalyzer('HGCalMappingESSourceTester')

process.p = cms.Path(process.tester)
