import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('-modules','--modules',type=str,
                    default='Geometry/HGCalMapping/data/ModuleMaps/modulelocator_trigger_test.txt',
                    help='Path to module mapper. Absolute, or relative to CMSSW src directory.'
                    )

parser.add_argument('-sicells','--sicells',type=str,
                    default='Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',
                    help='Path to Si cell mapper. Absolute, or relative to CMSSW src directory.'
                    )

parser.add_argument('-sipmcells','--sipmcells',type=str,
                    default='Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',
                    help='Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory.'
                    )
options = parser.parse_args()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# electronics mapping
from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
process = customise_hgcalmapper(process,
                                modules=options.modules,
                                sicells=options.sicells,
                                sipmcells=options.sipmcells)

# Geometry
process.load('Configuration.Geometry.GeometryExtended2025Reco_cff')

# tester
process.tester = cms.EDAnalyzer('HGCalMappingTriggerESSourceTester')

process.p = cms.Path(process.tester)
