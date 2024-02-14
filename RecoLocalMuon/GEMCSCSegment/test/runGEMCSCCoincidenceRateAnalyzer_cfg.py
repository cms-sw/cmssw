import sys
import argparse
import importlib
from pathlib import Path
import FWCore.ParameterSet.Config as cms

print(f'{sys.argv=}')
# NTOE when running cmsRun, __file__ is not defined

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--era', type=str, default='Phase2C17I13M9', help='era')
parser.add_argument('-t', '--global-tag', type=str, default='auto:phase2_realistic_T21', help='global tag')
parser.add_argument('-g', '--geometry', type=str, default='GeometryExtended2026D88Reco', help='geometry')
parser.add_argument('-m', '--max-events', type=int, default=-1, help='max events')

default_data_dir = Path('/eos/cms/store/relval/CMSSW_12_6_0_pre2/RelValSingleMuPt1000/GEN-SIM-RECO/125X_mcRun4_realistic_v2_2026D88noPU-v1/')
default_input_files = ['file:' + str(each) for each in default_data_dir.glob('**/*.root')]
parser.add_argument('-i', '--input-files', type=str, nargs='+', default=default_input_files, help='input files')
parser.add_argument('-o', '--output-file', type=str, default='output.root', help='output file')
args = parser.parse_args()

for key, value in vars(args).items():
    print(f'{key}={value}')

era_module = importlib.import_module(f'Configuration.Eras.Era_{args.era}_cff')
era = getattr(era_module, args.era)

process = cms.Process('GEMCSC', era)

process = cms.Process('TestGEMCSCSegment')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(f'Configuration.Geometry.{args.geometry}_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoLocalMuon.GEMCSCSegment.gemcscSegments_cfi')
process.load('RecoLocalMuon.GEMCSCSegment.gemcscCoincidenceRateAnalyzer_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, args.global_tag, '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(args.max_events)
)

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(*args.input_files)
)

process.gemcscSegments.enableME21GE21 = True

process.gemcscSegmentSeq = cms.Sequence(
    process.gemcscSegments *
    process.gemcscCoincidenceRateAnalyzer
)

process.TFileService = cms.Service('TFileService',
    fileName = cms.string(args.output_file)
)

process.p = cms.Path(process.gemcscSegmentSeq)
