import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# Writes a SiPixelClusterShapeLimits payload (pixel cluster shape filter cut windows) from the ASCII files
# in RecoTracker/PixelLowPtUtilities/data.
# A payload has tables (one per file) and rules: a module uses the table of the first rule that matches it
# (subdet BPix/FPix and layerOrDisk, 0 = any); BPix and FPix each need a catch-all rule at the end.
# The scenarios reproduce the file choices of RecoTracker/PixelLowPtUtilities/python/ClusterShapeHitFilterESProducer_cfi.py
def table(name):
    return cms.PSet(name = cms.string(name), file = cms.FileInPath('RecoTracker/PixelLowPtUtilities/data/%s.par' % name))

def rule(subdet, table, layerOrDisk = 0):
    return cms.PSet(subdet = cms.string(subdet), layerOrDisk = cms.int32(layerOrDisk), table = cms.string(table))

scenarios = {
    # same file for all modules
    'phase0': dict(tables = [table('pixelShapePhase0')],
                   rules = [rule('BPix', 'pixelShapePhase0'),
                            rule('FPix', 'pixelShapePhase0')]),
    # BPix layer 1 has its own file
    'phase1': dict(tables = [table('pixelShapePhase1_noL1'), table('pixelShapePhase1_loose')],
                   rules = [rule('BPix', 'pixelShapePhase1_loose', layerOrDisk = 1),
                            rule('BPix', 'pixelShapePhase1_noL1'),
                            rule('FPix', 'pixelShapePhase1_noL1')]),
    # same file for all modules
    'phase2': dict(tables = [table('ITShapePhase2_all')],
                   rules = [rule('BPix', 'ITShapePhase2_all'),
                            rule('FPix', 'ITShapePhase2_all')]),
}

options = VarParsing.VarParsing()
options.register('scenario', 'phase1', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'one of: ' + ', '.join(scenarios))
options.register('outputDB', 'SiPixelClusterShapeLimits.db', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'output sqlite file')
options.register('outputTag', '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'output tag (default: SiPixelClusterShapeLimits_<scenario>_v1)')
options.register('firstRun', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'first run of validity')
options.parseArguments()

if options.scenario not in scenarios:
    raise ValueError('unknown scenario %s, choose one of: %s' % (options.scenario, ', '.join(scenarios)))
tag = options.outputTag or 'SiPixelClusterShapeLimits_%s_v1' % options.scenario

process = cms.Process("SiPixelClusterShapeLimitsWriter")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1), reportEvery = cms.untracked.int32(1)),
    SiPixelClusterShapeLimitsWriter = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.firstRun),
    lastValue = cms.uint64(options.firstRun),
    interval = cms.uint64(1)
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:' + options.outputDB

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelClusterShapeLimitsRcd'),
        tag = cms.string(tag)
    ))
)

from CondTools.SiPixel.siPixelClusterShapeLimitsWriter_cfi import siPixelClusterShapeLimitsWriter
process.writer = siPixelClusterShapeLimitsWriter.clone(
    tables = scenarios[options.scenario]['tables'],
    rules = scenarios[options.scenario]['rules'],
)

process.p = cms.Path(process.writer)
