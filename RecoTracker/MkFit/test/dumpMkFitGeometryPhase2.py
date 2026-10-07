import FWCore.ParameterSet.Config as cms

# from Configuration.ProcessModifiers.trackingMkFit_cff import trackingMkFit
from Configuration.ProcessModifiers.trackingMkFitCommon_cff import trackingMkFitCommon
trackingMkFit = cms.ModifierChain(trackingMkFitCommon)

###################################################################
# Set default phase-2 settings
###################################################################
import os
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
# ONE variable selects the geometry, the era, the conditions, the stamp written
# into the binary and the output file name, so none of them can disagree with
# what was actually loaded. The release default is not always what a sample was
# made with -- CMSSW_20_1_0_pre2 defaults to Run4D127 while the D121 relvals are
# still in use -- so it is overridable:
#     MKFIT_GEOM_VERSION=Run4D121 cmsRun dumpMkFitGeometryPhase2.py
GEOM_VERSION = os.environ.get('MKFIT_GEOM_VERSION', _settings.DEFAULT_VERSION)
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(GEOM_VERSION)

# No era in Fireworks/Geom reco dumper
process = cms.Process('DUMP', _PH2_ERA, trackingMkFit)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended%sReco_cff' % GEOM_VERSION)
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')

# In Fireworks/Geom reco dumper:
# from Configuration.AlCa.autoCond import autoCond
# process.GlobalTag.globaltag = autoCond['phase2_realistic']

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.MkFitGeometryESProducer = dict(limit=-1)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1


process.add_(cms.ESProducer("MkFitGeometryESProducer"))

defaultOutputFileName = "phase2-trackerinfo-%s.bin" % GEOM_VERSION

# level: 0 - no printout; 1 - print layers, 2 - print shapes and modules
# outputFileName: binary dump file; no dump if empty string
process.dump = cms.EDAnalyzer("DumpMkFitGeometry",
                              level = cms.untracked.int32(1),
                              outputFileName = cms.untracked.string(defaultOutputFileName),
                              geometryVersion = cms.untracked.string(GEOM_VERSION)
                              )

print("Requesting MkFit geometry dump into file:", defaultOutputFileName,
      "  geometry version:", GEOM_VERSION, "\n");
process.p = cms.Path(process.dump)
