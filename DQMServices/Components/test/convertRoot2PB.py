import sys
import FWCore.ParameterSet.Config as cms

print len(sys.argv)
if len(sys.argv) < 3:
    print 'Error: expecting filename to be converted'
    sys.exit(1)
filename = sys.argv[2].split('.root')[0]

process = cms.Process("Converter")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load("DQMServices.Core.DQM_cfg")
process.DQMStore.verbose = 2

process.load("DQMServices.Components.DQMFileReader_cfi")
process.dqmFileReader.FileNames = cms.untracked.vstring ("%s.root" % filename)

process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.convention = cms.untracked.string("PB")
process.dqmSaver.forceRunNumber = cms.untracked.int32(60605)
filename = filename[1:].replace('/', '_')
filename = filename[1:].replace('.', '_')
process.dqmSaver.workflow = cms.untracked.string("/Conversion/Workflow/%s" % filename)

process.p = cms.Path(process.dqmFileReader
                     *process.dqmSaver)


# Local Variables:
# show-trailing-whitespace: t
# truncate-lines: t
# End:
