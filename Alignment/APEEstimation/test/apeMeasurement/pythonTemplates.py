###
# Python file templates for condor jobs
###

conditionsFileHeader="""
import FWCore.ParameterSet.Config as cms
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
def applyConditions(process):
    pass
"""

conditionsTemplate="""
    process.my{record}Conditions = poolDBESSource.clone(
    connect = cms.string('{source}'),
    toGet = cms.VPSet(cms.PSet(record = cms.string('{record}'),
                            tag = cms.string('{tag}')
                            )
                    )
    )
    process.prefer_my{record}Conditions = cms.ESPrefer("PoolDBESSource", "my{record}Conditions")
"""

# this is only used for MC samples as the file lists for data samples are created with an external script
fileListTemplate="""
import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles)

readFiles.extend( [
{files}
])


"""
-- dummy change --
