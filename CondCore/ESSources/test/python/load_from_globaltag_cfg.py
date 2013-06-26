import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("EmptyIOVSource",
                                lastValue = cms.uint64(3),
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(1),
                                interval = cms.uint64(1)
                            )

# process.load("Configuration.StandardSequences.Services_cff")

from CondCore.ESSources.GlobalTag import GlobalTag

# Prepare the list of globalTags
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
GT1 = GlobalTag("PRA_61_V1::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V1.db")
GT2 = GlobalTag("PRB_61_V2::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V2.db")
GT3 = GlobalTag("PRE_61_V3::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V3.db")
GT4 = GlobalTag("PRE_61_V4::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V4.db")

# Keep in mind that operator + has precedence
# globalTag = GT3 + GT4

# globalTag = GlobalTag("MAINGT", "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG|sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/AN_V4.db")
globalTag = GlobalTag("BASEGT", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/BASE1_V1.db|sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/BASE2_V1.db")

# globalTagList.append(GlobalTag("PRE_61_V4::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V4.db", "", ""))
# makeGlobalTag(globalTagList)
process.GlobalTag.connect = cms.string(globalTag.connect())
process.GlobalTag.globaltag = globalTag.gt()

print "Final connection string =", process.GlobalTag.connect
print "Final globalTag =", process.GlobalTag.globaltag


# process.GlobalTag.connect = cms.string(connectString)

process.path = cms.Path()
