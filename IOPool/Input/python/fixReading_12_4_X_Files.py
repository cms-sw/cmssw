import FWCore.ParameterSet.Config as cms

# ROOT version used in 12_4_X (2022 data taking), and in 13_0_X up to
# 13_0_3 had a bug where TStreamerInfo was missing in some cases. This
# customize function adds a Service to include the TStreamerInfo for
# the affected classes so that the old files can be read now that some
# of the affected data format classes have evolved.
def fixReading_12_4_X_Files(process):
    process.add_(cms.Service("FixMissingStreamerInfos",
        fileInPath = cms.untracked.FileInPath("IOPool/Input/data/fileContainingStreamerInfos_13_0_0.root")
    ))
    return process
