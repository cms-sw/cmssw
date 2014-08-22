import FWCore.ParameterSet.Config as cms

# Give the process a name
process = cms.Process("PickEvent")

# Tell the process which files to use as the sourdce
process.source = cms.Source ("PoolSource",
          fileNames = cms.untracked.vstring (
'/store/relval/CMSSW_5_2_7-START52_V10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/v1/00000/A694FA3D-3906-E211-8AD2-0018F3D0962E.root'
)
)

# tell the process to only run over 100 events (-1 would mean run over
#  everything
process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32 (20)

)

# Tell the process what filename to use to save the output
process.Out = cms.OutputModule("PoolOutputModule",
         fileName = cms.untracked.string ("MyOutputFile.root")
)

# make sure everything is hooked up
process.end = cms.EndPath(process.Out)

