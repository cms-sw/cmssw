import FWCore.ParameterSet.Config as cms

# output block for alcastream laserAlignmentT0Producer
# output module 
outLaserAlignmentT0Producer = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_laserAlignmentT0Producer_*_*',
        'keep SiStripEventSummary_siStripDigis_*_*')
)

