import FWCore.ParameterSet.Config as cms

# Customization for running on MC
#   - Call from cmsDriver.py with: `--customise DPGAnalysis/HcalNanoAOD/customiseHcalMC_cff.customiseHcalMC`
def customiseHcalMC(process):
  # Point to appropriate MC digi collections
  process.hcalDigiSortedTable.tagQIE11 = cms.untracked.InputTag("simHcalUnsuppressedDigis", "HBHEQIE11DigiCollection")
  process.hcalDigiSortedTable.tagQIE10 = cms.untracked.InputTag("simHcalUnsuppressedDigis","HFQIE10DigiCollection")
  process.hcalDigiSortedTable.tagHO = cms.untracked.InputTag("simHcalUnsuppressedDigis")

  # Use the appropriate number of samples for digis in MC
  process.hcalDigiSortedTable.nTS_HB = cms.untracked.uint32(10)
  process.hcalDigiSortedTable.nTS_HE = cms.untracked.uint32(10)

  return process
