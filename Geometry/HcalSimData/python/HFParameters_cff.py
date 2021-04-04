import FWCore.ParameterSet.Config as cms

# Several parameters needed for HF simulation

HFLibraryFileBlock = cms.PSet(
        FileName        = cms.FileInPath('SimG4CMS/Calo/data/HFShowerLibrary_oldpmt_noatt_eta4_16en_v3.root'),
        BackProbability = cms.double(0.2),
        TreeEMID        = cms.string('emParticles'),
        TreeHadID       = cms.string('hadParticles'),
        ApplyFiducialCut= cms.bool(True),
        Verbosity       = cms.untracked.bool(False),
        BranchPost      = cms.untracked.string(''),
        BranchEvt       = cms.untracked.string(''),
        BranchPre       = cms.untracked.string('')
)

HFShowerBlock = cms.PSet(
        ProbMax           = cms.double(1.0),
        CFibre            = cms.double(0.5),
        OnlyLong          = cms.bool(True)
)

##
## Change the HFShowerLibrary file from Run 2
##
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( HFLibraryFileBlock, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v4.root' )
run2_common.toModify( HFShowerBlock, ProbMax = 0.5)
