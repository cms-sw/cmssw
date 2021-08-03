import FWCore.ParameterSet.Config as cms

# Several parameters needed for HF simulation

HFLibraryFileBlock = cms.PSet(
        FileName        = cms.FileInPath('SimG4CMS/Calo/data/HFShowerLibrary_oldpmt_noatt_eta4_16en_v3.root'),
        BackProbability = cms.double(0.2),
        TreeEMID        = cms.string('emParticles'),
        TreeHadID       = cms.string('hadParticles'),
        ApplyFiducialCut= cms.bool(True),
        FileVersion     = cms.int32(0),
        Verbosity       = cms.untracked.bool(False),
        BranchPost      = cms.untracked.string(''),
        BranchEvt       = cms.untracked.string(''),
        BranchPre       = cms.untracked.string('')
)

HFShowerBlock = cms.PSet(
        ProbMax           = cms.double(1.0),
        CFibre            = cms.double(0.5),
        OnlyLong          = cms.bool(True),
        IgnoreTimeShift   = cms.bool(False)
)

##
## Change the HFShowerLibrary file from Run 2
##
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( HFLibraryFileBlock, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v4.root' )
run2_common.toModify( HFShowerBlock, ProbMax = 0.5 )

##
## Change for the new HFShowerLibrary file to be used for Run 3
##
from Configuration.Eras.Modifier_run3_HFSL_cff import run3_HFSL
run3_HFSL.toModify( HFLibraryFileBlock, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_run3_v5.root', FileVersion = 1 )
run3_HFSL.toModify( HFShowerBlock, IgnoreTimeShift = True )
