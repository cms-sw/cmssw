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
        EqualizeTimeShift   = cms.bool(False)
)

##
## Change the HFShowerLibrary file for Run2
##
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.ProcessModifiers.applyHFLibraryFix_cff import applyHFLibraryFix
#
#--- Default: to keep using the library with a problem
(~applyHFLibraryFix & run2_common).toModify( HFLibraryFileBlock, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v4.root' )
#
#--- Alternative: to use Run3 library with applyHFLibraryFix modifier
(applyHFLibraryFix & run2_common).toModify( HFLibraryFileBlock, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_run3_v7.root', FileVersion = 3 )
(applyHFLibraryFix & run2_common).toModify( HFShowerBlock, EqualizeTimeShift = True )
#
run2_common.toModify( HFShowerBlock, ProbMax = 0.5 )

##
## Change for the latest HFShowerLibrary file for Run 3
##
from Configuration.Eras.Modifier_run3_HFSL_cff import run3_HFSL
run3_HFSL.toModify( HFLibraryFileBlock, FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_run3_v7.root', FileVersion = 3 )
run3_HFSL.toModify( HFShowerBlock, EqualizeTimeShift = True )
