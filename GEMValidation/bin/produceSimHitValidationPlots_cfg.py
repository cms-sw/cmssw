import os
import FWCore.ParameterSet.Config as cms

process = cms.PSet()
process.verbose = cms.uint32( 1 )
process.inputFile = cms.string('/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_0_1_PostLS1v2_patch2/src/RPCGEM/GEMValidation/test/gem_sh_ana.test.root')
process.targetDir = cms.string('/afs/cern.ch/user/d/dildick/www/GEMValidation/Sim/8Part/MuonGunPt40Evt10k/')
#process.targetDir = cms.string('./')
process.ext = cms.string(".png")
process.nregion = cms.int32( 2 )
process.nlayer = cms.int32( 2 )
process.npart = cms.int32( 8 )
