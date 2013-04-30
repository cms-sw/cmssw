import os
import FWCore.ParameterSet.Config as cms

process = cms.PSet()
process.verbose = cms.uint32( 1 )
process.inputFile = cms.string('/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/gem_localrec_ana.root')
process.targetDir = cms.string('/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/new_dir/')
process.ext = cms.string(".png")
process.nregion = cms.int32( 2 )
process.nlayer = cms.int32( 2 )
process.npart = cms.int32( 8 )
