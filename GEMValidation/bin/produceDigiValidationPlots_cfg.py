import os
import FWCore.ParameterSet.Config as cms

process = cms.PSet()
process.inputFile = cms.string('/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/gem_digi_ana.root')
process.targetDir = cms.string('/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/new_dir/')
process.ext = cms.string(".png")
process.npads = cms.uint32( 96 )
