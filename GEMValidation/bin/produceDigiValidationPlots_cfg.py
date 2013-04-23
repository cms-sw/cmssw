import os
import FWCore.ParameterSet.Config as cms

process = cms.PSet()
process.verbose = cms.uint32( 1 )
process.inputFile = cms.string('../test/gem_digi_ana.MuonGunPt40Evt10k-8Part.root')
process.targetDir = cms.string('/afs/cern.ch/user/d/dildick/www/GEMValidation/DigiL1CSC/8Part/MuonGunPt40Evt10kPU0/')
process.ext = cms.string(".png")
process.npads = cms.uint32( 24 )
