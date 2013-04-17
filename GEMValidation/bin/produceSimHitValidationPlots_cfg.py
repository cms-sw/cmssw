import os
import FWCore.ParameterSet.Config as cms

process = cms.PSet()
process.verbose = cms.uint32( 1 )
process.inputFile = cms.string('gem_sh_ana.MuonGunPt40Evt10k-8Part.root')
process.targetDir = cms.string('/afs/cern.ch/user/d/dildick/www/GEMValidation/Sim/8Part/MuonGunPt40Evt10k/')
process.printFile = cms.bool(False)
process.muonSelection = cms.vstring('abs(particleType)==13','abs(particleType)!=13','')
process.titlePrefix = cms.vstring('Muon','Non muon','All')
process.histSuffix = cms.vstring('_muon','_nonmuon','_all')
process.ext = cms.string(".png")
process.nregion = cms.int32( 2 )
process.nlayer = cms.int32( 2 )
process.npart = cms.int32( 8 )
