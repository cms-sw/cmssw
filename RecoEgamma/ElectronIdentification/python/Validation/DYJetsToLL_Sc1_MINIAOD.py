import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [ 'root://eoscms.cern.ch//store/group/phys_egamma/ElectronValidationArchives/MiniAOD_IDValidation/miniAOD-prod_PAT_zee14_50ns.root'
        ] );


secFiles.extend( [
               ] )

