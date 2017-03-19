import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.CSCChamberMasker_cfi import CSCChamberMasker

def appendCSCChamberMaskerAtUnpacking(process):

    if hasattr(process,'muonCSCDigis') :
        
        print "[appendChamberMasker] : Found muonCSCDigis, applying filter"

        process.preCSCDigis = process.muonCSCDigis.clone()
        process.muonCSCDigis = CSCChamberMasker.clone()

        # process.muonCSCDigis.digiTag = cms.InputTag('preCSCDigis') 

        process.muonCSCDigis.stripDigiTag = cms.InputTag("preCSCDigis", "MuonCSCStripDigi")
        process.muonCSCDigis.wireDigiTag = cms.InputTag("preCSCDigis", "MuonCSCWireDigi") 
        process.muonCSCDigis.comparatorDigiTag = cms.InputTag("preCSCDigis", "MuonCSCComparatorDigi")
        process.muonCSCDigis.rpcDigiTag = cms.InputTag("preCSCDigis", "MuonCSCRPCDigi") 
        process.muonCSCDigis.alctDigiTag = cms.InputTag("preCSCDigis", "MuonCSCALCTDigi") 
        process.muonCSCDigis.clctDigiTag = cms.InputTag("preCSCDigis", "MuonCSCCLCTDigi") 

        process.filteredDigiSequence = cms.Sequence(process.preCSCDigis + process.muonCSCDigis)
        process.RawToDigi.replace(process.muonCSCDigis, process.filteredDigiSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.muonCSCDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                muonCSCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

    return process

def appendCSCChamberMaskerAtHLT(process):

    if hasattr(process,'hltMuonCSCDigis') :

        print "[appendChamberMasker] : Found hltMuonCSCDigis, applying filter"

        process.preHltCSCDigis = process.hltMuonCSCDigis.clone()
        process.hltMuonCSCDigis = CSCChamberMasker.clone()

        # process.hltMuonCSCDigis.stripDigiTag = "preHltCSCStripDigis"
        # process.hltMuonCSCDigis.wireDigiTag = "preHltCSCWireDigis"
        # process.hltMuonCSCDigis.comparatorDigiTag = "preHltCSCComparatorDigis"
        # process.hltMuonCSCDigis.rpcDigiTag = "preHltCSCRPCDigis"
        # process.hltMuonCSCDigis.alctDigiTag = "preHltCSCALCTDigis"
        # process.hltMuonCSCDigis.clctDigiTag = "preHltCSCCLCTDigis"

        process.hltMuonCSCDigis.stripDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCStripDigi")
        process.hltMuonCSCDigis.wireDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCWireDigi") 
        process.hltMuonCSCDigis.comparatorDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCComparatorDigi")
        process.hltMuonCSCDigis.rpcDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCRPCDigi") 
        process.hltMuonCSCDigis.alctDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCALCTDigi") 
        process.hltMuonCSCDigis.clctDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCCLCTDigi") 


        process.filteredHltCSCDigiSequence = cms.Sequence(process.preHltCSCDigis + process.hltMuonCSCDigis)
        process.HLTMuonLocalRecoSequence.replace(process.hltMuonCSCDigis, process.filteredHltCSCDigiSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.hltMuonCSCDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                hltMuonCSCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process

