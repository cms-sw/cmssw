import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.MuonSysAging.RPCChamberMasker_cfi import RPCChamberMasker
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

def appendRPCChamberMaskerAtReco(process):

    if phase2_muon.isChosen():
        appendRPCChamberMaskerBeforeRecHits(process)
    else :
        appendRPCChamberMaskerAtUnpacking(process)

    return process


# To be used for PhaseII
# no packin/unpacking is available for RE3/1 RE4/1
# must start from sim digis  
def appendRPCChamberMaskerBeforeRecHits(process):

    if hasattr(process,'rpcRecHits') :

        print "[appendRPCChamberMasker] : Found rpcRecHits, applying filter"

        process.rpcRecHits = process.rpcRecHits.clone()
        process.rpcAgedDigis = RPCChamberMasker.clone()
        process.rpcRecHits.rpcDigiLabel = cms.InputTag('rpcAgedDigis')

        process.filteredRpcDigiSequence = cms.Sequence(process.rpcAgedDigis \
                                                       + process.rpcRecHits)

        process.reconstruction.replace(process.rpcRecHits, \
                                       process.filteredRpcDigiSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.rpcAgedDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                rpcAgedDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

        return process


# To be used for PhaseII
# exploit unpacking of digis  
def appendRPCChamberMaskerAtUnpacking(process):

    if hasattr(process,'muonRPCDigis') :

        print "[appendRPCChamberMasker] : Found muonRPCDigis, applying filter"

        process.preRPCDigis = process.muonRPCDigis.clone()
        process.muonRPCDigis = RPCChamberMasker.clone()

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.muonRPCDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                muonRPCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

	process.filteredRPCDigiSequence = cms.Sequence(process.preRPCDigis \
                                                       + process.muonRPCDigis)
        process.RawToDigi.replace(process.muonRPCDigis, \
                                  process.filteredRPCDigiSequence)
    
    return process
