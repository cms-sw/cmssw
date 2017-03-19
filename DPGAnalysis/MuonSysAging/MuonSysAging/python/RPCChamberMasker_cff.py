import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.RPCChamberMasker_cfi import RPCChamberMasker

#def appendChamberMaskerAtUnpacking(process, doDigis, doTrigger, chambRegEx):
def appendRPCChamberMaskerAtUnpacking(process, doDigis, maskedRPCs):

    if doDigis and hasattr(process,'muonRPCDigis') :

        print "[appendChamberMasker] : Found muonRPCDigis, applying filter"

        process.preRPCDigis = process.muonRPCDigis.clone()
        process.muonRPCDigis = RPCChamberMasker.clone()
        if len(maskedRPCs) > 0 :
		process.muonRPCDigis.maskedRPCIDs = maskedRPCs
	process.filteredRPCDigiSequence = cms.Sequence(process.preRPCDigis + process.muonRPCDigis)
#        process.RawToDigi.replace(process.muonRPCDigis, process.filteredDigiSequence)
        process.RawToDigi.replace(process.muonRPCDigis, process.filteredRPCDigiSequence)
    
    return process

def appendRPCChamberMaskerAtUnpacking2(process, doDigis, maskedRPCs):
  if doDigis and hasattr(process,'muonRPCDigis') :
    print "[appendChamberMasker] : Found muonRPCDigis, applying filter"
    process.simMuonRPCDigis = RPCChamberMasker.clone()
    if len(maskedRPCs) > 0 :
        process.simMuonRPCDigis.maskedRPCIDs = maskedRPCs
        #process.simMuonRPCDigis.digiTag = "preSimMuonRPCDigis"
        process.simMuonRPCDigis.digiTag =  cms.InputTag("simMuonRPCDigis", processName = cms.InputTag.skipCurrentProcess())
        process.filteredRPCDigiSequence = cms.Sequence( process.simMuonRPCDigis)
        process.RawToDigi += process.filteredRPCDigiSequence

    return process


def appendRPCChamberMaskerAtUnpacking3(process, doDigis, maskedRPCs):
  if doDigis and hasattr(process,'muonRPCDigis') :
    print "[appendChamberMasker] : Found muonRPCDigis, applying filter"
    process.preMuonRPCDigis = process.muonRPCDigis.clone()
    process.simMuonRPCDigis = RPCChamberMasker.clone()
    if len(maskedRPCs) > 0 :
        process.simMuonRPCDigis.maskedRPCIDs = maskedRPCs
        process.simMuonRPCDigis.digiTag = "simMuonRPCDigis"
        process.filteredRPCDigiSequence = cms.Sequence(process.preMuonRPCDigis + process.simMuonRPCDigis)
        process.RawToDigi.replace(process.muonRPCDigis, process.filteredRPCDigiSequence)

    return process

def appendRPCChamberMaskerAtUnpacking4(process, doDigis, maskedRPCs):
    print "[appendChamberMasker] : Found muonRPCDigis, applying filter"
    process.preMuonRPCDigis = process.simMuonRPCDigis.clone()

    process.simMuonRPCDigis = RPCChamberMasker.clone()
    if len(maskedRPCs) > 0 :
        process.simMuonRPCDigis.maskedRPCIDs = maskedRPCs
        process.simMuonRPCDigis.digiTag = "preMuonRPCDigis"
        process.filteredRPCDigiSequence = cms.Sequence(process.preMuonRPCDigis + process.simMuonRPCDigis)
        process.muonDigi.replace(process.simMuonRPCDigis, process.filteredRPCDigiSequence)
    return process

def appendRPCChamberMaskerAtHLT(process, doDigis, maskedRPCs):
        
    print "[appendChamberMasker] : Found hltMuonRPCDigis, applying filter"
    # overwrite the current RPC digi module
    process.hltMuonRPCDigis = RPCChamberMasker.clone()
    if len(maskedRPCs) > 0 :
        process.hltMuonRPCDigis.maskedRPCIDs = maskedRPCs
        process.hltMuonRPCDigis.digiTag = "simMuonRPCDigis"

    return process
