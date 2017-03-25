import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.GEMChamberMasker_cfi import GEMChamberMasker

#def appendChamberMaskerAtUnpacking(process, doDigis, doTrigger, chambRegEx):
def appendGEMChamberMaskerAtUnpacking(process, doDigis, maskedGEMs):

    if doDigis and hasattr(process,'muonGEMDigis') :

        print "[appendChamberMasker] : Found muonGEMDigis, applying filter"

        process.preGEMDigis = process.muonGEMDigis.clone()
        process.muonGEMDigis = GEMChamberMasker.clone()
        if len(maskedGEMs) > 0 :
		process.muonGEMDigis.maskedGEMIDs = maskedGEMs
	process.filteredGEMDigiSequence = cms.Sequence(process.preGEMDigis + process.muonGEMDigis)
        process.RawToDigi.replace(process.muonGEMDigis, process.filteredGEMDigiSequence)
    
    return process

def appendGEMChamberMaskerAtUnpacking2(process, doDigis, maskedGEMs):
  #if doDigis and hasattr(process,'muonGEMDigi') :
    print "[appendChamberMasker] : Found muonGEMDigis, applying filter"
    process.simMuonGEMDigis = GEMChamberMasker.clone()
    if len(maskedGEMs) > 0 :
        process.simMuonGEMDigis.maskedGEMIDs = maskedGEMs
        process.simMuonGEMDigis.digiTag =  cms.InputTag("simMuonGEMDigis", processName = cms.InputTag.skipCurrentProcess())
        process.filteredGEMDigiSequence = cms.Sequence( process.simMuonGEMDigis)
        process.RawToDigi += process.filteredGEMDigiSequence

    return process


def appendGEMChamberMaskerAtHLT(process, doDigis, maskedGEMs):
        
    print "[appendChamberMasker] : Found hltMuonGEMDigis, applying filter"
    # overwrite the current GEM digi module
    process.hltMuonGEMDigis = GEMChamberMasker.clone()
    if len(maskedGEMs) > 0 :
        process.hltMuonGEMDigis.maskedGEMIDs = maskedGEMs
        process.hltMuonGEMDigis.digiTag = "simMuonGEMDigis"

    return process
