import FWCore.ParameterSet.Config as cms
from EventFilter.CSCRawToDigi.cscDigiFilterDef_cfi import cscDigiFilterDef

def appendCSCChamberMaskerAtUnpacking(process):

    if hasattr(process,'muonCSCDigis') :

        # clone the original producer
        process.preCSCDigis = process.muonCSCDigis.clone()
        # now apply the filter
        process.muonCSCDigis = cscDigiFilterDef.clone(
            stripDigiTag = "preCSCDigis:MuonCSCStripDigi",
            wireDigiTag = "preCSCDigis:MuonCSCWireDigi",
            comparatorDigiTag = "preCSCDigis:MuonCSCComparatorDigi",
            alctDigiTag = "preCSCDigis:MuonCSCALCTDigi",
            clctDigiTag = "preCSCDigis:MuonCSCCLCTDigi",
            lctDigiTag = "preCSCDigis:MuonCSCCorrelatedLCTDigi",
            showerDigiTag = "preCSCDigis:MuonCSCShowerDigi",
            gemPadClusterDigiTag = "preCSCDigis:MuonGEMPadDigiCluster",
            maskedChambers = [],
            selectedChambers = []
        )
        process.RawToDigiTask.add(process.preCSCDigis)

    return process

def maskExperimentalME11ChambersRun2(process):
    process = appendCSCChamberMaskerAtUnpacking(process)
    # these 3 chambers had Phase-2 firmware loaded partially during Run-2
    process.muonCSCDigis.maskedChambers = [
        "ME+1/1/9", "ME+1/1/10", "ME+1/1/11"]
