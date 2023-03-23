import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *

l1_float_precision_=12
print("l1pre",l1_float_precision_)
l1PtVars = cms.PSet(
    pt  = Var("pt",  float, precision=l1_float_precision_),
    phi = Var("phi", float, precision=l1_float_precision_),
)
l1P3Vars = cms.PSet(
    l1PtVars,
    eta = Var("eta", float, precision=l1_float_precision_),
)

l1ObjVars = cms.PSet(
    l1P3Vars,
    hwPt = Var("hwPt()","int16",doc="hardware pt"),
    hwEta = Var("hwEta()","int16",doc="hardware eta"),
    hwPhi = Var("hwPhi()","int16",doc="hardware phi"),
    hwQual = Var("hwQual()","int16",doc="hardware qual"),
    hwIso = Var("hwIso()","int16",doc="hardware iso")
)

l1CaloObjVars = cms.PSet(
    l1ObjVars,
    towerIEta = Var("towerIEta()","int16",doc="the ieta of the tower"),
    towerIPhi = Var("towerIPhi()","int16",doc="the iphi of the tower"),
)

l1JetReducedVars = cms.PSet(
    l1P3Vars
)

l1EtSumReducedVars = cms.PSet(
    l1PtVars,
    etSumType = Var("getType()",int,doc="et sum type"),
)
l1EGReducedVars = cms.PSet(
    l1P3Vars,
    hwIso = Var("hwIso()","int16",doc="hardware iso")
)

l1TauReducedVars = cms.PSet(
    l1P3Vars,
    hwIso = Var("hwIso()","int16",doc="hardware iso")
)

l1MuonReducedVars = cms.PSet(
    l1P3Vars,
    hwQual = Var("hwQual()",int,doc="hardware qual"),
    hwCharge = Var("hwCharge()","int16",doc="hardware charge"), 
    etaAtVtx = Var("etaAtVtx()",float,precision=l1_float_precision_,doc="eta estimated at the vertex"),
    phiAtVtx = Var("phiAtVtx()",float,precision=l1_float_precision_,doc="phi estimated at the vertex"),
    ptUnconstrained = Var("ptUnconstrained()",float,precision=l1_float_precision_,doc="pt when not constrained to the beamspot"),
    hwDXY = Var("hwDXY()","int16",doc="hardware impact parameter"),
)

l1MuTable = cms.EDProducer("SimpleTriggerL1MuonFlatTableProducer",
    src = cms.InputTag("gmtStage2Digis","Muon"),
    minBX = cms.int32(-2),
    maxBX = cms.int32(2),                           
    cut = cms.string(""), 
    name= cms.string("L1Mu"),
    doc = cms.string(""),
    extension = cms.bool(False), 
    variables = cms.PSet(l1ObjVars,
                         hwCharge = Var("hwCharge()","int16",doc="Charge (can be 0 if the charge measurement was not valid)"),
                         hwChargeValid = Var("hwChargeValid()","int16",doc=""),
                         tfMuonIndex = Var("tfMuonIndex()","uint16",doc="Index of muon at the uGMT input. 3 indices per link/sector/wedge. EMTF+ are 0-17, OMTF+ are 18-35, BMTF are 36-71, OMTF- are 72-89, EMTF- are 90-107"),
                         hwTag = Var("hwTag()","int16",doc="not in L1 ntuples"),
                         hwEtaAtVtx = Var("hwEtaAtVtx()","int16",doc="hardware eta estimated at the vertex"),
                         hwPhiAtVtx = Var("hwPhiAtVtx()","int16",doc="hardware phi estimated at the vertex"),
                         etaAtVtx = Var("etaAtVtx()",float,doc="eta estimated at the vertex"),
                         phiAtVtx = Var("phiAtVtx()",float,doc="phi estimated at the vertex"),
                         hwIsoSum = Var("hwIsoSum()","int16",doc="not in L1 ntuples"),
                         hwDPhiExtra = Var("hwDPhiExtra()","int16",doc="Delta between Pseudo-rapidity at the muon system and the projected coordinate at the vertex in HW unit (for future l1t-integration-tag"),
                         hwDEtaExtra = Var("hwDEtaExtra()","int16",doc="Delta between Azimuth at the muon system and the projected coordinate at the vertex in HW unit (for future l1t-integration-tag)"),
                         hwRank = Var("hwRank()","int16",doc="not in L1Ntuples"),
                         hwPtUnconstrained = Var("hwPtUnconstrained()","int16",doc=""),
                         ptUnconstrained = Var("ptUnconstrained()",float,doc=""),
                         hwDXY = Var("hwDXY()","uint16",doc=""),
                     )
)


l1JetTable = cms.EDProducer("SimpleTriggerL1JetFlatTableProducer",
    src = cms.InputTag("caloStage2Digis","Jet"),
    minBX = cms.int32(-2),
    maxBX = cms.int32(2),                           
    cut = cms.string(""), 
    name= cms.string("L1Jet"),
    doc = cms.string(""),
    extension = cms.bool(False),
    variables = cms.PSet(l1CaloObjVars,
                         rawEt = Var("rawEt()","int16",doc="raw (uncalibrated) et"),
                         seedEt = Var("seedEt()","int16",doc="et of the seed"),
                         puEt = Var("puEt()","int16",doc="pile up et "),
                         puDonutEt0 = Var("puDonutEt(0)","int16",doc=""),
                         puDonutEt1 = Var("puDonutEt(1)","int16",doc=""),
                         puDonutEt2 = Var("puDonutEt(2)","int16",doc=""),
                         puDonutEt3 = Var("puDonutEt(3)","int16",doc=""),
                     )
)

l1TauTable = cms.EDProducer("SimpleTriggerL1TauFlatTableProducer",
    src = cms.InputTag("caloStage2Digis","Tau"),
    minBX = cms.int32(-2),
    maxBX = cms.int32(2),                           
    cut = cms.string(""), 
    name= cms.string("L1Tau"),
    doc = cms.string(""),
    extension = cms.bool(False), # this is the main table for L1 EGs
    variables = cms.PSet(l1CaloObjVars,
                         rawEt = Var("rawEt()","int16",doc="raw Et of tau"),
                         isoEt = Var("isoEt()","int16",doc="raw isolation sum - cluster sum"),
                         nTT = Var("nTT()","int16",doc=" nr towers above threshold"),
                         hasEM = Var("hasEM()",bool,doc="has an em component"),
                         isMerged = Var("isMerged()",bool,doc="is merged"),

                     )
)

l1EtSumTable = cms.EDProducer("SimpleTriggerL1EtSumFlatTableProducer",
    src = cms.InputTag("caloStage2Digis","EtSum"),
    minBX = cms.int32(-2),
    maxBX = cms.int32(2),                           
    cut = cms.string(""), 
    name= cms.string("L1EtSum"),
    doc = cms.string(""),
    extension = cms.bool(False), 
    variables = cms.PSet(l1PtVars,
                         hwPt = Var("hwPt()",int,doc="hardware pt"),
                         hwPhi = Var("hwPhi()",int,doc="hardware phi"),
                         etSumType = Var("getType()",int,doc="the type of the ET Sum (https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1Trigger/interface/EtSum.h#L27-L56)"),
                     )
)

l1EGTable = cms.EDProducer("SimpleTriggerL1EGFlatTableProducer",
    src = cms.InputTag("caloStage2Digis","EGamma"),
    minBX = cms.int32(-2),
    maxBX = cms.int32(2),                           
    cut = cms.string(""), 
    name= cms.string("L1EG"),
    doc = cms.string(""),
    extension = cms.bool(False), 
    variables = cms.PSet(l1CaloObjVars,
                         rawEt = Var("rawEt()","int16",doc="raw et"),
                         isoEt = Var("isoEt()","int16",doc="iso et"),
                         footprintEt = Var("footprintEt()","int16",doc=" footprint et"),
                         nTT = Var("nTT()","int16",doc="nr trig towers"),
                         shape = Var("shape()","int16",doc="shape"),
                         towerHoE = Var("towerHoE()","int16",doc="tower H/E"),
                     )
)

l1TablesTask = cms.Task(l1EGTable,l1EtSumTable,l1TauTable,l1JetTable,l1MuTable)

def setL1NanoToReduced(process):
    """
    sets the L1 objects only have reduced information which is necessary 
    for central nano
    """
    #reduce the variables to the core variables
    #note et sum variables are already reduced
    process.l1EGTable.variables = cms.PSet(l1EGReducedVars)
    process.l1MuTable.variables = cms.PSet(l1MuonReducedVars)
    process.l1JetTable.variables = cms.PSet(l1JetReducedVars)
    process.l1TauTable.variables = cms.PSet(l1TauReducedVars)
    process.l1EtSumTable.variables = cms.PSet(l1EtSumReducedVars)
   
    #apply cuts
    process.l1EGTable.cut="pt>=10"
    process.l1TauTable.cut="pt>=24"
    process.l1JetTable.cut="pt>=30"
    process.l1MuTable.cut="pt>=3 && hwQual>=8"
    process.l1EtSumTable.cut="(getType==8 || getType==1 || getType==2 || getType==3)"
    
    return process
