import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from DPGAnalysis.MuonTools.common_cff import *

from DPGAnalysis.MuonTools.dtDigiFlatTableProducer_cfi import dtDigiFlatTableProducer

dtDigiFlatTable = dtDigiFlatTableProducer.clone(
    name = "dtDigi", 
    src = "muonDTDigis", 
    doc = "DT digi information",

    variables = cms.PSet(
        time = Var("time()", float, doc = "digi time"),
        wire = Var("wire()", "int16", doc="wire - [1:X] range"
                                      "<br />(X varies for different chambers SLs and layers)")
    ),

    detIdVariables = cms.PSet(
        wheel = DetIdVar("wheel()", "int16", doc = "wheel  -  [-2:2] range"),
        sector = DetIdVar("sector()", "int16", doc = "sector - [1:14] range"
                                            "<br />sector 13 used for the second MB4 of sector 4"
                                            "<br />sector 14 used for the second MB4 of sector 10"),
        station = DetIdVar("station()", "int16", doc = "station - [1:4] range"),
        superLayer = DetIdVar("superLayer()", "int16", doc = "superlayer - [1:3] range"
                                                    "<br />SL 1 and 3 are phi SLs"
                                                    "<br />SL 2 is theta SL"),
        layer = DetIdVar("layer()", "int16", doc = "layer  -  [1:4] range")
     )
)

from DPGAnalysis.MuonTools.rpcDigiFlatTableProducer_cfi import rpcDigiFlatTableProducer

rpcDigiFlatTable = rpcDigiFlatTableProducer.clone(
    name = "rpcDigi",
    src = "muonRPCDigis",
    doc = "RPC digi information",

    variables = cms.PSet(
        strip = Var("strip()", "uint8", doc = "index of the readout strip associated to the digi"),
        bx = Var("bx()", int, doc="bunch crossing associated to the digi")
    ),

    detIdVariables = cms.PSet(
        region = DetIdVar("region()", "int16", doc = "0: barrel, +/-1: endcap"),
        ring = DetIdVar("ring()", "int16", doc = "ring id:"
                                        "<br />wheel number in barrel - [-2:+2] range"
                                        "<br />ring number in endcap - [1:3] range"),
        station = DetIdVar("station()", "int16", doc = "chambers at same R in barrel, chambers at same Z ion endcap"),
        layer = DetIdVar("layer()", "int16", doc = "layer id:"
                                          "<br />barrel stations 1 and 2, have two layers of chambers "
                                          "(layer 1 is the inner chamber and layer 2 is the outer chamber)"),
        sector = DetIdVar("sector()", "int16", doc = "group of chambers at same phi"),
        subsector = DetIdVar("subsector()", "int16", doc = "Some sectors are divided along the phi direction in subsectors "
                                                  "(from 1 to 4 in Barrel, from 1 to 6 in Endcap)"),
        roll = DetIdVar("roll()", "int16", doc = "roll id (also known as eta partition):"
                                        "<br />each chamber is divided along the strip direction"),
        rawId = DetIdVar("rawId()", "uint", doc = "unique detector unit ID")
    )
)

from DPGAnalysis.MuonTools.gemDigiFlatTableProducer_cfi import gemDigiFlatTableProducer

gemDigiFlatTable = gemDigiFlatTableProducer.clone(
    name = "gemDigi",
    src = "muonGEMDigis",
    doc = "GEM digi information",

    variables = cms.PSet(
        strip = Var("strip()", "int16", doc = "index of the readout strip associated to the digi"),
        bx = Var("bx()", "int16", doc="bunch crossing associated to the digi")
    ),

    detIdVariables = cms.PSet(
        station = DetIdVar("station()", "int16", doc = "GEM station <br />(always 1 for GE1/1)"),
        region = DetIdVar("region()", "int16", doc = "GE11 region where the digi is detected"
                                            "<br />(int, positive endcap: +1, negative endcap: -1)"),
        roll = DetIdVar("roll()", "int16", doc = "roll id (also known as eta partition)"
                                        "<br />(partitions numbered from 1 to 8)"),
        chamber = DetIdVar("chamber()", "int16", doc = "GE11 superchamber where the hit is reconstructed"
                                              "<br />(chambers numbered from 0 to 35)"),
        layer = DetIdVar("layer()", "int16", doc = "GE11 layer where the hit is reconstructed"
                                          "<br />(layer1: 1, layer2: 2)")        
    )
)

from DPGAnalysis.MuonTools.gemohStatusFlatTableProducer_cfi import gemohStatusFlatTableProducer

gemohStatusFlatTable = gemohStatusFlatTableProducer.clone(
    name = "gemOHStatus",
    src = "muonGEMDigis:OHStatus:",
    doc = "GEM OH status information",

    variables = cms.PSet(
        chamberType = Var("chamberType()", "int", doc = "two digits number that specifies the module within a chamber<br /> 11,12 for GE1/1 chambers layer 1,2<br /> 21,22,23,24 for GE2/1 chambers module 1,2,3,4"),
        vfatMask = Var("vfatMask()", "uint", doc = "24 bit word that specifies the VFAT Mask<br /> nth bit == 0 means that the VFAT_n was masked from the DAQ in the event"),
        zsMask = Var("zsMask()", "uint", doc = "24 bit word that specifies the Zero Suppression<br /> nth bit == 1 means that the VFAT_n was zero suppressed"),
        missingVFATs = Var("missingVFATs()", "uint", doc = "24 bit word that specifies the missing VFAT mask<br /> nth bit == 1 means that the VFAT_n was expected in the payload but not found"),
        errors = Var("errors()", "uint16", doc = "code for GEM OH errors<br /> non-zero values indicate errors"),
        warnings = Var("warnings()", "uint16", doc = "code for GEM OH warnings<br /> non-zero values indicate warnings")
    ),

    detIdVariables = cms.PSet(
        station = DetIdVar("station()", "int16", doc = "GEM station <br />always 1 for GE1/1)"),
        region = DetIdVar("region()", "int16", doc = "region with which the GEMOHStatus is associated"
                                            "<br />int, positive endcap: +1, negative endcap: -1"),
        chamber = DetIdVar("chamber()", "int16", doc = "chamber with which the GEMOHStatus is associated"),
        layer = DetIdVar("layer()", "int16", doc = "layer with which the GEMOHStatus is associated<br /> either 1 or 2 for GE1/1 and GE2/1")
    )
)


from DPGAnalysis.MuonTools.cscWireDigiFlatTableProducer_cfi import cscWireDigiFlatTableProducer

cscWireDigiFlatTable = cscWireDigiFlatTableProducer.clone(
    name = "cscWireDigi",
    src = "muonCSCDigis:MuonCSCWireDigi",
    doc = "CSC wire digi information",

    variables = cms.PSet(
        timeBin = Var("getTimeBin()", "int16", doc = ""),
        wireGroup = Var("getWireGroup()", "int16", doc=""),
        wireGroupBX = Var("getWireGroupBX()", "int16", doc="")
    ),

    detIdVariables = cms.PSet(
        endcap = DetIdVar("endcap()", "int16", doc = ""),
        station = DetIdVar("station()", "int16", doc = ""),
        ring = DetIdVar("ring()", "int16", doc = ""),
        chamber = DetIdVar("chamber()", "int16", doc = ""),
        layer = DetIdVar("layer()", "int16", doc = "")
    )
)

from DPGAnalysis.MuonTools.cscAlctDigiFlatTableProducer_cfi import cscAlctDigiFlatTableProducer

cscAlctDigiFlatTable = cscAlctDigiFlatTableProducer.clone(
    name = "cscALCTDigi",
    src = "muonCSCDigis:MuonCSCALCTDigi:",
    doc = "CSC ALCT digi information",

    variables = cms.PSet(
        keyWireGroup = Var("getKeyWG()", "int16", doc = ""),
        bx = Var("getBX()", "int16", doc="")
    ),

    detIdVariables = cms.PSet(
        endcap = DetIdVar("endcap()", "int16", doc = ""),
        station = DetIdVar("station()", "int16", doc = ""),
        ring = DetIdVar("ring()", "int16", doc = ""),
        chamber = DetIdVar("chamber()", "int16", doc = ""),
        layer = DetIdVar("layer()", "int16", doc = "")
    )
)

muDigiTables = cms.Sequence(dtDigiFlatTable
                            + rpcDigiFlatTable
                            + gemDigiFlatTable
                            + gemohStatusFlatTable
                        )

muDigiTablesBkg = cms.Sequence(dtDigiFlatTable
                               + rpcDigiFlatTable
                               + cscAlctDigiFlatTable
                               + cscWireDigiFlatTable
                               + gemDigiFlatTable
                               + gemohStatusFlatTable
                        )
