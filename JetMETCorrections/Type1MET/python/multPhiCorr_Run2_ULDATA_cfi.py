import FWCore.ParameterSet.Config as cms

# UL 2018A DATA corrections
multPhiCorr_UL2018ADATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2018ADATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.263733,-1.91115),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0431304,-0.112043),
    ),
)

# UL 2018B DATA corrections
multPhiCorr_UL2018BDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2018BDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.400466,-3.05914),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.146125,-0.533233),
    ),
)

# UL 2018C DATA corrections
multPhiCorr_UL2018CDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2018CDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.430911,-1.42865),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0620083,-1.46021),
    ),
)

# UL 2018D DATA corrections
multPhiCorr_UL2018DDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2018DDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.457327,-1.56856),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0684071,-0.928372),
    ),
)

# UL 2017B DATA corrections
multPhiCorr_UL2017BDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2017BDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.211161,0.419333),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.251789,-1.28089),
    ),
)

# UL 2017C DATA corrections
multPhiCorr_UL2017CDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2017CDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.185184,-0.164009),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.200941,-0.56853),
    ),
)

# UL 2017D DATA corrections
multPhiCorr_UL2017DDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2017DDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.201606,0.426502),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.188208,-0.58313),
    ),
)

# UL 2017E DATA corrections
multPhiCorr_UL2017EDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2017EDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.162472,0.176329),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.138076,-0.250239),
    ),
)

# UL 2017F DATA corrections
multPhiCorr_UL2017FDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2017FDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.210639,0.72934),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.198626,1.028),
    ),
)

# UL 2016preVFPB DATA corrections
multPhiCorr_UL2016preVFPBDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016preVFPBDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0214894,-0.188255),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0876624,0.812885),
    ),
)

# UL 2016preVFPC DATA corrections
multPhiCorr_UL2016preVFPCDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016preVFPCDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.032209,0.067288),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.113917,0.743906),
    ),
)

# UL 2016preVFPD DATA corrections
multPhiCorr_UL2016preVFPDDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016preVFPDDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0293663,0.21106),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.11331,0.815787),
    ),
)

# UL 2016preVFPE DATA corrections
multPhiCorr_UL2016preVFPEDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016preVFPEDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0132046,0.20073),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.134809,0.679068),
    ),
)

# UL 2016preVFPF DATA corrections
multPhiCorr_UL2016preVFPFDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016preVFPFDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0543566,0.816597),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.114225,1.17266),
    ),
)

# UL 2016ostVFPF DATA corrections
multPhiCorr_UL2016postVFPFDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016postVFPFDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.134616,-0.89965),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0397736,1.0385),
    ),
)

# UL 2016ostVFPG DATA corrections
multPhiCorr_UL2016postVFPGDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016postVFPGDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.121809,-0.584893),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0558974,0.891234),
    ),
)

# UL 2016ostVFPH DATA corrections
multPhiCorr_UL2016postVFPHDATA = cms.VPSet(
    cms.PSet(
      name=cms.string("UL2016postVFPHDATA"),
      type=cms.int32(0),
      varType=cms.int32(1),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.0868828,-0.703489),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0888774,0.902632),
    ),
)
