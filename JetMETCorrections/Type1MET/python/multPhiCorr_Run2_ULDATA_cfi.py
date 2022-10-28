import FWCore.ParameterSet.Config as cms

# ULDATA 2018A  corrections
multPhiCorr_ULDATA2018A = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018A"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.263733,-1.91115),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0431304,-0.112043),
    ),
)

# ULDATA 2018B  corrections
multPhiCorr_ULDATA2018B = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018B"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.400466,-3.05914),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.146125,-0.533233),
    ),
)

# ULDATA 2018C  corrections
multPhiCorr_ULDATA2018C = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018C"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.430911,-1.42865),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0620083,-1.46021),
    ),
)

# ULDATA 2018D  corrections
multPhiCorr_ULDATA2018D = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018D"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.457327,-1.56856),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0684071,-0.928372),
    ),
)

# ULDATA 2017B  corrections
multPhiCorr_ULDATA2017B = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017B"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.211161,0.419333),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.251789,-1.28089),
    ),
)

# ULDATA 2017C  corrections
multPhiCorr_ULDATA2017C = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017C"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.185184,-0.164009),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.200941,-0.56853),
    ),
)

# ULDATA 2017D  corrections
multPhiCorr_ULDATA2017D = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017D"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.201606,0.426502),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.188208,-0.58313),
    ),
)

# ULDATA 2017E  corrections
multPhiCorr_ULDATA2017E = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017E"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.162472,0.176329),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.138076,-0.250239),
    ),
)

# ULDATA 2017F  corrections
multPhiCorr_ULDATA2017F = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017F"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.210639,0.72934),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.198626,1.028),
    ),
)

# ULDATA 2016preVFPB  corrections
multPhiCorr_ULDATA2016preVFPB = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPB"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0214894,-0.188255),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0876624,0.812885),
    ),
)

# ULDATA 2016preVFPC  corrections
multPhiCorr_ULDATA2016preVFPC = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPC"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.032209,0.067288),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.113917,0.743906),
    ),
)

# ULDATA 2016preVFPD  corrections
multPhiCorr_ULDATA2016preVFPD = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPD"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0293663,0.21106),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.11331,0.815787),
    ),
)

# ULDATA 2016preVFPE  corrections
multPhiCorr_ULDATA2016preVFPE = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPE"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0132046,0.20073),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.134809,0.679068),
    ),
)

# ULDATA 2016preVFPF  corrections
multPhiCorr_ULDATA2016preVFPF = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPF"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0543566,0.816597),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.114225,1.17266),
    ),
)

# ULDATA 2016ostVFPF  corrections
multPhiCorr_ULDATA2016postVFPF = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016postVFPF"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.134616,-0.89965),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0397736,1.0385),
    ),
)

# ULDATA 2016ostVFPG  corrections
multPhiCorr_ULDATA2016postVFPG = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016postVFPG"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.121809,-0.584893),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0558974,0.891234),
    ),
)

# ULDATA 2016ostVFPH  corrections
multPhiCorr_ULDATA2016postVFPH = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016postVFPH"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.0868828,-0.703489),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0888774,0.902632),
    ),
)
