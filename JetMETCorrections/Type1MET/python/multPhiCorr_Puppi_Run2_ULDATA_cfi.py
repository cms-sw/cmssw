import FWCore.ParameterSet.Config as cms

# ULDATA 2018A  corrections
multPhiCorr_Puppi_ULDATA2018A = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018A"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.0073377,0.0250294),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.000406059,0.0417346),
    ),
)

# ULDATA 2018B  corrections
multPhiCorr_Puppi_ULDATA2018B = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018B"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.00434261,0.00892927),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.00234695,0.20381),
    ),
)

# ULDATA 2018C  corrections
multPhiCorr_Puppi_ULDATA2018C = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018C"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.00198311,0.37026),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.016127,0.402029),
    ),
)

# ULDATA 2018D  corrections
multPhiCorr_Puppi_ULDATA2018D = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2018D"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.00220647,0.378141),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.0160244,0.471053),
    ),
)

# ULDATA 2017B  corrections
multPhiCorr_Puppi_ULDATA2017B = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017B"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00382117,-0.666228),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.0109034,0.172188),
    ),
)

# ULDATA 2017C  corrections
multPhiCorr_Puppi_ULDATA2017C = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017C"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00110699,-0.747643),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.0012184,0.303817),
    ),
)

# ULDATA 2017D  corrections
multPhiCorr_Puppi_ULDATA2017D = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017D"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00141442,-0.721382),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.0011873,0.21646),
    ),
)

# ULDATA 2017E  corrections
multPhiCorr_Puppi_ULDATA2017E = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017E"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.00593859,-0.851999),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.00754254,0.245956),
    ),
)

# ULDATA 2017F  corrections
multPhiCorr_Puppi_ULDATA2017F = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2017F"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.00765682,-0.945001),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.0154974,0.804176),
    ),
)

# ULDATA 2016preVFPB  corrections
multPhiCorr_Puppi_ULDATA2016preVFPB = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPB"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00109025,-0.338093),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.00356058,0.128407),
    ),
)

# ULDATA 2016preVFPC  corrections
multPhiCorr_Puppi_ULDATA2016preVFPC = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPC"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00271913,-0.342268),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(0.00187386,0.104),
    ),
)

# ULDATA 2016preVFPD  corrections
multPhiCorr_Puppi_ULDATA2016preVFPD = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPD"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00254194,-0.305264),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.00177408,0.164639),
    ),
)

# ULDATA 2016preVFPE  corrections
multPhiCorr_Puppi_ULDATA2016preVFPE = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPE"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(-0.00358835,-0.225435),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.000444268,0.180479),
    ),
)

# ULDATA 2016preVFPF  corrections
multPhiCorr_Puppi_ULDATA2016preVFPF = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016preVFPF"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.0056759,-0.454101),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.00962707,0.35731),
    ),
)

# ULDATA 2016ostVFPF  corrections
multPhiCorr_Puppi_ULDATA2016postVFPF = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016postVFPF"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.0234421,-0.371298),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.00997438,0.0809178),
    ),
)

# ULDATA 2016ostVFPG  corrections
multPhiCorr_Puppi_ULDATA2016postVFPG = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016postVFPG"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.0182134,-0.335786),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.0063338,0.093349),
    ),
)

# ULDATA 2016ostVFPH  corrections
multPhiCorr_Puppi_ULDATA2016postVFPH = cms.VPSet(
    cms.PSet(
      name=cms.string("ULDATA2016postVFPH"),
      type=cms.int32(0),
      varType=cms.int32(3),
      etaMin=cms.double(-9.9),
      etaMax=cms.double(9.9),
      fx=cms.string("((x*[0])+[1])"),
      px=cms.vdouble(0.015702,-0.340832),
      fy=cms.string("((x*[0])+[1])"),
      py=cms.vdouble(-0.00544957,0.199093),
    ),
)
