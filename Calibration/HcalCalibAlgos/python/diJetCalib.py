import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalCalib")

process.load("Configuration.StandardSequences.GeometryHCAL_cff")

# no CMS input files
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(0) )
process.source = cms.Source("EmptySource")

                          
process.hcalCalib = cms.EDAnalyzer("HcalCalibrator",



#-------- File containing the list of the input root files
  inputFileList = cms.untracked.string("test/inputFiles_DiJetCalib.list"),
                                   
#-------- calibration type: ISO_TRACK or DI_JET
  calibType = cms.untracked.string("DI_JET"),

#-------- calibration method: L3, MATRIX_INV_OF_ETA_AVE, L3_AND_MTRX_INV
#-------- for diJets use only L3!!!
  calibMethod = cms.untracked.string("L3"),

# --------min target energy: ref jet (had) energy, or isotrack momentum
  minTargetE = cms.untracked.double(30.0),

# ------- max target energy: ref jet (had) energy, or isotrack momentum                                   
  maxTargetE = cms.untracked.double(9999.0),

# ------- min energy in the cell                                   
  minCellE = cms.untracked.double(0.0),

# ------- minimum e/p to accept an IsoTrack for calibration (not used for DiJets)  
  minEOverP = cms.untracked.double(0.2),

# ------- maximum e/p to accept an IsoTrack for calibration (not used for DiJets)  
  maxEOverP = cms.untracked.double(999.0),

# ------- MIP cut: maximum EM energy associated with the track (not used for DiJEts)  
  maxTrkEmE = cms.untracked.double(1.0),                                  
                                   
#--------Maximum Et allowed for third jet in the event (affects only DiJets)                                   
  maxEtThirdJet = cms.untracked.double(5.0),

# ------- Minimum deltaPhi for the dijets in degrees: 0-180  (affects only DiJets)
  minDPhiDiJets = cms.untracked.double(150.0),

# ------- logical flag to sum depths in HCAL                                   
  sumDepths = cms.untracked.bool(True),

# ------- logical flag to combine phi in HCAL                                   
  combinePhi = cms.untracked.bool(True),

# ------- logical flag to sum depths 1,2 in HB for towers 15,16.
# --------If sumDepths="true" this flag has no effect
  sumSmallDepths = cms.untracked.bool(True),

# ------- cluster size in HB for isotracks: 3 or 5 (means 3x3, 5x5)                                   
  hbClusterSize = cms.untracked.int32(3),

# ------- cluster size in HE for isotracks: 3 or 5 (means 3x3, 5x5)
  heClusterSize = cms.untracked.int32(5),

# -------- flag to use cone clustering -> overrides the above cluster sizes
# -------- and uses cone size as specified below
 useConeClustering = cms.untracked.bool(True),

# -------- size of the cone (when useConeClustering=True)
 maxConeDist = cms.untracked.double(26.2),

# ------- max ABS(iEta) used in the calibration: for matrix inversion sets the range for performing inversion
# ------- For all methods: controls the range of correction factors are saved in the output file                                   
  calibAbsIEtaMax  = cms.untracked.int32(41),

# ------- min ABS(iEta) used in the calibration: for matrix inversion sets the range for performing inversion
# ------- For all methods: controls the range of correction factors are saved in the output file                                     
  calibAbsIEtaMin  = cms.untracked.int32(21),
                                   
# ------- max EmFraction of probe jet allowed (does not affect isotracks)
  maxProbeJetEmFrac  = cms.untracked.double(0.1),

# ------- max EmFraction of tag jet allowed (does not affect isotracks)
  maxTagJetEmFrac = cms.untracked.double(1.0),
                                   
# ------- max abs(eta) of tag jet allowed (does not affect isotracks)
  maxTagJetAbsEta = cms.untracked.double(1.0),

# ------- min Et of tag jet allowed (does not affect isotracks)
  minTagJetEt = cms.untracked.double(30.0),

# ------- min abs(eta) of probe jet allowed (does not affect isotracks)
  minProbeJetAbsEta = cms.untracked.double(1.4),
                                   
# ------- file containing the phi symmetry corrections (to apply on the fly in isoTrack calibration)
# ------- USER PROVIDED!
  phiSymCorFileName = cms.untracked.string("phiSymCor.txt"),

# ------- Flag to read phi symmetry corrections and apply on the fly (can be used for isoTrack)
# ------- Set to false if already applied in Analyzer
  applyPhiSymCorFlag = cms.untracked.bool(False),

# ------- output file with correction factors
  outputCorCoefFileName = cms.untracked.string("calibConst_DiJet.txt"),
                                   
# ------- output file with some hisograms
  histoFileName = cms.untracked.string("histoFile.root")

) 


process.p = cms.Path(process.hcalCalib)
