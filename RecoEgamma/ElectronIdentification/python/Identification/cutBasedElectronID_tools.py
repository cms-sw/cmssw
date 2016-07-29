
import FWCore.ParameterSet.Config as cms

# Barrel/endcap division in eta
ebCutOff = 1.479

# ===============================================
# Define containers used by cut definitions
# ===============================================

class IsolationCutInputs_V2:
    """
    A container class that holds the name of the file with the effective 
    area constants for pile-up corrections
    """
    def __init__(self, 
                 isoEffAreas
                 ):
                 self.isoEffAreas    = isoEffAreas


class EleWorkingPoint_V1:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts for electron cut-based ID
    """
    def __init__(self, 
                 idName,
                 dEtaInCut,
                 dPhiInCut,
                 full5x5_sigmaIEtaIEtaCut,
                 hOverECut,
                 dxyCut,
                 dzCut,
                 absEInverseMinusPInverseCut,
                 relCombIsolationWithDBetaLowPtCut,
                 relCombIsolationWithDBetaHighPtCut,
                 # conversion veto cut needs no parameters, so not mentioned
                 missingHitsCut
                 ):
        self.idName                       = idName                       
        self.dEtaInCut                    = dEtaInCut                   
        self.dPhiInCut                    = dPhiInCut                   
        self.full5x5_sigmaIEtaIEtaCut     = full5x5_sigmaIEtaIEtaCut    
        self.hOverECut                    = hOverECut                   
        self.dxyCut                       = dxyCut                      
        self.dzCut                        = dzCut                       
        self.absEInverseMinusPInverseCut  = absEInverseMinusPInverseCut 
        self.relCombIsolationWithDBetaLowPtCut  = relCombIsolationWithDBetaLowPtCut
        self.relCombIsolationWithDBetaHighPtCut = relCombIsolationWithDBetaHighPtCut
        # conversion veto cut needs no parameters, so not mentioned
        self.missingHitsCut               = missingHitsCut
        
class EleWorkingPoint_V2:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts for electron cut-based ID
    """
    def __init__(self, 
                 idName,
                 dEtaInCut,
                 dPhiInCut,
                 full5x5_sigmaIEtaIEtaCut,
                 hOverECut,
                 dxyCut,
                 dzCut,
                 absEInverseMinusPInverseCut,
                 relCombIsolationWithEALowPtCut,
                 relCombIsolationWithEAHighPtCut,
                 # conversion veto cut needs no parameters, so not mentioned
                 missingHitsCut
                 ):
        self.idName                       = idName                       
        self.dEtaInCut                    = dEtaInCut                   
        self.dPhiInCut                    = dPhiInCut                   
        self.full5x5_sigmaIEtaIEtaCut     = full5x5_sigmaIEtaIEtaCut    
        self.hOverECut                    = hOverECut                   
        self.dxyCut                       = dxyCut                      
        self.dzCut                        = dzCut                       
        self.absEInverseMinusPInverseCut  = absEInverseMinusPInverseCut 
        self.relCombIsolationWithEALowPtCut  = relCombIsolationWithEALowPtCut
        self.relCombIsolationWithEAHighPtCut = relCombIsolationWithEAHighPtCut
        # conversion veto cut needs no parameters, so not mentioned
        self.missingHitsCut               = missingHitsCut

class EleWorkingPoint_V3:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts for electron cut-based ID
    With resepect to V2, the impact parameter cuts on dxy and dz are removed.
    """
    def __init__(self, 
                 idName,
                 dEtaInSeedCut,
                 dPhiInCut,
                 full5x5_sigmaIEtaIEtaCut,
                 hOverECut,
                 absEInverseMinusPInverseCut,
                 relCombIsolationWithEALowPtCut,
                 relCombIsolationWithEAHighPtCut,
                 # conversion veto cut needs no parameters, so not mentioned
                 missingHitsCut
                 ):
        self.idName                       = idName                       
        self.dEtaInSeedCut                = dEtaInSeedCut                   
        self.dPhiInCut                    = dPhiInCut                   
        self.full5x5_sigmaIEtaIEtaCut     = full5x5_sigmaIEtaIEtaCut    
        self.hOverECut                    = hOverECut                   
        self.absEInverseMinusPInverseCut  = absEInverseMinusPInverseCut 
        self.relCombIsolationWithEALowPtCut  = relCombIsolationWithEALowPtCut
        self.relCombIsolationWithEAHighPtCut = relCombIsolationWithEAHighPtCut
        # conversion veto cut needs no parameters, so not mentioned
        self.missingHitsCut               = missingHitsCut

class EleHLTSelection_V1:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts for electron cut-based HLT-safe preselection
    """
    def __init__(self, 
                 idName,
                 full5x5_sigmaIEtaIEtaCut,
                 dEtaInSeedCut,
                 dPhiInCut,
                 hOverECut,
                 absEInverseMinusPInverseCut,
                 # isolations
                 ecalPFClusterIsoLowPtCut,
                 ecalPFClusterIsoHighPtCut,
                 hcalPFClusterIsoLowPtCut,
                 hcalPFClusterIsoHighPtCut,
                 trkIsoSlopeTerm,
                 trkIsoSlopeStart,
                 trkIsoConstTerm,
                 #
                 normalizedGsfChi2Cut
                 ):
        self.idName                       = idName                       
        self.full5x5_sigmaIEtaIEtaCut     = full5x5_sigmaIEtaIEtaCut    
        self.dEtaInSeedCut                = dEtaInSeedCut                   
        self.dPhiInCut                    = dPhiInCut                   
        self.hOverECut                    = hOverECut                   
        self.absEInverseMinusPInverseCut  = absEInverseMinusPInverseCut 
        self.ecalPFClusterIsoLowPtCut     = ecalPFClusterIsoLowPtCut     
        self.ecalPFClusterIsoHighPtCut    = ecalPFClusterIsoHighPtCut    
        self.hcalPFClusterIsoLowPtCut     = hcalPFClusterIsoLowPtCut     
        self.hcalPFClusterIsoHighPtCut    = hcalPFClusterIsoHighPtCut    
        self.trkIsoSlopeTerm              = trkIsoSlopeTerm              
        self.trkIsoSlopeStart             = trkIsoSlopeStart             
        self.trkIsoConstTerm              = trkIsoConstTerm              
        #                                                                
        self.normalizedGsfChi2Cut       = normalizedGsfChi2Cut       

        
# ==============================================================
# Define individual cut configurations used by complete cut sets
# ==============================================================


# The mininum pt cut is set to 5 GeV
def psetMinPtCut():
    return cms.PSet( 
        cutName = cms.string("MinPtCut"),
        minPt = cms.double(5.0),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)                
        )

# Take all particles in the eta ranges 0-ebCutOff and ebCutOff-2.5
def psetPhoSCEtaMultiRangeCut():
    return cms.PSet( 
        cutName = cms.string("GsfEleSCEtaMultiRangeCut"),
        useAbsEta = cms.bool(True),
        allowedEtaRanges = cms.VPSet( 
            cms.PSet( minEta = cms.double(0.0), 
                      maxEta = cms.double(ebCutOff) ),
            cms.PSet( minEta = cms.double(ebCutOff), 
                      maxEta = cms.double(2.5) )
            ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on full5x5 sigmaIEtaIEta
def psetPhoFull5x5SigmaIEtaIEtaCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleFull5x5SigmaIEtaIEtaCut'),
        full5x5SigmaIEtaIEtaCutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
        full5x5SigmaIEtaIEtaCutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on dEta seed
def psetDEtaInSeedCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleDEtaInSeedCut'),
        dEtaInSeedCutValueEB = cms.double( wpEB.dEtaInSeedCut ),
        dEtaInSeedCutValueEE = cms.double( wpEE.dEtaInSeedCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure dPhiIn cut
def psetDPhiInCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleDPhiInCut'),
        dPhiInCutValueEB = cms.double( wpEB.dPhiInCut ),
        dPhiInCutValueEE = cms.double( wpEE.dPhiInCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure H/E cut
def psetHadronicOverEMCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleHadronicOverEMCut'),
        hadronicOverEMCutValueEB = cms.double( wpEB.hOverECut ),
        hadronicOverEMCutValueEE = cms.double( wpEE.hOverECut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure |1/E-1/p| cut
def psetEInerseMinusPInverseCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEInverseMinusPInverseCut'),
        eInverseMinusPInverseCutValueEB = cms.double( wpEB.absEInverseMinusPInverseCut ),
        eInverseMinusPInverseCutValueEE = cms.double( wpEE.absEInverseMinusPInverseCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure ECAL PF Cluster isolation cut. Note that this cut requires
# effective area constants file as input
def psetEcalPFClusterIsoCut(wpEB, wpEE, ecalIsoInputs):
    return cms.PSet( 
        cutName = cms.string('GsfEleCalPFClusterIsoCut'),
        isoType = cms.int32( 0 ), # ECAL = 0, HCAL = 1, see cut class header for IsoType enum
        isoCutEBLowPt  = cms.double( wpEB.ecalPFClusterIsoLowPtCut  ),
        isoCutEBHighPt = cms.double( wpEB.ecalPFClusterIsoHighPtCut ),
        isoCutEELowPt  = cms.double( wpEE.ecalPFClusterIsoLowPtCut  ),
        isoCutEEHighPt = cms.double( wpEE.ecalPFClusterIsoHighPtCut ),
        isRelativeIso = cms.bool(True),
        ptCutOff = cms.double(20.0),          # high pT above this value, low pT below
        barrelCutOff = cms.double(ebCutOff),
        rho = cms.InputTag("fixedGridRhoFastjetCentralCalo"), # This rho is best for emulation 
        # while HLT uses ...AllCalo
        effAreasConfigFile = cms.FileInPath( ecalIsoInputs.isoEffAreas ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False) 
        )

# Configure HCAL PF Cluster isolation cut. Note that this cut requires
# effective area constants file as input
def psetHcalPFClusterIsoCut(wpEB, wpEE, hcalIsoInputs):
    return cms.PSet( 
        cutName = cms.string('GsfEleCalPFClusterIsoCut'),
        isoType = cms.int32( 1 ), # ECAL = 0, HCAL = 1, see cut class header for IsoType enum
        isoCutEBLowPt  = cms.double( wpEB.hcalPFClusterIsoLowPtCut  ),
        isoCutEBHighPt = cms.double( wpEB.hcalPFClusterIsoHighPtCut ),
        isoCutEELowPt  = cms.double( wpEE.hcalPFClusterIsoLowPtCut  ),
        isoCutEEHighPt = cms.double( wpEE.hcalPFClusterIsoHighPtCut ),
        isRelativeIso = cms.bool(True),
        ptCutOff = cms.double(20.0),          # high pT above this value, low pT below
        barrelCutOff = cms.double(ebCutOff),
        rho = cms.InputTag("fixedGridRhoFastjetCentralCalo"), # This rho is best for emulation
        # while HLT uses ...AllCalo
        effAreasConfigFile = cms.FileInPath( hcalIsoInputs.isoEffAreas ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False) 
        )

# Configure tracker isolation cut
def psetTrkPtIsoCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleTrkPtIsoCut'),
        # Three constants for the GsfEleTrkPtIsoCut
        #     cut = constTerm if Et < slopeStart
        #     cut = slopeTerm * (Et - slopeStart) + constTerm if Et >= slopeStart
        slopeTermEB = cms.double( wpEB.trkIsoSlopeTerm ),
        slopeTermEE = cms.double( wpEE.trkIsoSlopeTerm ),
        slopeStartEB = cms.double( wpEB.trkIsoSlopeStart ),
        slopeStartEE = cms.double( wpEE.trkIsoSlopeStart ),
        constTermEB = cms.double( wpEB.trkIsoConstTerm ),
        constTermEE = cms.double( wpEE.trkIsoConstTerm ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure GsfTrack chi2/NDOF cut
def psetNormalizedGsfChi2Cut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleNormalizedGsfChi2Cut'),
        normalizedGsfChi2CutValueEB = cms.double( wpEB.normalizedGsfChi2Cut ),
        normalizedGsfChi2CutValueEE = cms.double( wpEE.normalizedGsfChi2Cut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

def psetEffAreaPFIsoCut(wpEB, wpEE, isoInputs):
    return cms.PSet(
        cutName = cms.string('GsfEleEffAreaPFIsoCut'),
        isoCutEBLowPt  = cms.double( wpEB.relCombIsolationWithEALowPtCut  ),
        isoCutEBHighPt = cms.double( wpEB.relCombIsolationWithEAHighPtCut ),
        isoCutEELowPt  = cms.double( wpEE.relCombIsolationWithEALowPtCut  ),
        isoCutEEHighPt = cms.double( wpEE.relCombIsolationWithEAHighPtCut ),
        isRelativeIso = cms.bool(True),
        ptCutOff = cms.double(20.0),          # high pT above this value, low pT below
        barrelCutOff = cms.double(ebCutOff),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.isoEffAreas ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )

def psetConversionVetoCut():
    return cms.PSet(
        cutName = cms.string('GsfEleConversionVetoCut'),
        conversionSrc        = cms.InputTag('allConversions'),
        conversionSrcMiniAOD = cms.InputTag('reducedEgamma:reducedConversions'),
        beamspotSrc = cms.InputTag('offlineBeamSpot'),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )

def psetMissingHitsCut(wpEB, wpEE):
    return cms.PSet(
        cutName = cms.string('GsfEleMissingHitsCut'),
        maxMissingHitsEB = cms.uint32( wpEB.missingHitsCut ),
        maxMissingHitsEE = cms.uint32( wpEE.missingHitsCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False) 
        )


# -----------------------------
# Version V1 common definitions
# -----------------------------

# This cut set definition is in the old style, with everything configured
# in one go. It is kept to minimize changes. New definitions should use
# PSets defined above instead.
def configureVIDCutBasedEleID_V1( wpEB, wpEE ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet(
            cms.PSet( cutName = cms.string("MinPtCut"),
                      minPt = cms.double(5.0),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)                ),
            cms.PSet( cutName = cms.string("GsfEleSCEtaMultiRangeCut"),
                      useAbsEta = cms.bool(True),
                      allowedEtaRanges = cms.VPSet( 
                    cms.PSet( minEta = cms.double(0.0), 
                              maxEta = cms.double(ebCutOff) ),
                    cms.PSet( minEta = cms.double(ebCutOff), 
                              maxEta = cms.double(2.5) )
                    ),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDEtaInCut'),
                      dEtaInCutValueEB = cms.double( wpEB.dEtaInCut ),
                      dEtaInCutValueEE = cms.double( wpEE.dEtaInCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDPhiInCut'),
                      dPhiInCutValueEB = cms.double( wpEB.dPhiInCut ),
                      dPhiInCutValueEE = cms.double( wpEE.dPhiInCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleFull5x5SigmaIEtaIEtaCut'),
                      full5x5SigmaIEtaIEtaCutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
                      full5x5SigmaIEtaIEtaCutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleHadronicOverEMCut'),
                      hadronicOverEMCutValueEB = cms.double( wpEB.hOverECut ),
                      hadronicOverEMCutValueEE = cms.double( wpEE.hOverECut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDxyCut'),
                      dxyCutValueEB = cms.double( wpEB.dxyCut ),
                      dxyCutValueEE = cms.double( wpEE.dxyCut ),
                      vertexSrc        = cms.InputTag("offlinePrimaryVertices"),
                      vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDzCut'),
                      dzCutValueEB = cms.double( wpEB.dzCut ),
                      dzCutValueEE = cms.double( wpEE.dzCut ),
                      vertexSrc        = cms.InputTag("offlinePrimaryVertices"),
                      vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleEInverseMinusPInverseCut'),
                      eInverseMinusPInverseCutValueEB = cms.double( wpEB.absEInverseMinusPInverseCut ),
                      eInverseMinusPInverseCutValueEE = cms.double( wpEE.absEInverseMinusPInverseCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDeltaBetaIsoCutStandalone'),
                      isoCutEBLowPt  = cms.double( wpEB.relCombIsolationWithDBetaLowPtCut  ),
                      isoCutEBHighPt = cms.double( wpEB.relCombIsolationWithDBetaHighPtCut ),
                      isoCutEELowPt  = cms.double( wpEE.relCombIsolationWithDBetaLowPtCut  ),
                      isoCutEEHighPt = cms.double( wpEE.relCombIsolationWithDBetaHighPtCut ),
                      isRelativeIso = cms.bool(True),
                      deltaBetaConstant = cms.double(0.5),
                      ptCutOff = cms.double(20.0),          # high pT above this value, low pT below
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleConversionVetoCut'),
                      conversionSrc        = cms.InputTag('allConversions'),
                      conversionSrcMiniAOD = cms.InputTag('reducedEgamma:reducedConversions'),
                      beamspotSrc = cms.InputTag('offlineBeamSpot'),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleMissingHitsCut'),
                      maxMissingHitsEB = cms.uint32( wpEB.missingHitsCut ),
                      maxMissingHitsEE = cms.uint32( wpEE.missingHitsCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False) ),
            )
        )
    #
    return parameterSet

# -----------------------------
# Version V2 common definitions
# -----------------------------

# This cut set definition is in the old style, with everything configured
# in one go. It is kept to minimize changes. New definitions should use
# PSets defined above instead.
def configureVIDCutBasedEleID_V2( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V2, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    The third argument is an object that contains information necessary
    for isolation calculations.
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet(
            cms.PSet( cutName = cms.string("MinPtCut"),
                      minPt = cms.double(5.0),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)                ),
            cms.PSet( cutName = cms.string("GsfEleSCEtaMultiRangeCut"),
                      useAbsEta = cms.bool(True),
                      allowedEtaRanges = cms.VPSet( 
                    cms.PSet( minEta = cms.double(0.0), 
                              maxEta = cms.double(ebCutOff) ),
                    cms.PSet( minEta = cms.double(ebCutOff), 
                              maxEta = cms.double(2.5) )
                    ),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDEtaInCut'),
                      dEtaInCutValueEB = cms.double( wpEB.dEtaInCut ),
                      dEtaInCutValueEE = cms.double( wpEE.dEtaInCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDPhiInCut'),
                      dPhiInCutValueEB = cms.double( wpEB.dPhiInCut ),
                      dPhiInCutValueEE = cms.double( wpEE.dPhiInCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleFull5x5SigmaIEtaIEtaCut'),
                      full5x5SigmaIEtaIEtaCutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
                      full5x5SigmaIEtaIEtaCutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleHadronicOverEMCut'),
                      hadronicOverEMCutValueEB = cms.double( wpEB.hOverECut ),
                      hadronicOverEMCutValueEE = cms.double( wpEE.hOverECut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDxyCut'),
                      dxyCutValueEB = cms.double( wpEB.dxyCut ),
                      dxyCutValueEE = cms.double( wpEE.dxyCut ),
                      vertexSrc        = cms.InputTag("offlinePrimaryVertices"),
                      vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleDzCut'),
                      dzCutValueEB = cms.double( wpEB.dzCut ),
                      dzCutValueEE = cms.double( wpEE.dzCut ),
                      vertexSrc        = cms.InputTag("offlinePrimaryVertices"),
                      vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleEInverseMinusPInverseCut'),
                      eInverseMinusPInverseCutValueEB = cms.double( wpEB.absEInverseMinusPInverseCut ),
                      eInverseMinusPInverseCutValueEE = cms.double( wpEE.absEInverseMinusPInverseCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleEffAreaPFIsoCut'),
                      isoCutEBLowPt  = cms.double( wpEB.relCombIsolationWithEALowPtCut  ),
                      isoCutEBHighPt = cms.double( wpEB.relCombIsolationWithEAHighPtCut ),
                      isoCutEELowPt  = cms.double( wpEE.relCombIsolationWithEALowPtCut  ),
                      isoCutEEHighPt = cms.double( wpEE.relCombIsolationWithEAHighPtCut ),
                      isRelativeIso = cms.bool(True),
                      ptCutOff = cms.double(20.0),          # high pT above this value, low pT below
                      barrelCutOff = cms.double(ebCutOff),
                      rho = cms.InputTag("fixedGridRhoFastjetAll"),
                      effAreasConfigFile = cms.FileInPath( isoInputs.isoEffAreas ),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False) ),
            cms.PSet( cutName = cms.string('GsfEleConversionVetoCut'),
                      conversionSrc        = cms.InputTag('allConversions'),
                      conversionSrcMiniAOD = cms.InputTag('reducedEgamma:reducedConversions'),
                      beamspotSrc = cms.InputTag('offlineBeamSpot'),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('GsfEleMissingHitsCut'),
                      maxMissingHitsEB = cms.uint32( wpEB.missingHitsCut ),
                      maxMissingHitsEE = cms.uint32( wpEE.missingHitsCut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False) ),
            )
        )
    #
    return parameterSet


# ==============================================================
# Define the complete cut sets
# ==============================================================

def configureVIDCutBasedEleID_V3( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V3, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    The third argument is an object that contains information necessary
    for isolation calculations.
        In this version, the impact parameter cuts dxy and dz are not present
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet(
            psetMinPtCut(),
            psetPhoSCEtaMultiRangeCut(),                        # eta cut
            psetDEtaInSeedCut(wpEB, wpEE),                      # dEtaIn seed cut
            psetDPhiInCut(wpEB, wpEE),                          # dPhiIn cut
            psetPhoFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),         # full 5x5 sigmaIEtaIEta cut
            psetHadronicOverEMCut(wpEB, wpEE),                  # H/E cut
            psetEInerseMinusPInverseCut(wpEB, wpEE),            # |1/e-1/p| cut
            psetEffAreaPFIsoCut(wpEB, wpEE, isoInputs),         # rel. comb. PF isolation cut
            psetConversionVetoCut(),
            psetMissingHitsCut(wpEB, wpEE)
            )
        )
    #
    return parameterSet


# -----------------------------
# HLT-safe common definitions
# -----------------------------


def configureVIDCutBasedEleHLTPreselection_V1( wpEB, wpEE, ecalIsoInputs, hcalIsoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type EleHLTSelection_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    The third and fourth arguments are objects that contain information necessary
    for isolation calculations for ECAL and HCAL.
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet = cms.PSet(
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet(
            psetMinPtCut(),                                     # min pt cut
            psetPhoSCEtaMultiRangeCut(),                        # eta cut
            psetPhoFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),         # full 5x5 sigmaIEtaIEta cut
            psetDEtaInSeedCut(wpEB, wpEE),                      # dEtaIn seed cut
            psetDPhiInCut(wpEB, wpEE),                          # dPhiIn cut
            psetHadronicOverEMCut(wpEB, wpEE),                  # H/E cut
            psetEInerseMinusPInverseCut(wpEB, wpEE),            # |1/e-1/p| cut
            psetEcalPFClusterIsoCut(wpEB, wpEE, ecalIsoInputs),  # ECAL PF Cluster isolation
            psetHcalPFClusterIsoCut(wpEB, wpEE, hcalIsoInputs),  # HCAL PF Cluster isolation
            psetTrkPtIsoCut(wpEB, wpEE),                        # tracker isolation cut
            psetNormalizedGsfChi2Cut(wpEB, wpEE)                # GsfTrack chi2/NDOF cut
            )
        )
    #
    return parameterSet

