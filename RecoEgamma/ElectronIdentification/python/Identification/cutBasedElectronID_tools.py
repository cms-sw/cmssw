
import FWCore.ParameterSet.Config as cms

# Barrel/endcap division in eta
ebCutOff = 1.479

# ===============================================
# Define containers used by cut definitions
# ===============================================


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

class EleWorkingPoint_V4:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts for electron cut-based ID
    With respect to V3, the hOverE cut is made energy and pileup dependent as presented in
    https://indico.cern.ch/event/662749/contributions/2763092/attachments/1545209/2425054/talk_electron_ID_2017.pdf
    """
    def __init__(self, 
                 idName,
                 dEtaInSeedCut,
                 dPhiInCut,
                 full5x5_sigmaIEtaIEtaCut,
                 hOverECut_C0,
                 hOverECut_CE,
                 hOverECut_Cr,
                 absEInverseMinusPInverseCut,
                 relCombIsolationWithEALowPtCut,
                 relCombIsolationWithEAHighPtCut,
                 # conversion veto cut needs no parameters, so not mentioned
                 missingHitsCut
                 ):
        self.idName                          = idName                       
        self.dEtaInSeedCut                   = dEtaInSeedCut                   
        self.dPhiInCut                       = dPhiInCut                   
        self.full5x5_sigmaIEtaIEtaCut        = full5x5_sigmaIEtaIEtaCut    
        self.hOverECut_C0                    = hOverECut_C0
        self.hOverECut_CE                    = hOverECut_CE
        self.hOverECut_Cr                    = hOverECut_Cr
        self.absEInverseMinusPInverseCut     = absEInverseMinusPInverseCut 
        self.relCombIsolationWithEALowPtCut  = relCombIsolationWithEALowPtCut
        self.relCombIsolationWithEAHighPtCut = relCombIsolationWithEAHighPtCut
        # conversion veto cut needs no parameters, so not mentioned
        self.missingHitsCut                  = missingHitsCut


class EleWorkingPoint_V5:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts for electron cut-based ID
    With respect to V4, the isolation cut is made pt dependent as presented in the following meeting: https://indico.cern.ch/event/697079/
    """
    def __init__(self,
                 idName,
                 dEtaInSeedCut,
                 dPhiInCut,
                 full5x5_sigmaIEtaIEtaCut,
                 hOverECut_C0,
                 hOverECut_CE,
                 hOverECut_Cr,
                 absEInverseMinusPInverseCut,
                 relCombIsolationWithEACut_C0,
                 relCombIsolationWithEACut_Cpt,
                 # conversion veto cut needs no parameters, so not mentioned
                 missingHitsCut
                 ):
        self.idName                          = idName
        self.dEtaInSeedCut                   = dEtaInSeedCut
        self.dPhiInCut                       = dPhiInCut
        self.full5x5_sigmaIEtaIEtaCut        = full5x5_sigmaIEtaIEtaCut
        self.hOverECut_C0                    = hOverECut_C0
        self.hOverECut_CE                    = hOverECut_CE
        self.hOverECut_Cr                    = hOverECut_Cr
        self.absEInverseMinusPInverseCut     = absEInverseMinusPInverseCut
        self.relCombIsolationWithEACut_C0    = relCombIsolationWithEACut_C0
        self.relCombIsolationWithEACut_Cpt   = relCombIsolationWithEACut_Cpt
        # conversion veto cut needs no parameters, so not mentioned
        self.missingHitsCut                  = missingHitsCut




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
def psetFull5x5SigmaIEtaIEtaCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string("full5x5_sigmaIetaIeta"),
        cutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
        cutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on dEta seed
def psetDEtaInSeedCut(wpEB, wpEE):
    valid_cut_condition = "? superCluster.isNonnull && superCluster.seed.isNonnull ?"
    actual_cut_string = "abs(deltaEtaSuperClusterTrackAtVtx - superCluster.eta + superCluster.seed.eta)"
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string(valid_cut_condition + actual_cut_string + " : 999999."),
        cutValueEB = cms.double( wpEB.dEtaInSeedCut ),
        cutValueEE = cms.double( wpEE.dEtaInSeedCut ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure dEtaIn cut
def psetDEtaInCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string("abs(deltaEtaSuperClusterTrackAtVtx)"),
        cutValueEB = cms.double( wpEB.dEtaInCut ),
        cutValueEE = cms.double( wpEE.dEtaInCut ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure dPhiIn cut
def psetDPhiInCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string("abs(deltaPhiSuperClusterTrackAtVtx)"),
        cutValueEB = cms.double( wpEB.dPhiInCut ),
        cutValueEE = cms.double( wpEE.dPhiInCut ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure H/E cut
def psetHadronicOverEMCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string("hadronicOverEm"),
        cutValueEB = cms.double( wpEB.hOverECut ),
        cutValueEE = cms.double( wpEE.hOverECut ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure energy and pileup dependent H/E cut
def psetHadronicOverEMEnergyScaledCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleHadronicOverEMEnergyScaledCut'),
        barrelC0 = cms.double( wpEB.hOverECut_C0 ),
        barrelCE = cms.double( wpEB.hOverECut_CE ),
        barrelCr = cms.double( wpEB.hOverECut_Cr ),
        endcapC0 = cms.double( wpEE.hOverECut_C0 ),
        endcapCE = cms.double( wpEE.hOverECut_CE ),
        endcapCr = cms.double( wpEE.hOverECut_Cr ),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )


# Configure |1/E-1/p| cut
def psetEInerseMinusPInverseCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string("abs(1. - eSuperClusterOverP) / ecalEnergy"),
        cutValueEB = cms.double( wpEB.absEInverseMinusPInverseCut ),
        cutValueEE = cms.double( wpEE.absEInverseMinusPInverseCut ),
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
        effAreasConfigFile = cms.FileInPath( ecalIsoInputs ),
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
        effAreasConfigFile = cms.FileInPath( hcalIsoInputs ),
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
        useHEEPIso = cms.bool(False),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure GsfTrack chi2/NDOF cut
def psetNormalizedGsfChi2Cut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleEBEECut'),
        cutString = cms.string("? gsfTrack.isNonnull ? gsfTrack.normalizedChi2 : 999990."),
        cutValueEB = cms.double( wpEB.normalizedGsfChi2Cut ),
        cutValueEE = cms.double( wpEE.normalizedGsfChi2Cut ),
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
        effAreasConfigFile = cms.FileInPath( isoInputs ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )

def psetRelPFIsoScaledCut(wpEB, wpEE, isoInputs):
    return cms.PSet(
        cutName                 = cms.string('GsfEleRelPFIsoScaledCut'),
        barrelC0                = cms.double(wpEB.relCombIsolationWithEACut_C0),
        endcapC0                = cms.double(wpEE.relCombIsolationWithEACut_C0),
        barrelCpt               = cms.double(wpEB.relCombIsolationWithEACut_Cpt),
        endcapCpt               = cms.double(wpEE.relCombIsolationWithEACut_Cpt),
        barrelCutOff            = cms.double(ebCutOff),
        rho                     = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile      = cms.FileInPath( isoInputs ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored               = cms.bool(False)
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

def psetGsfEleDxyCut(wpEB, wpEE):
    return cms.PSet( cutName = cms.string('GsfEleDxyCut'),
        dxyCutValueEB = cms.double( wpEB.dxyCut ),
        dxyCutValueEE = cms.double( wpEE.dxyCut ),
        vertexSrc        = cms.InputTag("offlinePrimaryVertices"),
        vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False))

def psetGsfEleDzCut(wpEB, wpEE):
    return cms.PSet( cutName = cms.string('GsfEleDzCut'),
        dzCutValueEB = cms.double( wpEB.dzCut ),
        dzCutValueEE = cms.double( wpEE.dzCut ),
        vertexSrc        = cms.InputTag("offlinePrimaryVertices"),
        vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False))

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
            psetMinPtCut(),
            psetPhoSCEtaMultiRangeCut(),
            psetDEtaInCut(wpEB, wpEE),
            psetDPhiInCut(wpEB, wpEE),
            psetFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),
            psetHadronicOverEMCut(wpEB, wpEE),
            psetGsfEleDxyCut(wpEB, wpEE),
            psetGsfEleDzCut(wpEB, wpEE),
            psetEInerseMinusPInverseCut(wpEB, wpEE),
            psetEffAreaPFIsoCut(wpEB, wpEE, isoInputs),
            psetConversionVetoCut(),
            psetMissingHitsCut(wpEB, wpEE)
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
            psetFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),         # full 5x5 sigmaIEtaIEta cut
            psetHadronicOverEMCut(wpEB, wpEE),                  # H/E cut
            psetEInerseMinusPInverseCut(wpEB, wpEE),            # |1/e-1/p| cut
            psetEffAreaPFIsoCut(wpEB, wpEE, isoInputs),         # rel. comb. PF isolation cut
            psetConversionVetoCut(),
            psetMissingHitsCut(wpEB, wpEE)
            )
        )
    #
    return parameterSet

def configureVIDCutBasedEleID_V4( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V3, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    The third argument is an object that contains information necessary
    for isolation calculations.
        In this version, the energy and pileup dependent hOverE is introduced
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
            psetFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),         # full 5x5 sigmaIEtaIEta cut
            psetHadronicOverEMEnergyScaledCut(wpEB, wpEE),      # H/E cut
            psetEInerseMinusPInverseCut(wpEB, wpEE),            # |1/e-1/p| cut
            psetEffAreaPFIsoCut(wpEB, wpEE, isoInputs),         # rel. comb. PF isolation cut
            psetConversionVetoCut(),
            psetMissingHitsCut(wpEB, wpEE)
            )
        )
    #
    return parameterSet

def configureVIDCutBasedEleID_V5( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V3, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    The third argument is an object that contains information necessary
    for isolation calculations.
        In this version, the pt dependent isolation is introduced
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
            psetFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),         # full 5x5 sigmaIEtaIEta cut
            psetHadronicOverEMEnergyScaledCut(wpEB, wpEE),      # H/E cut
            psetEInerseMinusPInverseCut(wpEB, wpEE),            # |1/e-1/p| cut
            psetRelPFIsoScaledCut(wpEB, wpEE, isoInputs),       # rel. comb. PF isolation cut
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
            psetFull5x5SigmaIEtaIEtaCut(wpEB, wpEE),         # full 5x5 sigmaIEtaIEta cut
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

