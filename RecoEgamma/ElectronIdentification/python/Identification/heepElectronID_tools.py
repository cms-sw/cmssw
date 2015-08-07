import FWCore.ParameterSet.Config as cms

# Define eta constants
ebMax = 1.4442
eeMin = 1.566
ebCutOff=1.479


# ===============================================
# Define containers used by cut definitions
# ===============================================

class HEEP_WorkingPoint_V1:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts
    """
    def __init__(self, 
                 idName,
                 dEtaInSeedCut,
                 dPhiInCut,
                 full5x5SigmaIEtaIEtaCut,
                 # Two constants for the GsfEleFull5x5E2x5OverE5x5Cut
                 minE1x5OverE5x5Cut,
                 minE2x5OverE5x5Cut,
                 # Three constants for the GsfEleHadronicOverEMLinearCut:
                 #     cut = constTerm if value < slopeStart
                 #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
                 hOverESlopeTerm,
                 hOverESlopeStart,
                 hOverEConstTerm,
                 # Three constants for the GsfEleTrkPtIsoCut: 
                 #     cut = constTerm if value < slopeStart
                 #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
                 trkIsoSlopeTerm,
                 trkIsoSlopeStart,
                 trkIsoConstTerm,
                 # Three constants for the GsfEleEmHadD1IsoRhoCut: 
                 #     cut = constTerm if value < slopeStart
                 #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
                 # Also for the same cut, the effective area for the rho correction of the isolation
                 ehIsoSlopeTerm,
                 ehIsoSlopeStart,
                 ehIsoConstTerm,
                 effAreaForEHIso, 
                 # other cuts:
                 dxyCut,
                 maxMissingHitsCut,
                 ecalDrivenCut
                 ):
        # assign values taken from all the arguments above
        self.idName                  = idName
        self.dEtaInSeedCut           = dEtaInSeedCut          
        self.dPhiInCut               = dPhiInCut              
        self.full5x5SigmaIEtaIEtaCut = full5x5SigmaIEtaIEtaCut
        self.minE1x5OverE5x5Cut      = minE1x5OverE5x5Cut     
        self.minE2x5OverE5x5Cut      = minE2x5OverE5x5Cut     
        self.hOverESlopeTerm         = hOverESlopeTerm        
        self.hOverESlopeStart        = hOverESlopeStart       
        self.hOverEConstTerm         = hOverEConstTerm        
        self.trkIsoSlopeTerm         = trkIsoSlopeTerm        
        self.trkIsoSlopeStart        = trkIsoSlopeStart       
        self.trkIsoConstTerm         = trkIsoConstTerm        
        self.ehIsoSlopeTerm          = ehIsoSlopeTerm         
        self.ehIsoSlopeStart         = ehIsoSlopeStart        
        self.ehIsoConstTerm          = ehIsoConstTerm         
        self.effAreaForEHIso         = effAreaForEHIso
        self.dxyCut                  = dxyCut                 
        self.maxMissingHitsCut       = maxMissingHitsCut      
        self.ecalDrivenCut           = ecalDrivenCut
# ==============================================================
# Define individual cut configurations used by complete cut sets
# ==============================================================

# The mininum pt cut is set to 5 GeV
def psetMinPtCut():
    return cms.PSet( 
        cutName = cms.string("MinPtCut"),
        minPt = cms.double(35.0),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False) 
        )

# Take all particles in the eta ranges 0-ebMax and eeMin-2.5
def psetGsfEleSCEtaMultiRangeCut():
    return cms.PSet( 
        cutName = cms.string("GsfEleSCEtaMultiRangeCut"),
        useAbsEta = cms.bool(True),
        allowedEtaRanges = cms.VPSet( 
            cms.PSet( minEta = cms.double(0.0), 
                      maxEta = cms.double(ebMax) ),
            cms.PSet( minEta = cms.double(eeMin), 
                      maxEta = cms.double(2.5) )
            ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on the dEtaIn for the seed
def psetGsfEleDEtaInSeedCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleDEtaInSeedCut'),
        dEtaInSeedCutValueEB = cms.double( wpEB.dEtaInSeedCut ),
        dEtaInSeedCutValueEE = cms.double( wpEE.dEtaInSeedCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on the dPhiIn
def psetGsfEleDPhiInCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleDPhiInCut'),
        dPhiInCutValueEB = cms.double( wpEB.dPhiInCut ),
        dPhiInCutValueEE = cms.double( wpEE.dPhiInCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Confugure the full 5x5 sigmaIEtaIEta cut
def psetGsfEleFull5x5SigmaIEtaIEtaCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleFull5x5SigmaIEtaIEtaCut'),
        full5x5SigmaIEtaIEtaCutValueEB = cms.double( wpEB.full5x5SigmaIEtaIEtaCut ),
        full5x5SigmaIEtaIEtaCutValueEE = cms.double( wpEE.full5x5SigmaIEtaIEtaCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure XxX shower shape cuts
def psetGsfEleFull5x5E2x5OverE5x5Cut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleFull5x5E2x5OverE5x5Cut'),
        # E1x5 / E5x5
        minE1x5OverE5x5EB = cms.double( wpEB.minE1x5OverE5x5Cut ),
        minE1x5OverE5x5EE = cms.double( wpEE.minE1x5OverE5x5Cut ),
        # E2x5 / E5x5
        minE2x5OverE5x5EB = cms.double( wpEB.minE2x5OverE5x5Cut ),
        minE2x5OverE5x5EE = cms.double( wpEE.minE2x5OverE5x5Cut ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut of E/H
def psetGsfEleHadronicOverEMLinearCut(wpEB, wpEE) :
    return cms.PSet( 
        cutName = cms.string('GsfEleHadronicOverEMLinearCut'),
        # Three constants for the GsfEleHadronicOverEMLinearCut
        #     cut = constTerm if value < slopeStart
        #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
        slopeTermEB = cms.double( wpEB.hOverESlopeTerm ),
        slopeTermEE = cms.double( wpEE.hOverESlopeTerm ),
        slopeStartEB = cms.double( wpEB.hOverESlopeStart ),
        slopeStartEE = cms.double( wpEE.hOverESlopeStart ),
        constTermEB = cms.double( wpEB.hOverEConstTerm ),
        constTermEE = cms.double( wpEE.hOverEConstTerm ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on the tracker isolation
def psetGsfEleTrkPtIsoCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleTrkPtIsoCut'),
        # Three constants for the GsfEleTrkPtIsoCut
        #     cut = constTerm if value < slopeStart
        #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
        slopeTermEB = cms.double( wpEB.trkIsoSlopeTerm ),
        slopeTermEE = cms.double( wpEE.trkIsoSlopeTerm ),
        slopeStartEB = cms.double( wpEB.trkIsoSlopeStart ),
        slopeStartEE = cms.double( wpEE.trkIsoSlopeStart ),
        constTermEB = cms.double( wpEB.trkIsoConstTerm ),
        constTermEE = cms.double( wpEE.trkIsoConstTerm ),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on the EM + Had_depth_1 isolation with rho correction
def psetGsfEleEmHadD1IsoRhoCut(wpEB, wpEE):
    return cms.PSet(
        cutName = cms.string('GsfEleEmHadD1IsoRhoCut'),
        slopeTermEB = cms.double( wpEB.ehIsoSlopeTerm ),
        slopeTermEE = cms.double( wpEE.ehIsoSlopeTerm ),
        slopeStartEB = cms.double( wpEB.ehIsoSlopeStart ),
        slopeStartEE = cms.double( wpEE.ehIsoSlopeStart ),
        constTermEB = cms.double( wpEB.ehIsoConstTerm ),
        constTermEE = cms.double( wpEE.ehIsoConstTerm ),
        rhoConstant = cms.double( wpEB.effAreaForEHIso), # expected to be the same for EB and EE
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )

# Configure the dxy cut
def psetGsfEleDxyCut(wpEB, wpEE):
    return cms.PSet(
        cutName = cms.string('GsfEleDxyCut'),
        dxyCutValueEB = cms.double( wpEB.dxyCut ),
        dxyCutValueEE = cms.double( wpEE.dxyCut ),
        vertexSrc = cms.InputTag("offlinePrimaryVertices"),
        vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )

# Configure the cut on missing hits
def psetGsfEleMissingHitsCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleMissingHitsCut'),
        maxMissingHitsEB = cms.uint32( wpEB.maxMissingHitsCut ),
        maxMissingHitsEE = cms.uint32( wpEE.maxMissingHitsCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )
def psetGsfEleEcalDrivenCut(wpEB, wpEE):
    return cms.PSet(
        cutName = cms.string('GsfEleEcalDrivenCut'),
        ecalDrivenEB = cms.int32( wpEB.ecalDrivenCut ),
        ecalDrivenEE = cms.int32( wpEE.ecalDrivenCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        ) 

# ==============================================================
# Define the complete cut sets
# ==============================================================

def configureHEEPElectronID_V51(wpEB, wpEE):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type HEEP_WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    parameterSet = cms.PSet(
        idName = cms.string("heepElectronID-HEEPV51"),
        cutFlow = cms.VPSet(
            psetMinPtCut(),                               #0
            psetGsfEleSCEtaMultiRangeCut(),               #1
            psetGsfEleDEtaInSeedCut(wpEB,wpEE),           #2
            psetGsfEleDPhiInCut(wpEB,wpEE),               #3
            psetGsfEleFull5x5SigmaIEtaIEtaCut(wpEB,wpEE), #4
            psetGsfEleFull5x5E2x5OverE5x5Cut(wpEB,wpEE),  #5
            psetGsfEleHadronicOverEMLinearCut(wpEB,wpEE), #6 
            psetGsfEleTrkPtIsoCut(wpEB,wpEE),             #7
            psetGsfEleEmHadD1IsoRhoCut(wpEB,wpEE),        #8
            psetGsfEleDxyCut(wpEB,wpEE),                  #9
            psetGsfEleMissingHitsCut(wpEB,wpEE),          #10,
            psetGsfEleEcalDrivenCut(wpEB,wpEE)            #11
            )
        )
    return parameterSet

def configureHEEPElectronID_V60(wpEB, wpEE):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type HEEP_WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    parameterSet = cms.PSet(
        idName = cms.string("heepElectronID-HEEPV60"),
        cutFlow = cms.VPSet(
            psetMinPtCut(),                               #0
            psetGsfEleSCEtaMultiRangeCut(),               #1
            psetGsfEleDEtaInSeedCut(wpEB,wpEE),           #2
            psetGsfEleDPhiInCut(wpEB,wpEE),               #3
            psetGsfEleFull5x5SigmaIEtaIEtaCut(wpEB,wpEE), #4
            psetGsfEleFull5x5E2x5OverE5x5Cut(wpEB,wpEE),  #5
            psetGsfEleHadronicOverEMLinearCut(wpEB,wpEE), #6 
            psetGsfEleTrkPtIsoCut(wpEB,wpEE),             #7
            psetGsfEleEmHadD1IsoRhoCut(wpEB,wpEE),        #8
            psetGsfEleDxyCut(wpEB,wpEE),                  #9
            psetGsfEleMissingHitsCut(wpEB,wpEE),          #10,
            psetGsfEleEcalDrivenCut(wpEB,wpEE)            #11
            )
        )
    return parameterSet
