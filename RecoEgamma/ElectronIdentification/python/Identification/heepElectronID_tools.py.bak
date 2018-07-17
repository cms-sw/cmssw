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

class HEEP_WorkingPoint_V2:
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
                 trkIsoRhoCorrStart,
                 trkIsoEffArea,
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
        self.trkIsoRhoCorrStart      = trkIsoRhoCorrStart
        self.trkIsoEffArea           = trkIsoEffArea
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

def psetGsfEleFull5x5SigmaIEtaIEtaWithSatCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleFull5x5SigmaIEtaIEtaWithSatCut'),
        maxSigmaIEtaIEtaEB = cms.double( wpEB.full5x5SigmaIEtaIEtaCut ),
        maxSigmaIEtaIEtaEE = cms.double( wpEE.full5x5SigmaIEtaIEtaCut ),
        maxNrSatCrysIn5x5EB =cms.int32( 0 ),
        maxNrSatCrysIn5x5EE =cms.int32( 0 ),
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
# Configure XxX shower shape cuts
def psetGsfEleFull5x5E2x5OverE5x5WithSatCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleFull5x5E2x5OverE5x5WithSatCut'),
        # E1x5 / E5x5
        minE1x5OverE5x5EB = cms.double( wpEB.minE1x5OverE5x5Cut ),
        minE1x5OverE5x5EE = cms.double( wpEE.minE1x5OverE5x5Cut ),
        # E2x5 / E5x5
        minE2x5OverE5x5EB = cms.double( wpEB.minE2x5OverE5x5Cut ),
        minE2x5OverE5x5EE = cms.double( wpEE.minE2x5OverE5x5Cut ),
        maxNrSatCrysIn5x5EB =cms.int32( 0 ),
        maxNrSatCrysIn5x5EE =cms.int32( 0 ),
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
# Configure the cut on the tracker isolation with a rho correction (hack for 76X)
def psetGsfEleTrkPtIsoRhoCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleTrkPtIsoRhoCut'),
        # Three constants for the GsfEleTrkPtIsoCut
        #     cut = constTerm if value < slopeStart
        #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
        slopeTermEB = cms.double( wpEB.trkIsoSlopeTerm ),
        slopeTermEE = cms.double( wpEE.trkIsoSlopeTerm ),
        slopeStartEB = cms.double( wpEB.trkIsoSlopeStart ),
        slopeStartEE = cms.double( wpEE.trkIsoSlopeStart ),
        constTermEB = cms.double( wpEB.trkIsoConstTerm ),
        constTermEE = cms.double( wpEE.trkIsoConstTerm ),
        rhoEtStartEB = cms.double( wpEB.trkIsoRhoCorrStart),
        rhoEtStartEE = cms.double( wpEE.trkIsoRhoCorrStart),
        rhoEAEB = cms.double( wpEB.trkIsoEffArea),
        rhoEAEE = cms.double( wpEE.trkIsoEffArea),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )
def psetGsfEleTrkPtNoJetCoreIsoCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleValueMapIsoRhoCut'),
        # Three constants for the GsfEleTrkPtIsoCut
        #     cut = constTerm if value < slopeStart
        #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
        slopeTermEB = cms.double( wpEB.trkIsoSlopeTerm ),
        slopeTermEE = cms.double( wpEE.trkIsoSlopeTerm ),
        slopeStartEB = cms.double( wpEB.trkIsoSlopeStart ),
        slopeStartEE = cms.double( wpEE.trkIsoSlopeStart ),
        constTermEB = cms.double( wpEB.trkIsoConstTerm ),
        constTermEE = cms.double( wpEE.trkIsoConstTerm ),
        #no rho so we zero it out, if the input tag is empty, its ignored anyways
        rhoEtStartEB = cms.double( 999999.),
        rhoEtStartEE = cms.double( 999999.),
        rhoEAEB = cms.double( 0. ),
        rhoEAEE = cms.double( 0. ),
        rho = cms.InputTag(""),
        value = cms.InputTag("heepIDVarValueMaps","eleTrkPtIsoNoJetCore"),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )
def psetGsfEleTrkPtFall16IsoCut(wpEB, wpEE):
    return cms.PSet( 
        cutName = cms.string('GsfEleValueMapIsoRhoCut'),
        # Three constants for the GsfEleTrkPtIsoCut
        #     cut = constTerm if value < slopeStart
        #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
        slopeTermEB = cms.double( wpEB.trkIsoSlopeTerm ),
        slopeTermEE = cms.double( wpEE.trkIsoSlopeTerm ),
        slopeStartEB = cms.double( wpEB.trkIsoSlopeStart ),
        slopeStartEE = cms.double( wpEE.trkIsoSlopeStart ),
        constTermEB = cms.double( wpEB.trkIsoConstTerm ),
        constTermEE = cms.double( wpEE.trkIsoConstTerm ),
        #no rho so we zero it out, if the input tag is empty, its ignored anyways
        rhoEtStartEB = cms.double( 999999.),
        rhoEtStartEE = cms.double( 999999.),
        rhoEAEB = cms.double( 0. ),
        rhoEAEE = cms.double( 0. ),
        rho = cms.InputTag(""),
        value = cms.InputTag("heepIDVarValueMaps","eleTrkPtIso"),
        needsAdditionalProducts = cms.bool(True),
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


def configureHEEPElectronID_V70(idName, wpEB, wpEE):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type HEEP_WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    parameterSet = cms.PSet(
        idName = cms.string(idName),
        cutFlow = cms.VPSet(
            psetMinPtCut(),                               #0
            psetGsfEleSCEtaMultiRangeCut(),               #1
            psetGsfEleDEtaInSeedCut(wpEB,wpEE),           #2
            psetGsfEleDPhiInCut(wpEB,wpEE),               #3
            psetGsfEleFull5x5SigmaIEtaIEtaWithSatCut(wpEB,wpEE), #4
            psetGsfEleFull5x5E2x5OverE5x5WithSatCut(wpEB,wpEE),  #5
            psetGsfEleHadronicOverEMLinearCut(wpEB,wpEE), #6 
            psetGsfEleTrkPtFall16IsoCut(wpEB,wpEE),    #7
            psetGsfEleEmHadD1IsoRhoCut(wpEB,wpEE),        #8
            psetGsfEleDxyCut(wpEB,wpEE),                  #9
            psetGsfEleMissingHitsCut(wpEB,wpEE),          #10,
            psetGsfEleEcalDrivenCut(wpEB,wpEE)            #11
            )
        )
    return parameterSet


def configureHEEPElectronID_V60_80XAOD(idName, wpEB, wpEE):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type HEEP_WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    parameterSet = cms.PSet(
        idName = cms.string(idName),
        cutFlow = cms.VPSet(
            psetMinPtCut(),                               #0
            psetGsfEleSCEtaMultiRangeCut(),               #1
            psetGsfEleDEtaInSeedCut(wpEB,wpEE),           #2
            psetGsfEleDPhiInCut(wpEB,wpEE),               #3
            psetGsfEleFull5x5SigmaIEtaIEtaWithSatCut(wpEB,wpEE), #4
            psetGsfEleFull5x5E2x5OverE5x5WithSatCut(wpEB,wpEE),  #5
            psetGsfEleHadronicOverEMLinearCut(wpEB,wpEE), #6 
            psetGsfEleTrkPtNoJetCoreIsoCut(wpEB,wpEE),    #7
            psetGsfEleEmHadD1IsoRhoCut(wpEB,wpEE),        #8
            psetGsfEleDxyCut(wpEB,wpEE),                  #9
            psetGsfEleMissingHitsCut(wpEB,wpEE),          #10,
            psetGsfEleEcalDrivenCut(wpEB,wpEE)            #11
            )
        )
    return parameterSet

def configureHEEPElectronID_V61(wpEB, wpEE):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type HEEP_WorkingPoint_V2, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    parameterSet = cms.PSet(
        idName = cms.string("heepElectronID-HEEPV61"),
        cutFlow = cms.VPSet(
            psetMinPtCut(),                               #0
            psetGsfEleSCEtaMultiRangeCut(),               #1
            psetGsfEleDEtaInSeedCut(wpEB,wpEE),           #2
            psetGsfEleDPhiInCut(wpEB,wpEE),               #3
            psetGsfEleFull5x5SigmaIEtaIEtaCut(wpEB,wpEE), #4
            psetGsfEleFull5x5E2x5OverE5x5Cut(wpEB,wpEE),  #5
            psetGsfEleHadronicOverEMLinearCut(wpEB,wpEE), #6 
            psetGsfEleTrkPtIsoRhoCut(wpEB,wpEE),          #7
            psetGsfEleEmHadD1IsoRhoCut(wpEB,wpEE),        #8
            psetGsfEleDxyCut(wpEB,wpEE),                  #9
            psetGsfEleMissingHitsCut(wpEB,wpEE),          #10,
            psetGsfEleEcalDrivenCut(wpEB,wpEE)            #11
            )
        )
    return parameterSet

def addHEEPProducersToSeq(process,seq,useMiniAOD, task=None):

    newTask = cms.Task()
    seq.associate(newTask)
    if task is not None:
        task.add(newTask)

    process.load("RecoEgamma.ElectronIdentification.heepIdVarValueMapProducer_cfi")
    newTask.add(process.heepIDVarValueMaps)

    if useMiniAOD==False:
        process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
        process.load("PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi")
        process.load("PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi")
        process.load("PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi") 
        from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import packedPFCandidates
        process.packedCandsForTkIso = packedPFCandidates.clone()
        process.packedCandsForTkIso.PuppiSrc=cms.InputTag("")
        process.packedCandsForTkIso.PuppiNoLepSrc=cms.InputTag("")
        
        process.load("PhysicsTools.PatAlgos.slimming.lostTracks_cfi")
        from PhysicsTools.PatAlgos.slimming.lostTracks_cfi import lostTracks
        process.lostTracksForTkIso = lostTracks.clone()
        process.lostTracksForTkIso.packedPFCandidates =cms.InputTag("packedCandsForTkIso")
        newTask.add(process.primaryVertexAssociation,
                    process.offlineSlimmedPrimaryVertices,
                    process.packedCandsForTkIso,
                    process.lostTracksForTkIso)
