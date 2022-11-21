
import FWCore.ParameterSet.Config as cms

# Barrel/endcap division in eta
ebCutOff = 1.479

# ===============================================
# Define containers used by cut definitions
# ===============================================

class WorkingPoint_V1:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts
    """
    def __init__(self, 
                 idName,
                 hOverECut,
                 full5x5_sigmaIEtaIEtaCut,
                 # Isolation cuts are generally pt-dependent: cut = C1 + pt * C2
                 absPFChaHadIsoWithEACut_C1,
                 absPFChaHadIsoWithEACut_C2,
                 absPFNeuHadIsoWithEACut_C1,
                 absPFNeuHadIsoWithEACut_C2,
                 absPFPhoIsoWithEACut_C1,
                 absPFPhoIsoWithEACut_C2
                 ):
        self.idName    = idName
        self.hOverECut = hOverECut
        self.full5x5_sigmaIEtaIEtaCut = full5x5_sigmaIEtaIEtaCut
        self.absPFChaHadIsoWithEACut_C1  = absPFChaHadIsoWithEACut_C1 # charged hadron isolation C1
        self.absPFChaHadIsoWithEACut_C2  = absPFChaHadIsoWithEACut_C2 # ........ C2
        self.absPFNeuHadIsoWithEACut_C1  = absPFNeuHadIsoWithEACut_C1 # neutral hadron isolation C1
        self.absPFNeuHadIsoWithEACut_C2  = absPFNeuHadIsoWithEACut_C2 # ........ C2
        self.absPFPhoIsoWithEACut_C1     = absPFPhoIsoWithEACut_C1    # photon isolation C1
        self.absPFPhoIsoWithEACut_C2     = absPFPhoIsoWithEACut_C2    # ........ C2

class WorkingPoint_V2:
    """
    This is a container class to hold numerical cut values for either
    the barrel or endcap set of cuts
    This version of the container is different from the previous one
    by the fact that it contains three constants instead of two for
    the neutral hadron isolation cut, for exponantial parameterization
    """
    def __init__(self, 
                 idName,
                 hOverECut,
                 full5x5_sigmaIEtaIEtaCut,
                 # Isolation cuts are generally pt-dependent: cut = C1 + pt * C2
                 # except for the neutral hadron isolation where it is cut = C1+exp(pt*C2+C3)
                 absPFChaHadIsoWithEACut_C1,
                 absPFChaHadIsoWithEACut_C2,
                 absPFNeuHadIsoWithEACut_C1,
                 absPFNeuHadIsoWithEACut_C2,
                 absPFNeuHadIsoWithEACut_C3,
                 absPFPhoIsoWithEACut_C1,
                 absPFPhoIsoWithEACut_C2
                 ):
        self.idName    = idName
        self.hOverECut = hOverECut
        self.full5x5_sigmaIEtaIEtaCut = full5x5_sigmaIEtaIEtaCut
        self.absPFChaHadIsoWithEACut_C1  = absPFChaHadIsoWithEACut_C1 # charged hadron isolation C1
        self.absPFChaHadIsoWithEACut_C2  = absPFChaHadIsoWithEACut_C2 # ........ C2
        self.absPFNeuHadIsoWithEACut_C1  = absPFNeuHadIsoWithEACut_C1 # neutral hadron isolation C1
        self.absPFNeuHadIsoWithEACut_C2  = absPFNeuHadIsoWithEACut_C2 # ........ C2
        self.absPFNeuHadIsoWithEACut_C3  = absPFNeuHadIsoWithEACut_C3 # ........ C3
        self.absPFPhoIsoWithEACut_C1     = absPFPhoIsoWithEACut_C1    # photon isolation C1
        self.absPFPhoIsoWithEACut_C2     = absPFPhoIsoWithEACut_C2    # ........ C2


class WorkingPoint_V3:
    """
    This is a container class to hold the numerical cut values for either
    the barrel or endcap set of cuts.
    The changes made in this version:
    1) The charged hadron isolation and neutral hadron isolation are replaced
    by ECal cluster isolation and HCal cluster isolation.
    2) The isolation variables have pt-dependance: ECal cluster isolation has
    linear dependance and HCal cluster isolation has quadratic dependance
    3) Cone-based H/E (instead of tower-based, as done for Run2) is used, and
    PU-correction is done for it as well.
    """
    def __init__(self,
                 idName,
                 full5x5_sigmaIEtaIEtaCut,
                 hOverEwithEACut,
                 absPFChgHadIsoWithEACut_C1,
                 absPFChgHadIsoWithEACut_C2,
                 absPFECalClusIsoWithEACut_C1,
                 absPFECalClusIsoWithEACut_C2,
                 absPFHCalClusIsoWithEACut_C1,
                 absPFHCalClusIsoWithEACut_C2,
                 absPFHCalClusIsoWithEACut_C3
                 ):
                     self.idName = idName

                     self.full5x5_sigmaIEtaIEtaCut = full5x5_sigmaIEtaIEtaCut # sigmaIetaIeta
                     self.hOverEwithEACut          = hOverEwithEACut          # H/E (cone-based)

                     self.absPFChgHadIsoWithEACut_C1   = absPFChgHadIsoWithEACut_C1    # Charged hadron isolation (const. term)
                     self.absPFChgHadIsoWithEACut_C2   = absPFChgHadIsoWithEACut_C2    # Charged hadron isolation (linear term)
                     self.absPFECalClusIsoWithEACut_C1 = absPFECalClusIsoWithEACut_C1  # ECal Cluster isolation (const. term)
                     self.absPFECalClusIsoWithEACut_C2 = absPFECalClusIsoWithEACut_C1  # ECal Cluster isolation (linear term)
                     self.absPFHCalClusIsoWithEACut_C1 = absPFHCalClusIsoWithEACut_C1  # HCal Cluster isolation (const. term)
                     self.absPFHCalClusIsoWithEACut_C2 = absPFHCalClusIsoWithEACut_C2  # HCal Cluster isolation (linear term)
                     self.absPFHCalClusIsoWithEACut_C3 = absPFHCalClusIsoWithEACut_C3  # HCal Cluster isolation (quadratic term)


class IsolationCutInputs:
    """
    A container class that holds the names of the isolation maps in the event record
    and the names of the files with the effective area constants for pile-up corrections
    """
    def __init__(self, 
                 chHadIsolationMapName,
                 chHadIsolationEffAreas,
                 neuHadIsolationMapName,
                 neuHadIsolationEffAreas,
                 phoIsolationMapName,
                 phoIsolationEffAreas
                 ):
                 self.chHadIsolationMapName   = chHadIsolationMapName    
                 self.chHadIsolationEffAreas  = chHadIsolationEffAreas  
                 self.neuHadIsolationMapName  = neuHadIsolationMapName  
                 self.neuHadIsolationEffAreas = neuHadIsolationEffAreas 
                 self.phoIsolationMapName     = phoIsolationMapName     
                 self.phoIsolationEffAreas    = phoIsolationEffAreas    

class ClusterIsolationCutInputs:
    """
    A container class that holds the names of the cluster based isolation maps in the event record
    and the names of the files with the effective area constants for pile-up corrections
    """
    def __init__(self,
                 trkIsolationMapName,
                 trkIsolationEffAreas,
                 ecalClusIsolationMapName,
                 ecalClusIsolationEffAreas,
                 hcalClusIsolationMapName,
                 hcalClusIsolationEffAreas
                 ):
                 self.trkIsolationMapName       = trkIsolationMapName
                 self.trkIsolationEffAreas      = trkIsolationEffAreas
                 self.ecalClusIsolationMapName  = ecalClusIsolationMapName
                 self.ecalClusIsolationEffAreas = ecalClusIsolationEffAreas
                 self.hcalClusIsolationMapName  = hcalClusIsolationMapName
                 self.hcalClusIsolationEffAreas = hcalClusIsolationEffAreas


class HoverECutInputs:
    """
    A container class that holds the name of the file with the effective area constants
    for pile-up corrections
    """
    def __init__(self,
                 hOverEEffAreas
                 ):
                 self.hOverEEffAreas = hOverEEffAreas


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
        cutName = cms.string("PhoSCEtaMultiRangeCut"),
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

# Configure the cut on H/E
def psetPhoHcalOverEcalBcCut( wpEB, wpEE):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    """
    return cms.PSet( 
        cutName = cms.string('PhotonHcalOverEcalBcCut'),
        hcalOverEcalCutValueEB = cms.double( wpEB.hOverECut ),
        hcalOverEcalCutValueEE = cms.double( wpEE.hOverECut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on full5x5 sigmaIEtaIEta that uses a ValueMap,
# relying on an upstream producer that creates it. This was necessary
# for photons up to 7.2.0.
def psetPhoFull5x5SigmaIEtaIEtaValueMapCut(wpEB, wpEE):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    """
    return cms.PSet( 
        cutName = cms.string('PhoFull5x5SigmaIEtaIEtaValueMapCut'),
        cutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
        cutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
        full5x5SigmaIEtaIEtaMap = cms.InputTag('photonIDValueMapProducer:phoFull5x5SigmaIEtaIEta'),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False)
        )

# Configure the cut on full5x5 sigmaIEtaIEta that uses the native Photon field
# with this variable (works for releases past 7.2.0).
def psetPhoFull5x5SigmaIEtaIEtaCut(wpEB, wpEE):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    """
    return cms.PSet( 
        cutName = cms.string('PhoFull5x5SigmaIEtaIEtaCut'),
        cutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
        cutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
        barrelCutOff = cms.double(ebCutOff),
        needsAdditionalProducts = cms.bool(False),
        isIgnored = cms.bool(False)
        )

# Configure the cut on the charged hadron isolation that uses
# the linear pt scaling for barrel and endcap
def psetChHadIsoWithEALinScalingCut(wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    The third argument contains data for isolation calculation.
    The cut is (for lessThan==True), otherwise replace "<" with ">="
          X < constTerm + linearPtTerm*pt + quadPtTerm* pt*pt + rho*EA
    """
    return cms.PSet( 
        cutName = cms.string('PhoGenericRhoPtScaledCut'), 
        cutVariable = cms.string("chargedHadronIso"),
        lessThan = cms.bool(True),
        # cut 
        constTermEB = cms.double( wpEB.absPFChaHadIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFChaHadIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFChaHadIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFChaHadIsoWithEACut_C2 ),
        quadPtTermEB = cms.double( 0. ),
        quadPtTermEE = cms.double( 0. ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.chHadIsolationEffAreas ) 
        )

# Configure the cut on the neutral hadron isolation that uses
# the linear pt scaling for barrel and endcap
def psetNeuHadIsoWithEALinScalingCut( wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    The third argument contains data for isolation calculation.
    """
    return cms.PSet( 
        cutName = cms.string('PhoGenericRhoPtScaledCut'), 
        cutVariable = cms.string("neutralHadronIso"),
        lessThan = cms.bool(True),
        # cut 
        constTermEB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C2 ),
        quadPtTermEB = cms.double( 0. ),
        quadPtTermEE = cms.double( 0. ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.neuHadIsolationEffAreas ) 
        )

# Configure the cut on the neutral hadron isolation that uses
# the exponential pt scaling for barrel and the linear pt scaling for endcap
def psetNeuHadIsoWithEAExpoScalingEBCut(wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    The third argument contains data for isolation calculation.
    """
    return cms.PSet( 
        cutName = cms.string('PhoAnyPFIsoWithEAAndExpoScalingEBCut'), # Neutral hadrons isolation block
        # Barrel: cut = c1 + expo(pt*c2+c3)
        C1_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C1 ),
        C2_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C2 ),
        C3_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C3 ),
        # Endcap: cut = c1 + pt*c2
        C1_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C1 ),
        C2_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C2 ),
        anyPFIsoMap = cms.InputTag( isoInputs.neuHadIsolationMapName ),
        barrelCutOff = cms.double(ebCutOff),
        useRelativeIso = cms.bool(False),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.neuHadIsolationEffAreas ) 
        )

# Configure the cut on the neutral hadron isolation that uses
# the exponential pt scaling for both barrel and endcap
def psetNeuHadIsoWithEAExpoScalingCut(wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    The third argument contains data for isolation calculation.
    """
    return cms.PSet( 
        cutName = cms.string('PhoAnyPFIsoWithEAAndExpoScalingCut'), # Neutral hadrons isolation block
        # Barrel: cut = c1 + expo(pt*c2+c3)
        C1_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C1 ),
        C2_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C2 ),
        C3_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C3 ),
        # Endcap: cut = cut = c1 + expo(pt*c2+c3)
        C1_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C1 ),
        C2_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C2 ),
        C3_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C3 ),
        anyPFIsoMap = cms.InputTag( isoInputs.neuHadIsolationMapName ),
        barrelCutOff = cms.double(ebCutOff),
        useRelativeIso = cms.bool(False),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.neuHadIsolationEffAreas ) 
        )

# Configure the cut on the neutral hadron isolation that uses
# the quadratic polynomial pt scaling for both barrel and endcap
def psetNeuHadIsoWithEAQuadScalingCut(wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    The third argument contains data for isolation calculation.
    """
    return cms.PSet(
        cutName = cms.string('PhoGenericRhoPtScaledCut'), 
        cutVariable = cms.string("neutralHadronIso"),
        lessThan = cms.bool(True),
        # cut 
        constTermEB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C2 ),
        quadPtTermEB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C3 ),
        quadPtTermEE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C3 ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.neuHadIsolationEffAreas )
        )

# Configure the cut on the photon isolation that uses
# the linear pt scaling for barrel and endcap
def psetPhoIsoWithEALinScalingCut(wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_*
    The third argument contains data for isolation calculation.
    """
    return cms.PSet(
        cutName = cms.string('PhoGenericRhoPtScaledCut'), 
        cutVariable = cms.string("photonIso"),
        lessThan = cms.bool(True),
        # cut 
        constTermEB = cms.double( wpEB.absPFPhoIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFPhoIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFPhoIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFPhoIsoWithEACut_C2 ),
        quadPtTermEB = cms.double( 0. ),
        quadPtTermEE = cms.double( 0. ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        effAreasConfigFile = cms.FileInPath( isoInputs.phoIsolationEffAreas )
        )


# Configure the cut on H/E (cone-based)
def psetPhoHcalOverEcalWithEACut( wpEB, wpEE, hOverEInputs ):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_V3
    Third argument contains the effective areas for pile-up corrections.
    """
    return cms.PSet(
        cutName = cms.string('PhoGenericQuadraticRhoPtScaledCut'),
        cutVariable = cms.string("hcalOverEcal"),
        lessThan = cms.bool(True),
        # cut
        constTermEB = cms.double( wpEB.hOverEwithEACut ),
        constTermEE = cms.double( wpEE.hOverEwithEACut ),
        linearPtTermEB = cms.double( 0. ),
        linearPtTermEE = cms.double( 0. ),
        quadraticPtTermEB = cms.double( 0. ),
        quadraticPtTermEE = cms.double( 0. ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        quadEAflag = cms.bool(True),
        effAreasConfigFile = cms.FileInPath( hOverEInputs.hOverEEffAreas )
        )


# Configure the cut on the charged hadron isolation that uses
# the quadratic pt scaling for barrel and endcap
def psetChgHadIsoWithEAQuadScalingCut(wpEB, wpEE, isoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_V3
    The third argument contains data for isolation calculation.
    The cut is (for lessThan==True), otherwise replace "<" with ">="
          X < constTerm + linearPtTerm*pt + quadPtTerm* pt*pt + rho*EA
    """
    return cms.PSet(
        cutName = cms.string('PhoGenericQuadraticRhoPtScaledCut'),
        cutVariable = cms.string("chargedHadronIso"),
        lessThan = cms.bool(True),
        # cut
        constTermEB = cms.double( wpEB.absPFChgHadIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFChgHadIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFChgHadIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFChgHadIsoWithEACut_C2 ),
        quadraticPtTermEB = cms.double( 0. ),
        quadraticPtTermEE = cms.double( 0. ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        quadEAflag = cms.bool(True),
        effAreasConfigFile = cms.FileInPath( isoInputs.chHadIsolationEffAreas )
        )



# Configure the cut on the ECal cluster isolation that uses
# the quadratic pt scaling for barrel and endcap
def psetECalClusIsoWithEAQuadScalingCut(wpEB, wpEE, clusterIsoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_V3
    The third argument contains data for isolation calculation.
    """
    return cms.PSet(
        cutName = cms.string('PhoGenericQuadraticRhoPtScaledCut'),
        cutVariable = cms.string("ecalPFClusterIso"),
        lessThan = cms.bool(True),
        # cut
        constTermEB = cms.double( wpEB.absPFECalClusIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFECalClusIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFECalClusIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFECalClusIsoWithEACut_C2 ),
        quadraticPtTermEB = cms.double( 0. ),
        quadraticPtTermEE = cms.double( 0. ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        quadEAflag = cms.bool(True),
        effAreasConfigFile = cms.FileInPath( clusterIsoInputs.ecalClusIsolationEffAreas )
        )


# Configure the cut on the HCal cluster isolation that uses
# the quadratic polynomial pt scaling for both barrel and endcap
def psetHCalClusIsoWithEAQuadScalingCut(wpEB, wpEE, clusterIsoInputs):
    """
    Arguments: two containers of working point cut values of the type WorkingPoint_V3
    The third argument contains data for isolation calculation.
    """
    return cms.PSet(
        cutName = cms.string('PhoGenericQuadraticRhoPtScaledCut'),
        cutVariable = cms.string("hcalPFClusterIso"),
        lessThan = cms.bool(True),
        # cut
        constTermEB = cms.double( wpEB.absPFHCalClusIsoWithEACut_C1 ),
        constTermEE = cms.double( wpEE.absPFHCalClusIsoWithEACut_C1 ),
        linearPtTermEB = cms.double( wpEB.absPFHCalClusIsoWithEACut_C2 ),
        linearPtTermEE = cms.double( wpEE.absPFHCalClusIsoWithEACut_C2 ),
        quadraticPtTermEB = cms.double( wpEB.absPFHCalClusIsoWithEACut_C3 ),
        quadraticPtTermEE = cms.double( wpEE.absPFHCalClusIsoWithEACut_C3 ),
        needsAdditionalProducts = cms.bool(True),
        isIgnored = cms.bool(False),
        rho = cms.InputTag("fixedGridRhoFastjetAll"),
        quadEAflag = cms.bool(True),
        effAreasConfigFile = cms.FileInPath( clusterIsoInputs.hcalClusIsolationEffAreas )
        )



# ==============================================================
# Define the complete cut sets
# ==============================================================


def configureVIDCutBasedPhoID_V1( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    The third argument contains data for isolation calculation.
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet( 
            psetMinPtCut(),                                        # pt cut
            psetPhoSCEtaMultiRangeCut(),                           # eta cut
            psetPhoHcalOverEcalBcCut(wpEB,wpEE),                   # H/E cut
            psetPhoFull5x5SigmaIEtaIEtaValueMapCut(wpEB,wpEE),     # full 5x5 sigmaIEtaIEta cut
            psetChHadIsoWithEALinScalingCut(wpEB,wpEE,isoInputs),  # charged hadron isolation cut
            psetNeuHadIsoWithEALinScalingCut(wpEB,wpEE,isoInputs), # neutral hadron isolation cut
            psetPhoIsoWithEALinScalingCut(wpEB,wpEE,isoInputs)     # photon isolation cut
            )
        )
    #
    return parameterSet

def configureVIDCutBasedPhoID_V2( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: first object is of the type WorkingPoint_V2, second object
    is of the type WorkingPoint_V1, containing the cuts for the Barrel (EB) 
    and the other one for the Endcap (EE).
    The third argument contains data for isolation calculation.

    The V2 with respect to V1 has one change: the neutral hadron isolation
    cut has an exponential pt scaling for the barrel.
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet( 
            psetMinPtCut(),                                           # pt cut
            psetPhoSCEtaMultiRangeCut(),                              # eta cut
            psetPhoHcalOverEcalBcCut(wpEB,wpEE),                      # H/E cut
            psetPhoFull5x5SigmaIEtaIEtaValueMapCut(wpEB,wpEE),        # full 5x5 sigmaIEtaIEta cut
            psetChHadIsoWithEALinScalingCut(wpEB,wpEE,isoInputs),     # charged hadron isolation cut
            psetNeuHadIsoWithEAExpoScalingEBCut(wpEB,wpEE,isoInputs), # neutral hadron isolation cut
            psetPhoIsoWithEALinScalingCut(wpEB,wpEE,isoInputs)        # photon isolation cut
            )
        )
    #
    return parameterSet

def configureVIDCutBasedPhoID_V3( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: first object is of the type WorkingPoint_V2, second object
    is of the type WorkingPoint_V1, containing the cuts for the Barrel (EB) 
    and the other one for the Endcap (EE).
    The third argument contains data for isolation calculation.

    The V3 with respect to V2 has one change: the full5x5 sigmaIEtaIEta
    is taken from the native reco::Photon method and not from a ValueMap
    produced upstream by some producer module.
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet( 
            psetMinPtCut(),                                           # pt cut
            psetPhoSCEtaMultiRangeCut(),                              # eta cut
            psetPhoHcalOverEcalBcCut(wpEB,wpEE),                      # H/E cut
            psetPhoFull5x5SigmaIEtaIEtaCut(wpEB,wpEE),                # full 5x5 sigmaIEtaIEta cut
            psetChHadIsoWithEALinScalingCut(wpEB,wpEE,isoInputs),     # charged hadron isolation cut
            psetNeuHadIsoWithEAExpoScalingEBCut(wpEB,wpEE,isoInputs), # neutral hadron isolation cut
            psetPhoIsoWithEALinScalingCut(wpEB,wpEE,isoInputs)        # photon isolation cut
            )
        )
    #
    return parameterSet

def configureVIDCutBasedPhoID_V4( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: first object is of the type WorkingPoint_V2, second object
    is of the type WorkingPoint_V2 as well, first containing the cuts for the 
    Barrel (EB) and the other one for the Endcap (EE).
    The third argument contains data for isolation calculation.

    The V4 with respect to V3 has one change: both barrel and endcap
    use the exponential scaling for the neutral hadron isolation cut
    (in V3 it was only done for the barrel)
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet( 
            psetMinPtCut(),                                           # pt cut
            psetPhoSCEtaMultiRangeCut(),                              # eta cut
            psetPhoHcalOverEcalBcCut(wpEB,wpEE),                      # H/E cut
            psetPhoFull5x5SigmaIEtaIEtaCut(wpEB,wpEE),                # full 5x5 sigmaIEtaIEta cut
            psetChHadIsoWithEALinScalingCut(wpEB,wpEE,isoInputs),     # charged hadron isolation cut
            psetNeuHadIsoWithEAExpoScalingCut(wpEB,wpEE,isoInputs), # neutral hadron isolation cut
            psetPhoIsoWithEALinScalingCut(wpEB,wpEE,isoInputs)        # photon isolation cut
            )
        )
    #
    return parameterSet

def configureVIDCutBasedPhoID_V5( wpEB, wpEE, isoInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: first object is of the type WorkingPoint_V2, second object
    is of the type WorkingPoint_V2 as well, first containing the cuts for the 
    Barrel (EB) and the other one for the Endcap (EE).
    The third argument contains data for isolation calculation.

    The V5 with respect to V4 has one change: the neutral hadron isolation
    for both barrel and endcap now uses quadratic polynomial scaling.
    """
    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet( 
            psetMinPtCut(),                                           # pt cut
            psetPhoSCEtaMultiRangeCut(),                              # eta cut
            psetPhoHcalOverEcalBcCut(wpEB,wpEE),                      # H/E cut
            psetPhoFull5x5SigmaIEtaIEtaCut(wpEB,wpEE),                # full 5x5 sigmaIEtaIEta cut
            psetChHadIsoWithEALinScalingCut(wpEB,wpEE,isoInputs),     # charged hadron isolation cut
            psetNeuHadIsoWithEAQuadScalingCut(wpEB,wpEE,isoInputs),   # neutral hadron isolation cut
            psetPhoIsoWithEALinScalingCut(wpEB,wpEE,isoInputs)        # photon isolation cut
            )
        )
    #
    return parameterSet



def configureVIDCutBasedPhoID_V6( wpEB, wpEE, isoInputs, clusterIsoInputs, hOverEInputs ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: first object is of the type WorkingPoint_V3, second object
    is of the type WorkingPoint_V3 as well, first containing the cuts for the
    Barrel (EB) and the other one for the Endcap (EE).
    The third argument contains data for isolation calculation.

    The V6 with respect to V5 has following changes:
        1) H/E is now cone-based instead of tower-based.
        2) Neutral hadron isolation is replaced by HCal Cluster isolation
        for both barrel and endcap (and uses quadratic polynomial scaling
        as done before for neutral hadron isolation).
        3) Photon isolation is replaced by ECal Cluster isolation for both
        barrel and endcap (and uses linear polynomial scaling as done before
        for photon isolation).
    """

    # print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet(
            psetMinPtCut(),                                           # pt cut
            psetPhoSCEtaMultiRangeCut(),                              # eta cut
            psetPhoFull5x5SigmaIEtaIEtaCut(wpEB,wpEE),                # full 5x5 sigmaIEtaIEta cut
            psetPhoHcalOverEcalWithEACut(wpEB,wpEE,hOverEInputs),     # H/E cut
            psetChgHadIsoWithEAQuadScalingCut(wpEB,wpEE,isoInputs),          # charged hadron isolation cut
            psetECalClusIsoWithEAQuadScalingCut(wpEB,wpEE,clusterIsoInputs), # ecal cluster isolation cut
            psetHCalClusIsoWithEAQuadScalingCut(wpEB,wpEE,clusterIsoInputs)  # hcal cluster isolation cut
            )
        )
    #
    return parameterSet


