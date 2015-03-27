
import FWCore.ParameterSet.Config as cms

ebCutOff = 1.479

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

def configureVIDCutBasedPhoID_V1( wpEB, wpEE ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: two objects of the type WorkingPoint_V1, one
    containing the cuts for the Barrel (EB) and the other one for the Endcap (EE).
    """
    print "VID: Configuring cut set %s" % wpEB.idName
    parameterSet =  cms.PSet(
        #
        idName = cms.string( wpEB.idName ), # same name stored in the _EB and _EE objects
        cutFlow = cms.VPSet(
            cms.PSet( cutName = cms.string("MinPtCut"),
                      minPt = cms.double(5.0),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)                ),
            cms.PSet( cutName = cms.string("PhoSCEtaMultiRangeCut"),
                      useAbsEta = cms.bool(True),
                      allowedEtaRanges = cms.VPSet( 
                    cms.PSet( minEta = cms.double(0.0), 
                              maxEta = cms.double(ebCutOff) ),
                    cms.PSet( minEta = cms.double(ebCutOff), 
                              maxEta = cms.double(2.5) )
                    ),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('PhoFull5x5SigmaIEtaIEtaCut'),
                      full5x5SigmaIEtaIEtaCutValueEB = cms.double( wpEB.full5x5_sigmaIEtaIEtaCut ),
                      full5x5SigmaIEtaIEtaCutValueEE = cms.double( wpEE.full5x5_sigmaIEtaIEtaCut ),
                      full5x5SigmaIEtaIEtaMap = cms.InputTag('photonIDValueMapProducer:phoFull5x5SigmaIEtaIEta'),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)),
            cms.PSet( cutName = cms.string('PhoHadronicOverEMCut'),
                      hadronicOverEMCutValueEB = cms.double( wpEB.hOverECut ),
                      hadronicOverEMCutValueEE = cms.double( wpEE.hOverECut ),
                      barrelCutOff = cms.double(ebCutOff),
                      needsAdditionalProducts = cms.bool(False),
                      isIgnored = cms.bool(False))
            #cms.PSet( cutName = cms.string('PhoAnyPFIsoWithEACut'),
            #          anyPFIsoWithEACutValue_C1_EB = cms.double( wpEB.absPFChaHadIsoWithEACut_C1 ),
            #          anyPFIsoWithEACutValue_C2_EB = cms.double( wpEB.absPFChaHadIsoWithEACut_C2 ),
            #          anyPFIsoWithEACutValue_C1_EE = cms.double( wpEE.absPFChaHadIsoWithEACut_C1 ),
            #          anyPFIsoWithEACutValue_C2_EE = cms.double( wpEE.absPFChaHadIsoWithEACut_C2 ),
            #          anyPFIsoWithEAMap = cms.InputTag('photonIDValueMapProducer:phoChargedIsolation'),
            #          barrelCutOff = cms.double(ebCutOff),
            #          useRelativeIso = cms.bool(False),
            #          needsAdditionalProducts = cms.bool(True),
            #          isIgnored = cms.bool(False)),
            #cms.PSet( cutName = cms.string('PhoAnyPFIsoWithEACut'),
            #          anyPFIsoWithEACutValue_C1_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C1 ),
            #          anyPFIsoWithEACutValue_C2_EB = cms.double( wpEB.absPFNeuHadIsoWithEACut_C2 ),
            #          anyPFIsoWithEACutValue_C1_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C1 ),
            #          anyPFIsoWithEACutValue_C2_EE = cms.double( wpEE.absPFNeuHadIsoWithEACut_C2 ),
            #          anyPFIsoWithEAMap = cms.InputTag('photonIDValueMapProducer:phoNeutralHadronIsolation'),
            #          barrelCutOff = cms.double(ebCutOff),
            #          useRelativeIso = cms.bool(False),
            #          needsAdditionalProducts = cms.bool(True),
            #          isIgnored = cms.bool(False)),
            #cms.PSet( cutName = cms.string('PhoAnyPFIsoWithEACut'),
            #          anyPFIsoWithEACutValue_C1_EB = cms.double( wpEB.absPFPhoIsoWithEACut_C1 ),
            #          anyPFIsoWithEACutValue_C2_EB = cms.double( wpEB.absPFPhoIsoWithEACut_C2 ),
            #          anyPFIsoWithEACutValue_C1_EE = cms.double( wpEE.absPFPhoIsoWithEACut_C1 ),
            #          anyPFIsoWithEACutValue_C2_EE = cms.double( wpEE.absPFPhoIsoWithEACut_C2 ),
            #          anyPFIsoWithEAMap = cms.InputTag('photonIDValueMapProducer:phoPhotonIsolation'),
            #          barrelCutOff = cms.double(ebCutOff),
            #          useRelativeIso = cms.bool(False),
            #          needsAdditionalProducts = cms.bool(True),
            #          isIgnored = cms.bool(False))
            )
        )
    #
    return parameterSet

