import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry


ebMax = 1.4442
eeMin = 1.566
ebCutOff=1.479
heepElectronID_HEEPV51 = cms.PSet(
    idName = cms.string("heepElectronID-HEEPV51"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MinPtCut"),  #1
                  minPt = cms.double(35.0),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                ),
        cms.PSet( cutName = cms.string("GsfEleSCEtaMultiRangeCut"),#2
                  useAbsEta = cms.bool(True),
                  allowedEtaRanges = cms.VPSet( 
                                  cms.PSet( minEta = cms.double(0.0), 
                                            maxEta = cms.double(ebMax) ),
                                  cms.PSet( minEta = cms.double(eeMin), 
                                            maxEta = cms.double(2.5) )
                                  ),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleDEtaInSeedCut'),#3
                  dEtaInSeedCutValueEB = cms.double(0.004),
                  dEtaInSeedCutValueEE = cms.double(0.006),
                  barrelCutOff = cms.double(ebCutOff),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleDPhiInCut'),#4
                  dPhiInCutValueEB = cms.double(0.06),
                  dPhiInCutValueEE = cms.double(0.06),
                  barrelCutOff = cms.double(ebCutOff),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleFull5x5SigmaIEtaIEtaCut'),#5
                  full5x5SigmaIEtaIEtaCutValueEB = cms.double(9999),
                  full5x5SigmaIEtaIEtaCutValueEE = cms.double(0.03),
                  full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer","eleFull5x5SigmaIEtaIEta"),
                  barrelCutOff = cms.double(ebCutOff),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleFull5x5E2x5OverE5x5Cut'),#6
                  minE1x5OverE5x5EB = cms.double(0.83),
                  minE1x5OverE5x5EE = cms.double(-1),
                  minE2x5OverE5x5EB = cms.double(0.94),
                  minE2x5OverE5x5EE = cms.double(-1),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleHadronicOverEMLinearCut'),#7
                  slopeTermEB = cms.double(0.05),
                  slopeTermEE = cms.double(0.05),
                  slopeStartEB = cms.double(0),
                  slopeStartEE = cms.double(0),
                  constTermEB = cms.double(2),
                  constTermEE = cms.double(12.5),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleTrkPtIsoCut'),#8
                  slopeTermEB = cms.double(0),
                  slopeTermEE = cms.double(0),
                  slopeStartEB = cms.double(0),
                  slopeStartEE = cms.double(0),
                  constTermEB = cms.double(5),
                  constTermEE = cms.double(5),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleEmHadD1IsoRhoCut'),#9
                  slopeTermEB = cms.double(0.03),
                  slopeTermEE = cms.double(0.03),
                  slopeStartEB = cms.double(0),
                  slopeStartEE = cms.double(50),
                  constTermEB = cms.double(2),
                  constTermEE = cms.double(2.5),
                  rhoConstant = cms.double(0.28),
                  rho = cms.InputTag("fixedGridRhoFastjetAll"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False)),
    
        cms.PSet( cutName = cms.string('GsfEleDxyCut'),#10 so would be got to 9
                  dxyCutValueEB = cms.double(0.02),
                  dxyCutValueEE = cms.double(0.05),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  barrelCutOff = cms.double(ebCutOff),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False)),
        cms.PSet( cutName = cms.string('GsfEleMissingHitsCut'),#11
                  maxMissingHitsEB = cms.uint32(1),
                  maxMissingHitsEE = cms.uint32(1),
                  barrelCutOff = cms.double(ebCutOff),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)),
                
    )
)



def convertToMiniAOD(idPSet):
    idPSet.idName=idPSet.idName.value()+"-miniAOD"
    for pset in idPSet.cutFlow:
        if hasattr(pset,'vertexSrc'):
           pset.vertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices")
    
heepElectronID_HEEPV51_miniAOD = heepElectronID_HEEPV51.clone()
convertToMiniAOD(heepElectronID_HEEPV51_miniAOD)

central_id_registry.register(heepElectronID_HEEPV51.idName,"7abd8f20746894a7e01a89620d337aba")
central_id_registry.register(heepElectronID_HEEPV51_miniAOD.idName,"ba5810b8bac423cb6d84b964a6132232")
