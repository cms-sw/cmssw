import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import eleIsoFromDepsEcalFromHits,eleIsoFromDepsHcalFromTowers,eleIsoFromDepsTk

allLayer1Electrons = cms.EDProducer("PATElectronProducer",
    # General configurables
    electronSource = cms.InputTag("allLayer0Electrons"),

                                    
    # user data to add
    userData = cms.PSet(
      # add custom classes here
      userClasses = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add doubles here
      userFloats = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add ints here
      userInts = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(""),
      userFunctionLabels = cms.vstring("")
    ),

    # Embedding of AOD items
    embedTrack        = cms.bool(False), ## whether to embed in AOD externally stored track (note: gsf electrons don't have a track)
    embedGsfTrack     = cms.bool(True), ## whether to embed in AOD externally stored gsf track
    embedSuperCluster = cms.bool(True), ## whether to embed in AOD externally stored supercluster

    # resolution configurables
    addResolutions   = cms.bool(False),

    # Store isolation values
    isolation = cms.PSet(
        tracker = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositTk"),
            # parameters to compute isolation (Egamma POG defaults)
            vetos  = eleIsoFromDepsTk.deposits[0].vetos,
            deltaR = eleIsoFromDepsTk.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True), # This overrides previous settings
#           # Or set your own vetos...
#            deltaR = cms.double(0.3),
#            vetos = cms.vstring('0.015', # inner radius veto cone
#                'Threshold(1.0)'),       # threshold on individual track pt
        ),
        ecal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits"),
            # parameters to compute isolation (Egamma POG defaults)
            vetos  = eleIsoFromDepsEcalFromHits.deposits[0].vetos,
            deltaR = eleIsoFromDepsEcalFromHits.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True), # This overrides previous settings
#           # Or set your own vetos...
#            deltaR = cms.double(0.4),
#            vetos = cms.vstring('EcalBarrel:0.040', 'EcalBarrel:RectangularEtaPhiVeto(-0.01,0.01,-0.5,0.5)',  # Barrel (|eta| < 1.479)
#                                'EcalEndcaps:0.070','EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'),
        ),
        hcal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositHcalFromTowers"),
            # parameters to compute isolation (Egamma POG defaults)
            vetos  = eleIsoFromDepsHcalFromTowers.deposits[0].vetos,
            deltaR = eleIsoFromDepsHcalFromTowers.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True),  # This overrides previous settings
#           # Or set your own vetos...
#            deltaR = cms.double(0.4),
        ),
        user = cms.VPSet(),
    ),
    # Store IsoDeposits
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("layer0ElectronIsolations","eleIsoDepositTk"),
        ecal    = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits"),
        hcal    = cms.InputTag("layer0ElectronIsolations","eleIsoDepositHcalFromTowers"),
    ),


    # electron ID configurables
    addElectronID = cms.bool(True),
    electronIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag>
        # you can comment out those you don't want to save some disk space
        eidRobustLoose = cms.InputTag("patElectronIds","eidRobustLoose"),
        eidRobustTight = cms.InputTag("patElectronIds","eidRobustTight"),
        eidLoose       = cms.InputTag("patElectronIds","eidLoose"),
        eidTight       = cms.InputTag("patElectronIds","eidTight"),
        eidRobustHighEnergy = cms.InputTag("patElectronIds","eidRobustHighEnergy"),
    ),

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    # trigger primitive sources to be used for the matching
    trigPrimMatch = cms.VInputTag(
            cms.InputTag("electronTrigMatchHLT1ElectronRelaxed"), 
            cms.InputTag("electronTrigMatchCandHLT1ElectronStartup")
    ),

    # MC matching configurables
    addGenMatch      = cms.bool(True),
    embedGenMatch    = cms.bool(False),
    genParticleMatch = cms.InputTag("electronMatch"), ## Association between electrons and generator particles

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),
    
    # electron cluster shape configurables
    addElectronShapes = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),

)


