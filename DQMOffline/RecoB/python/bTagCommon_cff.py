import FWCore.ParameterSet.Config as cms

bTagCommonBlock = cms.PSet(
    # rec. jet
    ptRecJetMin = cms.double(30.0),
    # This option enables/disables the output of the full list of histograms
    # With false, only a subset of the histograms (the most useful) will
    # be written to file.
    allHistograms = cms.bool(False),
    epsBaseName = cms.string(''),
    produceEps = cms.bool(False),
    etaMax = cms.double(2.4),
    # use pre-computed jet flavour identification
    # new default is to use genparticles - and it is the only option
#    fastMC = cms.bool(True),
    # Parameters which are common to all tagger algorithms
    # To add to an already existing set of plots new data, set update to true, and
    # inputfile to the file containing the existing histograms.
    # Beware that the all parameters (ranges, cuts, etc) have to be the same!
    update = cms.bool(False),
    psBaseName = cms.string(''),
    # eta
    etaMin = cms.double(0.0),
    # parton pt
    ptPartonMin = cms.double(0.0),
    # lepton momentum to jet energy ratio, if you use caloJets put ratioMin to -1.0 and ratioMax to 0.8
    ratioMin = cms.double(-9999.0),
    ratioMax = cms.double(9999.0),
    # CHOOSE, IF YOU WANT TO DEFINE THE PT/ETA BINS USING THE UNDERLYING PARTON OR
    # THE RECONSTRUCTED JET
    # BE CAREFUL CHOOSING THE PARTON KINEMATICS WHEN USING THE ALGORITHMIC DEFINITION
    partonKinematics = cms.bool(True),
    # Section for the jet flavour identification
    jetIdParameters = cms.PSet(
        vetoFlavour = cms.vstring(),
        rejectBCSplitting = cms.bool(False),
        physicsDefinition = cms.bool(False),
        coneSizeToAssociate = cms.double(0.3),
        fillLeptons = cms.bool(False),
        fillHeavyHadrons = cms.bool(False),
        fillPartons = cms.bool(True),
        mcSource = cms.string('source')
    ),
    jetMCSrc = cms.InputTag("mcJetFlavour"),
    softLeptonInfo = cms.InputTag("softPFElectronsTagInfos"),
    ptRanges = cms.vdouble(50.0, 80.0, 120.0),
    # eta and pt ranges
    etaRanges = cms.vdouble(0.0, 1.4, 2.4),
    ptRecJetMax = cms.double(40000.0),
    ptPartonMax = cms.double(99999.0),
    producePs = cms.bool(False),
    inputfile = cms.string(''),
    doJetID = cms.bool(False),
    doJEC = cms.bool(False),
    JECsource = cms.string("ak5PFCHSL1FastL2L3")
)


