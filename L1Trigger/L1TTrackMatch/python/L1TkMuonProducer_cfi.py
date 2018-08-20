import FWCore.ParameterSet.Config as cms

L1TkMuons = cms.EDProducer("L1TkMuonProducer",
    L1BMTFInputTag  = cms.InputTag("simBmtfDigis","BMTF"),
    L1OMTFInputTag  = cms.InputTag("simOmtfDigis","OMTF"),
    L1EMTFInputTag  = cms.InputTag("simEmtfDigis","EMTF"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
    ETAMIN = cms.double(0),
    ETAMAX = cms.double(5.),        # no cut
    ZMAX = cms.double( 25. ),       # in cm
    CHI2MAX = cms.double( 100. ),
    PTMINTRA = cms.double( 2. ),    # in GeV
    DRmax = cms.double( 0.5 ),
    nStubsmin = cms.int32( 3 ),        # minimum number of stubs
#    closest = cms.bool( True ),
    emtfMatchAlgoVersion = cms.int32( 1 ),        # version of matching EMTF with Trackes (1 or 2)
    correctGMTPropForTkZ = cms.bool(True),
    use5ParameterFit = cms.bool(False) #use 4-pars by defaults

)


