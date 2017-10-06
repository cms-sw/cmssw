from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import * 

muonSimClassifier = cms.EDProducer("MuonSimClassifier",
    muons = cms.InputTag("muons"),
    trackType = cms.string("glb_or_trk"),  # 'inner','outer','global','segments','glb_or_trk'
    trackingParticles = cms.InputTag("mix","MergedTrackTruth"), # default TrackingParticle collection (should exist in the Event)      
    associatorLabel   = cms.InputTag("muonAssociatorByHitsNoSimHitsHelper"),
    decayRho  = cms.double(200), # to classify differently decay muons included in ppMuX
    decayAbsZ = cms.double(400), # and decay muons that could not be in ppMuX
    linkToGenParticles = cms.bool(True),          # produce also a collection of GenParticles for secondary muons
    genParticles = cms.InputTag("genParticles"),  # and associations to primary and secondaries
)

muonSimClassificationByHitsSequence = cms.Sequence(
    muonAssociatorByHitsNoSimHitsHelper + muonSimClassifier
)

#def addUserData(patMuonProducer,labels=['classByHitsTM', 'classByHitsSta', 'classByHitsGlbOrTrk'], extraInfo = False):
# def addUserData(patMuonProducer,labels=['classByHitsGlbOrTrk'], extraInfo = False):
#     for label in labels:
#         patMuonProducer.userData.userInts.src.append( cms.InputTag(label) )
#         patMuonProducer.userData.userInts.src.append( cms.InputTag(label, "ext") )
#         if extraInfo:
#             for ints in ("flav", "hitsPdgId", "G4processType", "momPdgId", "gmomPdgId", "momFlav", "gmomFlav", "hmomFlav", "tpId", "tpBx", "tpEv", "momStatus"):
#                 patMuonProducer.userData.userInts.src.append(cms.InputTag(label, ints))
#             for ins in ("signp", "pt", "eta", "phi", "prodRho", "prodZ", "tpAssoQuality", "momRho", "momZ"):

#                 patMuonProducer.userData.userFloats.src.append(cms.InputTag(label, ins))

# def addGenParticleRef(patMuonProducer, label = 'classByHitsGlbOrTrk'):
#     patMuonProducer.addGenMatch = True
#     patMuonProducer.genParticleMatch = cms.VInputTag(cms.InputTag(label, "toPrimaries"), cms.InputTag(label, "toSecondaries"))
    
