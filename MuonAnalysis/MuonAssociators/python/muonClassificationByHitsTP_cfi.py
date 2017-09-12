### Add MC classification by hits
### Note: this cfi needs TrackingParticle collection from the Event
 
from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import * 

classByHitsTM = cms.EDProducer("MuonMCClassifier",
    muons = cms.InputTag("muons"),
    muonPreselection = cms.string("muonID('TrackerMuonArbitrated')"), # definition of "duplicates" depends on the preselection
    trackType = cms.string("segments"),  # 'inner','outer','global','segments','glb_or_trk'
    trackingParticles = cms.InputTag("mix","MergedTrackTruth"), # default TrackingParticle collection (should exist in the Event)      
    associatorLabel   = cms.InputTag("muonAssociatorByHitsNoSimHitsHelper"),
    decayRho  = cms.double(200), # to classify differently decay muons included in ppMuX
    decayAbsZ = cms.double(400), # and decay muons that could not be in ppMuX
    linkToGenParticles = cms.bool(True),          # produce also a collection of GenParticles for secondary muons
    genParticles = cms.InputTag("genParticles"),  # and associations to primary and secondaries
)
classByHitsTMLSAT = classByHitsTM.clone(
    muonPreselection = cms.string("muonID('TMLastStationAngTight')")
)
classByHitsGlb = classByHitsTM.clone(
    muonPreselection = cms.string("isGlobalMuon"),
    trackType = "global"
)
classByHitsSta = classByHitsTM.clone(
    muonPreselection = cms.string("isStandAloneMuon"),
    trackType = "outer"
)
classByHitsGlbOrTrk = classByHitsTM.clone(
    muonPreselection = cms.string("isGlobalMuon || muonID('TrackerMuonArbitrated')"),
    trackType = "glb_or_trk"
)


muonClassificationByHits = cms.Sequence(
    muonAssociatorByHitsNoSimHitsHelper +
    ( 
#      classByHitsTM      +
#      classByHitsTMLSAT  +
#      classByHitsGlb     +  
#      classByHitsSta     +
      classByHitsGlbOrTrk
    )
)
#def addUserData(patMuonProducer,labels=['classByHitsTM', 'classByHitsSta', 'classByHitsGlbOrTrk'], extraInfo = False):
def addUserData(patMuonProducer,labels=['classByHitsGlbOrTrk'], extraInfo = False):
    for label in labels:
        patMuonProducer.userData.userInts.src.append( cms.InputTag(label) )
        patMuonProducer.userData.userInts.src.append( cms.InputTag(label, "ext") )
        if extraInfo:
            for ints in ("flav", "hitsPdgId", "G4processType", "momPdgId", "gmomPdgId", "momFlav", "gmomFlav", "hmomFlav", "tpId", "tpBx", "tpEv", "momStatus"):
                patMuonProducer.userData.userInts.src.append(cms.InputTag(label, ints))
            for ins in ("signp", "pt", "eta", "phi", "prodRho", "prodZ", "tpAssoQuality", "momRho", "momZ"):

                patMuonProducer.userData.userFloats.src.append(cms.InputTag(label, ins))

def addGenParticleRef(patMuonProducer, label = 'classByHitsGlbOrTrk'):
    patMuonProducer.addGenMatch = True
    patMuonProducer.genParticleMatch = cms.VInputTag(cms.InputTag(label, "toPrimaries"), cms.InputTag(label, "toSecondaries"))
    
