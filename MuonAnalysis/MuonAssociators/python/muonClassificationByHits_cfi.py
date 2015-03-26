### Add MC classification by hits
# Requires:
#   SimGeneral/TrackingAnalysis V04-01-05    (35X+)
#   SimTracker/TrackAssociation V01-08-17    (35X+)
#   SimMuon/MCTruth             V02-05-00-03 (35X) or V02-06-00+ (37X+)

from SimGeneral.MixingModule.mixNoPU_cfi                          import *
trackingParticlesNoSimHits = mix.clone(
    digitizers = cms.PSet(
        mergedtruth = mix.digitizers.mergedtruth.clone(
            simHitCollections = cms.PSet(
                pixel = cms.VInputTag(),
                tracker = cms.VInputTag(),
                muon = cms.VInputTag(),
            )
        ),
    ),
    mixObjects = cms.PSet(
        mixHepMC    = mix.mixObjects.mixHepMC.clone(),
        mixVertices = mix.mixObjects.mixVertices.clone(),
        mixTracks   = mix.mixObjects.mixTracks.clone(),
    ),
)
from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import * 

classByHitsTM = cms.EDProducer("MuonMCClassifier",
    muons = cms.InputTag("muons"),
    muonPreselection = cms.string("isTrackerMuon"),  #
    #muonPreselection = cms.string("muonID('TrackerMuonArbitrated')"), # You might want this
    trackType = cms.string("segments"),  # or 'inner','outer','global'
    trackingParticles = cms.InputTag("trackingParticlesNoSimHits","MergedTrackTruth"),         
    associatorLabel   = cms.string("muonAssociatorByHitsNoSimHitsHelper"),
    decayRho  = cms.double(200), # to classifiy differently decay muons included in ppMuX
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


muonClassificationByHits = cms.Sequence(
    #mix +
    trackingParticlesNoSimHits +
    muonAssociatorByHitsNoSimHitsHelper +
    ( classByHitsTM      +
      classByHitsTMLSAT  +
      classByHitsGlb     +  
      classByHitsSta )
)
def addUserData(patMuonProducer,labels=['classByHitsGlb', 'classByHitsTM', 'classByHitsTMLSAT', 'classByHitsSta'], extraInfo = False):
    for label in labels:
        patMuonProducer.userData.userInts.src.append( cms.InputTag(label) )
        patMuonProducer.userData.userInts.src.append( cms.InputTag(label, "ext") )
        if extraInfo:
            for ints in ("flav", "hitsPdgId", "momPdgId", "gmomPdgId", "momFlav", "gmomFlav", "hmomFlav", "tpId", "momStatus"):
                patMuonProducer.userData.userInts.src.append(cms.InputTag(label, ints))
            for ins in ("prodRho", "prodZ", "tpAssoQuality", "momRho", "momZ"):
                patMuonProducer.userData.userFloats.src.append(cms.InputTag(label, ins))
def addGenParticleRef(patMuonProducer, label = 'classByHitsGlb'):
    patMuonProducer.addGenMatch = True
    patMuonProducer.genParticleMatch = cms.VInputTag(cms.InputTag(label, "toPrimaries"), cms.InputTag(label, "toSecondaries"))
    
