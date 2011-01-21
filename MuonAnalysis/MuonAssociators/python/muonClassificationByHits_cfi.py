### Add MC classification by hits
# Requires:
#   SimGeneral/TrackingAnalysis V04-01-05    (35X+)
#   SimTracker/TrackAssociation V01-08-17    (35X+)
#   SimMuon/MCTruth             V02-05-00-03 (35X) or V02-06-00+ (37X+)

from SimGeneral.MixingModule.mixNoPU_cfi                          import *
from SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi   import * 
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi import * 

classByHitsTM = cms.EDProducer("MuonMCClassifier",
    muons = cms.InputTag("muons"),
    muonPreselection = cms.string("isTrackerMuon"),  #
    #muonPreselection = cms.string("muonID('TrackerMuonArbitrated')"), # You might want this
    trackType = cms.string("segments"),  # or 'inner','outer','global'
    trackingParticles = cms.InputTag("mergedtruthNoSimHits"),         
    associatorLabel   = cms.string("muonAssociatorByHits_NoSimHits"),
    decayRho  = cms.double(200), # to classifiy differently decay muons included in ppMuX
    decayAbsZ = cms.double(400), # and decay muons that could not be in ppMuX
    #makeGenParticles = cms.bool(False),           #   N O T
    #linkGenParticles = cms.bool(False),           #   Y E T
    #genParticles = cms.InputTag("genParticles"),  # T H E R E
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
    mix +
    trackingParticlesNoSimHits +
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
