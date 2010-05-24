### Add MC classification by hits
# Requires:
#   SimGeneral/TrackingAnalysis V04-01-00-02 (35X) or V04-01-03+ (37X+)
#   SimTracker/TrackAssociation V01-08-17    (35X+)
#   SimMuon/MCTruth             V02-05-00-01 (35X) or V02-06-00+ (37X+)

from SimGeneral.MixingModule.mixNoPU_cfi                          import *
from SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi   import * 
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi import * 

classByHitsTM = cms.EDProducer("MuonMCClassifier",
    muons = cms.InputTag("muons"),
    trackType = cms.string("segments"),  # or 'inner','outer','global'
    trackingParticles = cms.InputTag("mergedtruthNoSimHits"),         # RECO Only
    associatorLabel   = cms.string("muonAssociatorByHits_NoSimHits"), # RECO Only
    #trackingParticles = cms.InputTag("mergedtruth"),                 # RAW+RECO
    #associatorLabel = cms.string("muonAssociatorByHits"),            # RAW+RECO
)
classByHitsGlb = classByHitsTM.clone(trackType = "global")

muonClassificationByHits = cms.Sequence(
    mix +
    trackingParticlesNoSimHits +
    ( classByHitsTM +
      classByHitsGlb  )
)
def addUserData(patMuonProducer,labels=['classByHitsGlb', 'classByHitsTM'], extraInfo = False):
    for label in labels:
        patMuonProducer.userData.userInts.src.append( cms.InputTag(label) )
        if extraInfo:
            for ins in ("flav", "hitsPdgId", "momPdgId", "gmomPdgId", "momFlav", "gmomFlav"):
                patMuonProducer.userData.userInts.src.append(cms.InputTag(label, ins))
            for ins in ("prodRho", "prodZ"):
                patMuonProducer.userData.userFloats.src.append(cms.InputTag(label, ins))


