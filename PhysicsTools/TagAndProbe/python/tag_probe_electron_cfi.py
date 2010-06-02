import FWCore.ParameterSet.Config as cms

# 
#  GsfElectrons  ################
# 
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

from RecoEgamma.EgammaHFProducers.hfEMClusteringSequence_cff import *

from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *

from Geometry.CaloEventSetup.CaloGeometry_cff import *

from Configuration.EventContent.EventContent_cff import *

#from RecoVertex.BeamSpotProducer.BeamSpot_cff import *


hfRecoEcalCandidate.Correct = True
##hfRecoEcalCandidate.e9e25Cut = 0
##hfRecoEcalCandidate.intercept2DCut = -99

hfSuperClusterCandidate = hfRecoEcalCandidate.clone()
hfSuperClusterCandidate.e9e25Cut = 0
hfSuperClusterCandidate.intercept2DCut = -99

# 
#  Calculate efficiency for *SuperClusters passing as GsfElectron* 
#
#  Tag           =  isolated GsfElectron with Robust ID, passing HLT, and  
#                    within the fiducial volume of ECAL
#  Probe --> Passing Probe   =  
#  SC --> GsfElectron --> isolation --> id --> Trigger
#
#
#  SuperClusters  ################
# 
################################################
# Direct HF SC's taken out as they do not start with EM calibration but pion calibration
# We need to use the HF electron ID class with no cuts to get the proper calibrations
# that requirement will be removed in the future
################################################
#HFSuperClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
#    src = cms.InputTag("hfEMClusters"),
#    particleType = cms.string('gamma')
#)
#HFSuperClusters = cms.EDFilter("CandViewSelector",
#    src = cms.InputTag("HFSuperClusterCands"),
#    cut = cms.string('abs( eta ) > 3.0')
#)
#theHFSuperClusters = cms.EDFilter("CandViewSelector",
#    src = cms.InputTag("HFSuperClusters"),
#    cut = cms.string('et > 10.0')
#)

theHFSuperClusters = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("hfSuperClusterCandidate"),
    cut = cms.string('et > 10.0')
)


superClusters = cms.EDFilter("SuperClusterMerger",
   src = cms.VInputTag(cms.InputTag("hybridSuperClusters","", "RECO"),
                       cms.InputTag("multi5x5SuperClustersWithPreshower","", "RECO"),
                       #cms.InputTag("theHFSuperClusters","", "RECO")
                       ) 
)


theSuperClusters = cms.EDProducer("ConcreteEcalCandidateProducer",
   src = cms.InputTag("superClusters"),
   particleType = cms.int32(11),
   cut = cms.string("((energy)*sin(position.theta)>20.0)"
                    " && (abs(eta)<2.5) && !(1.4442<abs(eta)<1.560)"
                    )
)

###Old sequence to be used again when HF calibration is in the corrected SuperCluster Step.
#sc_sequence = cms.Sequence( HFSuperClusterCands * HFSuperClusters *
#theHFSuperClusters *superClusters * theSuperClusters)

sc_sequence = cms.Sequence( theHFSuperClusters * superClusters * theSuperClusters)




#  GsfElectron within acceptance: no other cuts
theGsfElectrons = cms.EDFilter("gsfElectronSelector",
    src = cms.untracked.string('gsfElectrons'),
    etMin = cms.untracked.double(20.0)
)


#HFElectronIDCands = cms.EDProducer("ConcreteEcalCandidateProducer",
#    src = cms.InputTag("hfRecoEcalCandidate"),
#    particleType = cms.string('gamma')
#)

HFElectronID = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("hfRecoEcalCandidate"),
    cut = cms.string('et > 10.0')
)


## #  isolation  ################
theIsolation = cms.EDFilter("gsfElectronSelector",
    src = cms.untracked.string('gsfElectrons'),
    etMin = cms.untracked.double(20.0),
    requireTkIso = cms.untracked.bool(True),                
    tkIsoCutBarrel = cms.untracked.double(7.2),
    tkIsoCutEndcaps = cms.untracked.double(5.1)
)


## # Cut-based Robust electron ID  ######
from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *
import RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi
eidRobust = RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi.eidCutBasedExt.clone()
eidRobust.src = cms.InputTag('theIsolation')

theId = cms.EDProducer("eidCandProducer",
    electronCollection = cms.untracked.InputTag('theIsolation'),  
    electronLabelLoose = cms.InputTag('eidRobust')
)



# Trigger  ##################
theHLT = cms.EDProducer("trgMatchedGsfElectronProducer",
    InputProducer = cms.InputTag('theId'),
    hltTag = cms.untracked.InputTag("HLT_Ele15_SW_LooseTrackIso_L1R","","HLT")
)

electron_sequence = cms.Sequence(theGsfElectrons * theIsolation *
                                 eidRobust * theId * theHLT * HFElectronID )



# 
#  All Tag / Probe Association Maps  ###############
# 
# Remember that tag will always be "theHLT" collection.
#
# Probe can be one of the following collections: 
# "theSuperClusters", "theGsfElectrons", "theIsolation", "theId".
# 
# Passing Probe can be one of the following collections: 
# "theGsfElectrons", "theIsolation", "theId", "theHLT".
#

tpMapSuperClusters = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(60.0),
    ProbeCollection = cms.InputTag("theSuperClusters")
)

tpMapGsfElectrons = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(60.0),
    ProbeCollection = cms.InputTag("theGsfElectrons")
)

tpMapIsolation = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(60.0),
    ProbeCollection = cms.InputTag("theIsolation")
)

tpMapId = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("theHLT"),
    MassMinCut = cms.untracked.double(60.0),
    ProbeCollection = cms.InputTag("theId")
)

tpMapHFSuperClusters = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("theHLT"),         
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("theHFSuperClusters")
)

tpMap_sequence = cms.Sequence(tpMapSuperClusters + tpMapGsfElectrons +
                              tpMapIsolation + tpMapId + tpMapHFSuperClusters)




# 
#  All Truth-matched collections  ###################
# 
# find generator particles matching by DeltaR


SuperClustersMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theSuperClusters"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

GsfElectronsMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theGsfElectrons"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

IsolationMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theIsolation"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

IdMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theId"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

HLTMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theHLT"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

HFSCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("theHFSuperClusters"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

HFIDMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(11),
    src = cms.InputTag("HFElectronID"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)



truthMatch_sequence = cms.Sequence(SuperClustersMatch + GsfElectronsMatch +
                                   IsolationMatch + IdMatch + HLTMatch +
                                   HFSCMatch + HFIDMatch)


lepton_cands = cms.Sequence(genParticles * hfEMClusteringSequence *
                            sc_sequence * electron_sequence *
                            tpMap_sequence * truthMatch_sequence)

