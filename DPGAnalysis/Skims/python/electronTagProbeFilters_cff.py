import FWCore.ParameterSet.Config as cms

#from HLTrigger.HLTfilters.hltHighLevel_cfi import *
#exoticaMuHLT = hltHighLevel
#Define the HLT path to be used.
#exoticaMuHLT.HLTPaths =['HLT_L1MuOpen']
#exoticaMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

#Define the HLT quality cut 
#exoticaHLTElectronFilter = cms.EDFilter("HLTSummaryFilter",
#    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
#    member  = cms.InputTag("hltL3ElectronCandidates","","HLT8E29"),      # filter or collection									
#    cut     = cms.string("pt>0"),                     # cut on trigger object
#    minN    = cms.int32(0)                  # min. # of passing objects needed
# )
                               

#Define the Reco quality cut
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Make the charged candidate collections from tracks
allElectronTracks = cms.EDProducer("TrackViewCandidateProducer",
                                   src = cms.InputTag("generalTracks"),
                                   particleType = cms.string('e-'),
                                   cut = cms.string('pt > 0'),
                                   filter = cms.bool(True)
                                   )

# Make the input candidate collections
electronTagCands = cms.EDFilter("GsfElectronRefSelector",
                        src = cms.InputTag("gedGsfElectrons"),
                        cut = cms.string('pt > 1.0 && abs(eta) < 2.1'),
                        filter = cms.bool(True)
                        )


# Tracker Electrons (to be matched)
electronProbeCands = cms.EDFilter("RecoChargedCandidateRefSelector",
                            src = cms.InputTag("allElectronTracks"),
                            cut = cms.string('pt > 0.5'),
                            filter = cms.bool(True)
                            )

# Make the tag probe association map
JPsiEETagProbeMap = cms.EDProducer("TagProbeMassProducer",
                                     MassMaxCut = cms.untracked.double(10.0),
                                     TagCollection = cms.InputTag("electronTagCands"),
                                     MassMinCut = cms.untracked.double(2.0),
                                     ProbeCollection = cms.InputTag("electronProbeCands"),
                                     PassingProbeCollection = cms.InputTag("electronProbeCands")
                                 )

JPsiEETPFilter = cms.EDFilter("TagProbeMassEDMFilter",
                        tpMapName = cms.string('JPsiEETagProbeMap')
                        )

ZEETagProbeMap = cms.EDProducer("TagProbeMassProducer",
                                 MassMaxCut = cms.untracked.double(120.0),
                                 TagCollection = cms.InputTag("electronTagCands"),
                                 MassMinCut = cms.untracked.double(50.0),
                                 ProbeCollection = cms.InputTag("electronProbeCands"),
                                 PassingProbeCollection = cms.InputTag("electronProbeCands")
                                 )

ZEETPFilter = cms.EDFilter("TagProbeMassEDMFilter",
                        tpMapName = cms.string('ZEETagProbeMap')
                        )


#Define group sequence, using HLT/Reco quality cut. 
#exoticaMuHLTQualitySeq = cms.Sequence()
electronTagProbeSeq = cms.Sequence(allElectronTracks+electronTagCands+electronProbeCands)

electronJPsiEERecoQualitySeq = cms.Sequence(
    #exoticaMuHLT+
    electronTagProbeSeq+JPsiEETagProbeMap+JPsiEETPFilter
    )

electronZEERecoQualitySeq = cms.Sequence(
    #exoticaMuHLT+
    electronTagProbeSeq+ZEETagProbeMap+ZEETPFilter
    )

