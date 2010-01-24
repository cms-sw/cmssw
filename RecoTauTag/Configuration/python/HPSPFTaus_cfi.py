import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.HPSPFRecoTauProducer_cfi import *

hpsPFTauProducer = copy.deepcopy(hpsPFRecoTauProducer)


# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

#Discriminator By Decay Mode Finding
#OK After discussing we decided that if there is no decay mode reconstructed
#we create a PF tau with the LV of the PF Jet but no refs.
#So if a lead track exists the decay  mode has been reconstructed
#Therefore we call a discriminator byDecayModeFinding but in the reality
#it is a leading track finding discriminator

hpsPFTauDiscriminationByDecayModeFinding             = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
setTauSource(hpsPFTauDiscriminationByDecayModeFinding, 'hpsPFTauProducer')




requireDecayMode = cms.PSet(
          BooleanOperator = cms.string("and"),
                decayMode = cms.PSet(
                    Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
                    cut = cms.double(0.5)
                      )
)


#copying the Discriminator by Isolation


hpsPFTauDiscriminationByLooseIsolation                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByLooseIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants = requireDecayMode



#Define a discriminator By Medium Isolation!
#You need to loosen qualityCuts for this
mediumPFTauQualityCuts = cms.PSet(
      signalQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.8),  # filter PFChargedHadrons below given pt
         maxTrackChi2                 = cms.double(100.), # require track Chi2
         maxTransverseImpactParameter = cms.double(0.03), # w.r.t. PV
         maxDeltaZ                    = cms.double(0.2),  # w.r.t. PV
         minTrackPixelHits            = cms.uint32(0),    # pixel-only hits (note that these cuts are turned off, 
                                                          # the tracking cuts might be higher)
         minTrackHits                 = cms.uint32(3),    # total track hits
         minGammaEt                   = cms.double(0.5),  # filter PFgammas below given Pt
         useTracksInsteadOfPFHadrons  = cms.bool(False),  # if true, use generalTracks, instead of PFChargedHadrons
      ),
      isolationQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.8), 
         maxTrackChi2                 = cms.double(100.),
         maxTransverseImpactParameter = cms.double(0.03),
         maxDeltaZ                    = cms.double(0.2),
         minTrackPixelHits            = cms.uint32(0),
         minTrackHits                 = cms.uint32(3),
         minGammaEt                   = cms.double(0.8),
         useTracksInsteadOfPFHadrons  = cms.bool(False),
      )
)

hpsPFTauDiscriminationByMediumIsolation                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByMediumIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants = requireDecayMode
hpsPFTauDiscriminationByMediumIsolation.qualityCuts      = mediumPFTauQualityCuts# set the standard quality cuts



#Define a discriminator By Tight Isolation!
#You need to loosen qualityCuts for this
loosePFTauQualityCuts = cms.PSet(
      signalQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.5),  # filter PFChargedHadrons below given pt
         maxTrackChi2                 = cms.double(100.), # require track Chi2
         maxTransverseImpactParameter = cms.double(0.03), # w.r.t. PV
         maxDeltaZ                    = cms.double(0.2),  # w.r.t. PV
         minTrackPixelHits            = cms.uint32(0),    # pixel-only hits (note that these cuts are turned off, 
                                                          # the tracking cuts might be higher)
         minTrackHits                 = cms.uint32(3),    # total track hits
         minGammaEt                   = cms.double(0.5),  # filter PFgammas below given Pt
         useTracksInsteadOfPFHadrons  = cms.bool(False),  # if true, use generalTracks, instead of PFChargedHadrons
      ),
      isolationQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.5), 
         maxTrackChi2                 = cms.double(100.),
         maxTransverseImpactParameter = cms.double(0.03),
         maxDeltaZ                    = cms.double(0.2),
         minTrackPixelHits            = cms.uint32(0),
         minTrackHits                 = cms.uint32(3),
         minGammaEt                   = cms.double(0.5),
         useTracksInsteadOfPFHadrons  = cms.bool(False),
      )
)


hpsPFTauDiscriminationByTightIsolation                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByTightIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByTightIsolation.Prediscriminants = requireDecayMode
hpsPFTauDiscriminationByTightIsolation.qualityCuts      = loosePFTauQualityCuts# set the standard quality cuts



#copying discriminator against electrons and muons
hpsPFTauDiscriminationAgainstElectron                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(hpsPFTauDiscriminationAgainstElectron, 'hpsPFTauProducer')
hpsPFTauDiscriminationAgainstElectron.Prediscriminants = noPrediscriminants

hpsPFTauDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(hpsPFTauDiscriminationAgainstMuon, 'hpsPFTauProducer')
hpsPFTauDiscriminationAgainstMuon.Prediscriminants = noPrediscriminants

produceAndDiscriminateHPSPFTaus = cms.Sequence(
                 hpsPFTauProducer*
                 hpsPFTauDiscriminationByDecayModeFinding*
                 hpsPFTauDiscriminationByLooseIsolation*
                 hpsPFTauDiscriminationByMediumIsolation*
                 hpsPFTauDiscriminationByTightIsolation*
                 hpsPFTauDiscriminationAgainstElectron*
                 hpsPFTauDiscriminationAgainstMuon
)




