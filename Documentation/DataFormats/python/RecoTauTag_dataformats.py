'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoTauTag collections (in RECO and AOD)",
    "data": [
     {
      "instance": "coneIsolationTauJetTags",
      "container": "reco::JetTag ",
      "desc": "Obsolete since 1_6_0."
     },
     {
      "instance": "coneIsolationTauJetTags",
      "container": "reco::IsolatedTauTagInfo ",
      "desc": "ConeIsolation dedicated TagInfo. Selected tracks and methods to re-compute the discriminator are present."
     },
     {
      "instance": "pfRecoTauProducer",
      "container": "reco::PFTau ",
      "desc": "corresponds to the hadronic tau-jet cand. -starting from a PFJet object- that the analysts would use ; in the RECO and AOD content since CMSSW_1_7_0."
     },
     {
      "instance": "pfRecoTauTagInfoProducer",
      "container": "reco::PFTauTagInfo ",
      "desc": "contains treated informations from JetTracksAssociation < a PFJet,a list of Tracks > object which are used for PFTau object elaboration ;  in the RECO and AOD content since CMSSW_1_7_0."
     },
     {
      "instance": "caloRecoTauDiscriminationByIsolation",
      "container": "reco::CaloTauDiscriminatorByIsolation ",
      "desc": "associates to each CaloTau object the response of a hadr. tau-jet / q/g-jet discrimination procedure based on tracker isolation ; in the RECO and AOD content since CMSSW_1_7_0."
     },
     {
      "instance": "pfRecoTauDiscriminationByIsolation",
      "container": "reco::PFTauDiscriminatorByIsolation ",
      "desc": "associates to each PFTau object the response of a hadr. tau-jet / q/g-jet discrimination procedure based on tracker(+ECAL) isolation ; in the RECO and AOD content since CMSSW_1_7_0."
     },
     {
      "instance": "combinedTauTag",
      "container": "reco::JetTag ",
      "desc": "Results of the CombinedTauTag Algorithm. In the Reco content since 1_6_0."
     },
     {
      "instance": "combinedTauTag",
      "container": "reco::CombinedTauTagInfo ",
      "desc": "CombinedTauTag dedicated TagInfo. The Discriminator is computed on a track isolation criteria and a Likelihood computed on the basis of the neutral activity. In the Reco content since 1_6_0."
     },
     {
      "instance": "tauPFProducer (tauCaloProducer)",
      "container": "reco::Tau ",
      "desc": "Tau class equivalent to the Muon, Electron and Jet one. Almost all the code for the tagging algorithm is being migrating to use this class as input. In this case the PFConeIsolation is used to fill the relevant variables. In the Reco content since 1_6_0."
     },
     {
      "instance": "pfConeIsolation",
      "container": "reco::PFIsolatedTauTagInfo ",
      "desc": "Equivalent of the ConeIsolation based tagInfo but made with ParticleFlow objects. The isolation can be computed using charged and neutral particles. In the Reco content since 1_6_0."
     },
     {
      "instance": "hpsPFTauProducer",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak5PFJetsRecoTauPiZeros",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "shrinkingConePFTauProducer",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsPFTauDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsTancTaus",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "shrinkingConePFTauDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "TCTauJetPlusTrackZSPCorJetAntiKt5",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsTancTausDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "caloRecoTauProducer",
      "container": "reco::CaloTau ",
      "desc": "corresponds to the hadronic tau-jet cand. -starting from a CaloJet object- that the analysts would use ; in the RECO and AOD content since CMSSW_1_7_0."
     },
     {
      "instance": "caloRecoTauTagInfoProducer",
      "container": "reco::CaloTauTagInfo ",
      "desc": "contains treated informations from JetTracksAssociation < a CaloJet,a list of Tracks > and Island ECAL BasicCluster objects which are used for CaloTau object elaboration ;  in the RECO and AOD content since CMSSW_1_7_0."
     }
    ]
  },
  "aod": {
    "title": "RecoTauTag collections (in AOD only)",
    "data": [
     {
      "instance": "hpsPFTauProducer",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak5PFJetsRecoTauPiZeros",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsTancTaus",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsPFTauDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsTancTausDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoTauTag collections (in RECO only)",
    "data": [
     {
      "instance": "hpsPFTauProducer",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak5PFJetsRecoTauPiZeros",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "shrinkingConePFTauProducer",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsPFTauDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsTancTaus",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "shrinkingConePFTauDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "TCTauJetPlusTrackZSPCorJetAntiKt5",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hpsTancTausDiscrimination*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "caloRecoTauProducer",
      "container": "reco::CaloTau ",
      "desc": "corresponds to the hadronic tau-jet cand. -starting from a CaloJet object- that the analysts would use ; in the RECO and AOD content since CMSSW_1_7_0."
     },
     {
      "instance": "caloRecoTauTagInfoProducer",
      "container": "reco::CaloTauTagInfo ",
      "desc": "contains treated informations from JetTracksAssociation < a CaloJet,a list of Tracks > and Island ECAL BasicCluster objects which are used for CaloTau object elaboration ;  in the RECO and AOD content since CMSSW_1_7_0."
     }
    ]
  }
}
