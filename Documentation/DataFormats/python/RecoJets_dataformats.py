'''
    Created on Jun 26, 2013
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/

    @responsible:

'''

json = {
  "full": {
    "title": "RecoJets collections (in RECO and AOD)",
    "data": [
     {
      "instance": "ak7JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "sc5JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ic5JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak5JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorExplicit",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "gk5JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "recoCaloJets",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "recoTrackJets",
      "desc": "No documentation"
     },
     {
      "instance": "caloTowers",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "CastorTowerReco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ic5JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "sisCone5JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "sisCone5JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "sisCone5JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt6GenJets",
      "container": "reco::GenJetCollection",
      "desc": "Fastjet kT R=0.6 jets reconstructed from stable generator particles"
     },
     {
      "instance": "ak7GenJets",
      "container": "reco::GenJetCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from stable generator particles. Note that the label is antikt7GenJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "kt4GenJets",
      "container": "reco::GenJetCollection",
      "desc": "Fastjet kT R=0.4 jets reconstructed from stable generator particles"
     },
     {
      "instance": "ak7CastorJetID",
      "container": "reco::CastorJetIDValueMap",
      "desc": "Corresponding JetID object to go with the ak7CastorJets, contains various information on how a jet in CASTOR looks, see CASTOR reconstruction page for more info"
     },
     {
      "instance": "ak5GenJets",
      "container": "reco::GenJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from stable generator particles. Note that the label is antikt5GenJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "kt4TrackJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7CastorJets",
      "container": "reco::CastorTowerCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from CastorTowers"
     },
     {
      "instance": "JetPlusTrackZSPCorJetAntiKt5",
      "container": "reco::JPTJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from CaloTowers, corrected with track response within the jet cone."
     },
     {
      "instance": "ak4TrackJets",
      "container": "reco::TrackJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from tracks."
     },
     {
      "instance": "kt6PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet kT R=0.6 jets reconstructed from PF particles"
     },
     {
      "instance": "iterativeCone5PFJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "recoPFJets",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "recoJPTJets",
      "desc": "No documentation"
     },
     {
      "instance": "towerMaker",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackRefsForJets",
      "container": "recoRecoChargedRefCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet kT R=0.4 jets reconstructed from PF particles"
     },
     {
      "instance": "ak7PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from PF particles. Note that the label is antikt7PFJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "ak4CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from CaloTowers with pT>0.5 GeV. Note that the label is antikt5CaloJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "kt6CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet kT R=0.6 jets reconstructed from CaloTowers with pT>0.5 GeV"
     },
     {
      "instance": "kt4CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet kT R=0.4 jets reconstructed from CaloTowers with pT>0.5 GeV"
     },
     {
      "instance": "ca4JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from PF particles. Note that the label is antikt5PFJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "iterativeCone15CaloJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5CaloJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from CaloTowers with pT>0.5 GeV. Note that the label is antikt7CaloJets for CMSSW_3_1_X (Summer09 MC production)"
     }
    ]
  },
  "aod": {
    "title": "RecoJets collections (in AOD only)",
    "data": [
     {
      "instance": "kt6PFJetsCentralNeutral",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "kt6PFJetsCentralNeutralTight",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "fixedGridRho*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4PFJetsCHS*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from CaloTowers with pT>0.5 GeV. Note that the label is antikt5CaloJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "ak4PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from PF particles. Note that the label is antikt5PFJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "kt6PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet kT R=0.6 jets reconstructed from PF particles"
     },
     {
      "instance": "ak4TrackJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7PFJets",
      "container": "reco::PFJetCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from PF particles. Note that the label is antikt7PFJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "caloTowers",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackRefsForJets",
      "container": "recoRecoChargedRefCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorAtVertex",
      "container": " ",
      "desc": "tracks associated to all ak4CaloJets within a Cone R=0.5 at the vertex"
     },
     {
      "instance": "CastorTowerReco",
      "container": "reco::CastorTowerCollection",
      "desc": "Collection of towers in CASTOR (RecHits in one phi sector summed over z)"
     },
     {
      "instance": "ak7JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorExplicit",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak5JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7CastorJetID",
      "container": "reco::CastorJetIDValueMap",
      "desc": "Corresponding JetID object to go with the ak7CastorJets, contains various information on how a jet in CASTOR looks, see CASTOR reconstruction page for more info"
     },
     {
      "instance": "ak7CastorJets",
      "container": "reco::CastorTowerCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from CastorTowers"
     },
     {
      "instance": "kt6PFJetsCentralChargedPileUp",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "kt6CaloJetsCentral",
      "container": "double",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoJets collections (in RECO only)",
    "data": [
     {
      "instance": "kt4JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorAtVertex",
      "container": " ",
      "desc": "tracks associated to all ak4CaloJets within a Cone R=0.5 at the vertex"
     },
     {
      "instance": "ak4JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetTracksAssociatorExplicit",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak5JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet kT R=0.4 jets reconstructed from CaloTowers with pT>0.5 GeV"
     },
     {
      "instance": "ak4CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from CaloTowers with pT>0.5 GeV. Note that the label is antikt5CaloJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "iterativeCone5CaloJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt4PFJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4PFJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "iterativeCone5PFJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4PFJetsCHS*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak4TrackJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "JetPlusTrackZSPCorJetAntiKt5",
      "container": "reco::JPTJetCollection",
      "desc": "Fastjet Anti-kT R=0.5 jets reconstructed from CaloTowers, corrected with track response within the jet cone."
     },
     {
      "instance": "trackRefsForJets",
      "container": "recoRecoChargedRefCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "kt4TrackJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "towerMaker",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "caloTowers",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ic5JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "CastorTowerReco",
      "container": "reco::CastorTowerCollection",
      "desc": "Collection of towers in CASTOR (RecHits in one phi sector summed over z)"
     },
     {
      "instance": "fixedGridRho*",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt6PFJetsCentralNeutral",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "kt6PFJetsCentralNeutralTight",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "kt6CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet kT R=0.6 jets reconstructed from CaloTowers with pT>0.5 GeV"
     },
     {
      "instance": "ak7CaloJets",
      "container": "reco::CaloJetCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from CaloTowers with pT>0.5 GeV. Note that the label is antikt7CaloJets for CMSSW_3_1_X (Summer09 MC production)"
     },
     {
      "instance": "iterativeCone15CaloJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt6PFJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7PFJets",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "kt6PFJetsCentralChargedPileUp",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "kt6CaloJetsCentral",
      "container": "double",
      "desc": "No documentation"
     },
     {
      "instance": "ak4JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7JetExtender",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7JetTracksAssociatorAtCaloFace",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7JetTracksAssociatorAtVertex",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ak7CastorJetID",
      "container": "reco::CastorJetIDValueMap",
      "desc": "Corresponding JetID object to go with the ak7CastorJets, contains various information on how a jet in CASTOR looks, see CASTOR reconstruction page for more info"
     },
     {
      "instance": "ak7CastorJets",
      "container": "reco::CastorTowerCollection",
      "desc": "Fastjet Anti-kT R=0.7 jets reconstructed from CastorTowers"
     },
     {
      "instance": "kt4JetID",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ic5JetID",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  }
}
