
full_title = "RecoTauTag collections (in RECO and AOD)"

full = {
    '0':['ak5PFJetsRecoTauPiZeros', '*', 'No documentation'] ,
    '1':['hpsPFTauProducer', '*', 'No documentation'] ,
    '2':['hpsPFTauDiscrimination*', '*', 'No documentation'] ,
    '3':['shrinkingConePFTauProducer', '*', 'No documentation'] ,
    '4':['shrinkingConePFTauDiscrimination*', '*', 'No documentation'] ,
    '5':['hpsTancTaus', '*', 'No documentation'] ,
    '6':['hpsTancTausDiscrimination*', '*', 'No documentation'] ,
    '7':['TCTauJetPlusTrackZSPCorJetAntiKt5', '*', 'No documentation'] ,
    '8':['caloRecoTauTagInfoProducer','reco::CaloTauTagInfo ','contains treated informations from JetTracksAssociation < a CaloJet,a list of Tracks > and Island ECAL BasicCluster objects which are used for CaloTau object elaboration ;  in the RECO and AOD content since CMSSW_1_7_0.'],
    '9':['caloRecoTauProducer','reco::CaloTau ','corresponds to the hadronic tau-jet cand. -starting from a CaloJet object- that the analysts would use ; in the RECO and AOD content since CMSSW_1_7_0.'],
    
      # Correction needed, because not matched with Event Content
    '10':['coneIsolationTauJetTags','reco::IsolatedTauTagInfo ','ConeIsolation dedicated TagInfo. Selected tracks and methods to re-compute the discriminator are present.'],
    '11':['coneIsolationTauJetTags','reco::JetTag ','Obsolete since 1_6_0.'],
    '12':['pfRecoTauTagInfoProducer','reco::PFTauTagInfo ','contains treated informations from JetTracksAssociation < a PFJet,a list of Tracks > object which are used for PFTau object elaboration ;  in the RECO and AOD content since CMSSW_1_7_0.'],
    '13':['pfRecoTauProducer','reco::PFTau ','corresponds to the hadronic tau-jet cand. -starting from a PFJet object- that the analysts would use ; in the RECO and AOD content since CMSSW_1_7_0.'],
    '14':['pfRecoTauDiscriminationByIsolation','reco::PFTauDiscriminatorByIsolation ','associates to each PFTau object the response of a hadr. tau-jet / q/g-jet discrimination procedure based on tracker(+ECAL) isolation ; in the RECO and AOD content since CMSSW_1_7_0.'],
    '15':['caloRecoTauDiscriminationByIsolation','reco::CaloTauDiscriminatorByIsolation ','associates to each CaloTau object the response of a hadr. tau-jet / q/g-jet discrimination procedure based on tracker isolation ; in the RECO and AOD content since CMSSW_1_7_0.'],
    '16':['combinedTauTag','reco::CombinedTauTagInfo ','CombinedTauTag dedicated TagInfo. The Discriminator is computed on a track isolation criteria and a Likelihood computed on the basis of the neutral activity. In the Reco content since 1_6_0.'],
    '17':['combinedTauTag','reco::JetTag ','Results of the CombinedTauTag Algorithm. In the Reco content since 1_6_0.'],
    '18':['pfConeIsolation','reco::PFIsolatedTauTagInfo ','Equivalent of the ConeIsolation based tagInfo but made with ParticleFlow objects. The isolation can be computed using charged and neutral particles. In the Reco content since 1_6_0.'],
    '19':['tauPFProducer (tauCaloProducer)','reco::Tau ','Tau class equivalent to the Muon, Electron and Jet one. Almost all the code for the tagging algorithm is being migrating to use this class as input. In this case the PFConeIsolation is used to fill the relevant variables. In the Reco content since 1_6_0.'] 
}

reco_title = "RecoTauTag collections (in RECO only)"

reco = {
    '0':['ak5PFJetsRecoTauPiZeros', '*', 'No documentation'] ,
    '1':['hpsPFTauProducer', '*', 'No documentation'] ,
    '2':['hpsPFTauDiscrimination*', '*', 'No documentation'] ,
    '3':['shrinkingConePFTauProducer', '*', 'No documentation'] ,
    '4':['shrinkingConePFTauDiscrimination*', '*', 'No documentation'] ,
    '5':['hpsTancTaus', '*', 'No documentation'] ,
    '6':['hpsTancTausDiscrimination*', '*', 'No documentation'] ,
    '7':['TCTauJetPlusTrackZSPCorJetAntiKt5', '*', 'No documentation'] ,
    '8':['caloRecoTauTagInfoProducer','reco::CaloTauTagInfo ','contains treated informations from JetTracksAssociation < a CaloJet,a list of Tracks > and Island ECAL BasicCluster objects which are used for CaloTau object elaboration ;  in the RECO and AOD content since CMSSW_1_7_0.'],
    '9':['caloRecoTauProducer','reco::CaloTau ','corresponds to the hadronic tau-jet cand. -starting from a CaloJet object- that the analysts would use ; in the RECO and AOD content since CMSSW_1_7_0.'] 
}

aod_title = "RecoTauTag collections (in AOD only)"

aod = {
    '0':['ak5PFJetsRecoTauPiZeros', '*', 'No documentation'] ,
    '1':['hpsPFTauProducer', '*', 'No documentation'] ,
    '2':['hpsPFTauDiscrimination*', '*', 'No documentation'] ,
    '3':['hpsTancTaus', '*', 'No documentation'] ,
    '4':['hpsTancTausDiscrimination*', '*', 'No documentation'] 
}