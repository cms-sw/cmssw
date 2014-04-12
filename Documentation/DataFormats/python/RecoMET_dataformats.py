'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoMET collections (in RECO and AOD)",
    "data": [
     {
      "instance": "metOptHO",
      "container": "reco::CaloMETCollection",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, HF and HO with optimized threshold parameters"
     },
     {
      "instance": "metOptNoHFHO",
      "container": "reco::CaloMETCollection",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HO with optimized threshold parameters"
     },
     {
      "instance": "htMetIC5",
      "container": "reco::METCollection",
      "desc": "Raw Missing Transverse Energy calculated using IC5 CaloJets"
     },
     {
      "instance": "metNoHFHO",
      "container": "reco::CaloMETCollection",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HO"
     },
     {
      "instance": "metOpt",
      "container": "reco::CaloMETCollection",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF with optimized threshold parameters"
     },
     {
      "instance": "metOptNoHF",
      "container": "reco::CaloMETCollection",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, and HE with optimized threshold parameters"
     },
     {
      "instance": "metNoHF",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, and HE"
     },
     {
      "instance": "met",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF"
     },
     {
      "instance": "corMetGlobalMuons",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF with corrections for muons"
     },
     {
      "instance": "metHO",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, HF, and HO"
     },
     {
      "instance": "tcMetWithPFclusters",
      "container": "recoMETs",
      "desc": "Track Corrected MET using `particleFlowClusters`, `muons`, `gsfElectrons` and `generalTracks`"
     },
     {
      "instance": "tcMet",
      "container": "recoMETs",
      "desc": "Track Corrected MET using `met`, `muons`, `gsfElectrons` and `generalTracks`"
     },
     {
      "instance": "muonMETValueMapProducer",
      "container": "recoMuonMETCorrectionDataedmValueMap",
      "desc": "Information on how muons were used to correct MET and what associated MIP deposits are used"
     },
     {
      "instance": "pfMet",
      "container": "recoPFMETs",
      "desc": "MET in reconstructed particles in the particle flow algorithm"
     },
     {
      "instance": "hcalnoise",
      "container": "recoHcalNoiseRBXs",
      "desc": "No documentation"
     },
     {
      "instance": "muonTCMETValueMapProducer",
      "container": "recoMuonMETCorrectionDataedmValueMap",
      "desc": "Information on how muons were used to correct tcMET and what associated calo deposits are if the muon is treated as a pion"
     },
     {
      "instance": "*",
      "container": "*HaloData",
      "desc": "No documentation"
     },
     {
      "instance": "hcalnoise",
      "container": "HcalNoiseSummary",
      "desc": "No documentation"
     },
     {
      "instance": "genMetTrue",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding neutrinos, excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos"
     },
     {
      "instance": "BeamHaloSummary",
      "container": "*BeamHaloSummary",
      "desc": "No documentation"
     },
     {
      "instance": "genMetCaloAndNonPrompt",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos and, additionally, excluding muons and neutrinos coming from the decay of gauge bosons and top quarks"
     },
     {
      "instance": "genMetCalo",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding neutrinos, excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos and also muons"
     },
     {
      "instance": "htMetAK7",
      "container": "reco::METCollection",
      "desc": "Raw Missing Transverse Energy calculated using anti-KT7 CaloJets"
     },
     {
      "instance": "htMetAK5",
      "container": "reco::METCollection",
      "desc": "Raw Missing Transverse Energy calculated using anti-KT5 CaloJets"
     },
     {
      "instance": "htMetKT6",
      "container": "reco::METCollection",
      "desc": "Raw Missing Transverse Energy calculated using FastKt6 CaloJets"
     },
     {
      "instance": "htMetKT4",
      "container": "reco::METCollection",
      "desc": "Raw Missing Transverse Energy calculated using FastKt4 CaloJets"
     }
    ]
  },
  "aod": {
    "title": "RecoMET collections (in AOD only)",
    "data": [
     {
      "instance": "BeamHaloSummary",
      "container": "*BeamHaloSummary",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "*GlobalHaloData",
      "desc": "No documentation"
     },
     {
      "instance": "genMetCalo",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding neutrinos, excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos and also muons"
     },
     {
      "instance": "genMetTrue",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding neutrinos, excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos"
     },
     {
      "instance": "genMetCaloAndNonPrompt",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos and, additionally, excluding muons and neutrinos coming from the decay of gauge bosons and top quarks"
     },
     {
      "instance": "metNoHF",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, and HE"
     },
     {
      "instance": "met",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF"
     },
     {
      "instance": "corMetGlobalMuons",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF with corrections for muons"
     },
     {
      "instance": "metHO",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, HF, and HO"
     },
     {
      "instance": "tcMetWithPFclusters",
      "container": "recoMETs",
      "desc": "Track Corrected MET using `particleFlowClusters`, `muons`, `gsfElectrons` and `generalTracks`"
     },
     {
      "instance": "tcMet",
      "container": "recoMETs",
      "desc": "Track Corrected MET using `met`, `muons`, `gsfElectrons` and `generalTracks`"
     },
     {
      "instance": "muonMETValueMapProducer",
      "container": "recoMuonMETCorrectionDataedmValueMap",
      "desc": "Information on how muons were used to correct MET and what associated MIP deposits are used"
     },
     {
      "instance": "pfMet",
      "container": "recoPFMETs",
      "desc": "MET in reconstructed particles in the particle flow algorithm"
     },
     {
      "instance": "hcalnoise",
      "container": "HcalNoiseSummary",
      "desc": "No documentation"
     },
     {
      "instance": "muonTCMETValueMapProducer",
      "container": "recoMuonMETCorrectionDataedmValueMap",
      "desc": "Information on how muons were used to correct tcMET and what associated calo deposits are if the muon is treated as a pion"
     }
    ]
  },
  "reco": {
    "title": "RecoMET collections (in RECO only)",
    "data": [
     {
      "instance": "*",
      "container": "*HaloData",
      "desc": "No documentation"
     },
     {
      "instance": "hcalnoise",
      "container": "HcalNoiseSummary",
      "desc": "No documentation"
     },
     {
      "instance": "genMetTrue",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding neutrinos, excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos"
     },
     {
      "instance": "BeamHaloSummary",
      "container": "*BeamHaloSummary",
      "desc": "No documentation"
     },
     {
      "instance": "genMetCaloAndNonPrompt",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos and, additionally, excluding muons and neutrinos coming from the decay of gauge bosons and top quarks"
     },
     {
      "instance": "genMetCalo",
      "container": "reco::GenMETCollection",
      "desc": "MET in generated particles in simulation in their final states but excluding neutrinos, excited neutrinos, right-handed neutrinos, sneutrinos, neutralinos, gravitons, gravitinos and also muons"
     },
     {
      "instance": "metNoHF",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, and HE"
     },
     {
      "instance": "met",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF"
     },
     {
      "instance": "corMetGlobalMuons",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, and HF with corrections for muons"
     },
     {
      "instance": "metHO",
      "container": "recoCaloMETs",
      "desc": "MET in energy deposits in calorimeter towers in EB, EE, HB, HE, HF, and HO"
     },
     {
      "instance": "tcMetWithPFclusters",
      "container": "recoMETs",
      "desc": "Track Corrected MET using `particleFlowClusters`, `muons`, `gsfElectrons` and `generalTracks`"
     },
     {
      "instance": "tcMet",
      "container": "recoMETs",
      "desc": "Track Corrected MET using `met`, `muons`, `gsfElectrons` and `generalTracks`"
     },
     {
      "instance": "muonMETValueMapProducer",
      "container": "recoMuonMETCorrectionDataedmValueMap",
      "desc": "Information on how muons were used to correct MET and what associated MIP deposits are used"
     },
     {
      "instance": "pfMet",
      "container": "recoPFMETs",
      "desc": "MET in reconstructed particles in the particle flow algorithm"
     },
     {
      "instance": "hcalnoise",
      "container": "recoHcalNoiseRBXs",
      "desc": "No documentation"
     },
     {
      "instance": "muonTCMETValueMapProducer",
      "container": "recoMuonMETCorrectionDataedmValueMap",
      "desc": "Information on how muons were used to correct tcMET and what associated calo deposits are if the muon is treated as a pion"
     }
    ]
  }
}
