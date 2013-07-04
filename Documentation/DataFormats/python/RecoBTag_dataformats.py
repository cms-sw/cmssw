'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoBTag collections (in RECO and AOD)",
    "data": [
     {
      "instance": "combinedMVABJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonTagInfos",
      "container": "reco::SoftLeptonTagInfo ",
      "desc": "soft muon dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft muon in the jet"
     },
     {
      "instance": "softMuonBJetTags",
      "container": "reco::JetTag ",
      "desc": "results of b-tagging a jet using the SoftMuonTagInfo and the default soft muon tagger, which uses a neural network to combine most muon properties to improve rejection of non-b jets"
     },
     {
      "instance": "softMuonByIP3dBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonByPtBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackCountingHighEffBJetTags",
      "container": "reco::JetTag",
      "desc": "Result of track counting algorithm (requiring two tracks to have a significance above the discriminator). To be used for high efficiency selection (B eff > 50%, mistag rate > 1% )"
     },
     {
      "instance": "impactParameterTagInfos",
      "container": "reco::TrackIPTagInfo",
      "desc": "contains information used for btagging about track properties such as impact parameters, decay len, probability to originate from th primary vertex. Uses ak5JetTracksAssociatorAtVertex collection as input."
     },
     {
      "instance": "jetProbabilityBJetTags",
      "container": "reco::JetTag ",
      "desc": "result of jetProbability algorithm (based on TrackIPTagInfo)."
     },
     {
      "instance": "trackCountingHighPurBJetTags",
      "container": "reco::JetTag",
      "desc": "Result of track counting algorithm (requiring three tracks to have a significance above the discriminator). To be used for high purity selection (B eff < 50%, mistag rate < 1% )"
     },
     {
      "instance": "secondaryVertexTagInfos",
      "container": "reco::SecondaryVertexTagInfo ",
      "desc": "contains the reconstructed displaced secondary vertices in a jet and associated information, uses impactParameterTagInfos as input"
     },
     {
      "instance": "jetBProbabilityBJetTags",
      "container": "reco::JetTag ",
      "desc": "result of jetProbability algorithm in the `jetBProbability` variant."
     },
     {
      "instance": "simpleSecondaryVertexBJetTags",
      "container": "reco::JetTag ",
      "desc": "Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets"
     },
     {
      "instance": "ghostTrackVertexTagInfos",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "simpleSecondaryVertexHighPurBJetTags",
      "container": "reco::JetTag ",
      "desc": "Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with three or more tracks."
     },
     {
      "instance": "simpleSecondaryVertexHighEffBJetTags",
      "container": "reco::JetTag ",
      "desc": "Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with two or more tracks. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets"
     },
     {
      "instance": "combinedSecondaryVertexMVABJetTags",
      "container": "reco::JetTag ",
      "desc": "uses the PhysicsTools/MVAComputer framework to compute a discriminator from the impactParameterTagInfos and secondaryVertexTagInfos with an uptodate calibration from the the CMS conditions database, using a neural network instead of a likelihood ratio in case an actual secondary vertex was reconstructed"
     },
     {
      "instance": "combinedSecondaryVertexBJetTags",
      "container": "reco::JetTag ",
      "desc": "Result of application of a likelihood estimator to the tagging variables for the three possible algorithm outcomes (tracks only, pseudo vertex from at least two tracks or successful secondary vertex fit), obtained from impactParameterTagInfos and secondaryVertexTagInfos"
     },
     {
      "instance": "btagSoftElectrons",
      "container": "reco::Electron ",
      "desc": "Electron candidates identified by the dedicated btagging SoftElectronProducer, starting from reco::Tracks"
     },
     {
      "instance": "ghostTrackBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softPFElectrons",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronCands",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronBJetTags",
      "container": "reco::JetTag ",
      "desc": "results of b-tagging a jet using the SoftElectronTagInfo and the default soft electron tagger, which uses a neural network to combine most electron properties to improve rejection of non-b jets"
     },
     {
      "instance": "softElectronTagInfos",
      "container": "reco::SoftLeptonTagInfo ",
      "desc": "soft electron dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft electron in the jet"
     },
     {
      "instance": "softElectronByPtBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronByIP3dBJetTags",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoBTag collections (in AOD only)",
    "data": [
     {
      "instance": "softElectronByIP3dBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronByPtBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonByPtBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonByIP3dBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "combinedMVABJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackCountingHighPurBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackCountingHighEffBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "jetBProbabilityBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "jetProbabilityBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "simpleSecondaryVertexHighEffBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "simpleSecondaryVertexBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "combinedSecondaryVertexBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "simpleSecondaryVertexHighPurBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ghostTrackBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "combinedSecondaryVertexMVABJetTags",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoBTag collections (in RECO only)",
    "data": [
     {
      "instance": "combinedMVABJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonTagInfos",
      "container": "reco::SoftLeptonTagInfo ",
      "desc": "soft muon dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft muon in the jet"
     },
     {
      "instance": "softMuonBJetTags",
      "container": "reco::JetTag ",
      "desc": "results of b-tagging a jet using the SoftMuonTagInfo and the default soft muon tagger, which uses a neural network to combine most muon properties to improve rejection of non-b jets"
     },
     {
      "instance": "softMuonByIP3dBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softMuonByPtBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackCountingHighEffBJetTags",
      "container": "reco::JetTag",
      "desc": "Result of track counting algorithm (requiring two tracks to have a significance above the discriminator). To be used for high efficiency selection (B eff > 50%, mistag rate > 1% )"
     },
     {
      "instance": "impactParameterTagInfos",
      "container": "reco::TrackIPTagInfo",
      "desc": "contains information used for btagging about track properties such as impact parameters, decay len, probability to originate from th primary vertex. Uses ak5JetTracksAssociatorAtVertex collection as input."
     },
     {
      "instance": "jetProbabilityBJetTags",
      "container": "reco::JetTag ",
      "desc": "result of jetProbability algorithm (based on TrackIPTagInfo)."
     },
     {
      "instance": "trackCountingHighPurBJetTags",
      "container": "reco::JetTag",
      "desc": "Result of track counting algorithm (requiring three tracks to have a significance above the discriminator). To be used for high purity selection (B eff < 50%, mistag rate < 1% )"
     },
     {
      "instance": "secondaryVertexTagInfos",
      "container": "reco::SecondaryVertexTagInfo ",
      "desc": "contains the reconstructed displaced secondary vertices in a jet and associated information, uses impactParameterTagInfos as input"
     },
     {
      "instance": "jetBProbabilityBJetTags",
      "container": "reco::JetTag ",
      "desc": "result of jetProbability algorithm in the `jetBProbability` variant."
     },
     {
      "instance": "simpleSecondaryVertexBJetTags",
      "container": "reco::JetTag ",
      "desc": "Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets"
     },
     {
      "instance": "ghostTrackVertexTagInfos",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "simpleSecondaryVertexHighPurBJetTags",
      "container": "reco::JetTag ",
      "desc": "Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with three or more tracks."
     },
     {
      "instance": "simpleSecondaryVertexHighEffBJetTags",
      "container": "reco::JetTag ",
      "desc": "Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with two or more tracks. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets"
     },
     {
      "instance": "combinedSecondaryVertexMVABJetTags",
      "container": "reco::JetTag ",
      "desc": "uses the PhysicsTools/MVAComputer framework to compute a discriminator from the impactParameterTagInfos and secondaryVertexTagInfos with an uptodate calibration from the the CMS conditions database, using a neural network instead of a likelihood ratio in case an actual secondary vertex was reconstructed"
     },
     {
      "instance": "combinedSecondaryVertexBJetTags",
      "container": "reco::JetTag ",
      "desc": "Result of application of a likelihood estimator to the tagging variables for the three possible algorithm outcomes (tracks only, pseudo vertex from at least two tracks or successful secondary vertex fit), obtained from impactParameterTagInfos and secondaryVertexTagInfos"
     },
     {
      "instance": "btagSoftElectrons",
      "container": "reco::Electron ",
      "desc": "Electron candidates identified by the dedicated btagging SoftElectronProducer, starting from reco::Tracks"
     },
     {
      "instance": "ghostTrackBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softPFElectrons",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronCands",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronBJetTags",
      "container": "reco::JetTag ",
      "desc": "results of b-tagging a jet using the SoftElectronTagInfo and the default soft electron tagger, which uses a neural network to combine most electron properties to improve rejection of non-b jets"
     },
     {
      "instance": "softElectronTagInfos",
      "container": "reco::SoftLeptonTagInfo ",
      "desc": "soft electron dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft electron in the jet"
     },
     {
      "instance": "softElectronByPtBJetTags",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "softElectronByIP3dBJetTags",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  }
}
