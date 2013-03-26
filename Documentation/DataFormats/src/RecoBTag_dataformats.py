

full_title = "RecoBTag collections (in RECO and AOD)"

full = {
    '0':['impactParameterTagInfos', 'reco::TrackIPTagInfo', 'contains information used for btagging about track properties such as impact parameters, decay len, probability to originate from th primary vertex. Uses ak5JetTracksAssociatorAtVertex collection as input.'] ,
    '1':['trackCountingHighEffBJetTags', 'reco::JetTag', 'Result of track counting algorithm (requiring two tracks to have a significance above the discriminator). To be used for high efficiency selection (B eff > 50%, mistag rate > 1% )'] ,
    '2':['trackCountingHighPurBJetTags', 'reco::JetTag', 'Result of track counting algorithm (requiring three tracks to have a significance above the discriminator). To be used for high purity selection (B eff < 50%, mistag rate < 1% )'] ,
    '3':['jetProbabilityBJetTags','reco::JetTag ','result of jetProbability algorithm (based on TrackIPTagInfo).'],
    '4':['jetBProbabilityBJetTags','reco::JetTag ','result of jetProbability algorithm in the `jetBProbability` variant.'],
    '5':['secondaryVertexTagInfos','reco::SecondaryVertexTagInfo ','contains the reconstructed displaced secondary vertices in a jet and associated information, uses impactParameterTagInfos as input'],
    '6':['ghostTrackVertexTagInfos', '*', 'No documentation'] ,
    '7':['simpleSecondaryVertexBJetTags','reco::JetTag ','Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets'],
    '8':['simpleSecondaryVertexHighEffBJetTags','reco::JetTag ','Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with two or more tracks. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets'],
    '9':['simpleSecondaryVertexHighPurBJetTags','reco::JetTag ','Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with three or more tracks.'],
    '10':['combinedSecondaryVertexBJetTags','reco::JetTag ','Result of application of a likelihood estimator to the tagging variables for the three possible algorithm outcomes (tracks only, pseudo vertex from at least two tracks or successful secondary vertex fit), obtained from impactParameterTagInfos and secondaryVertexTagInfos'],
    '11':['combinedSecondaryVertexMVABJetTags','reco::JetTag ','uses the PhysicsTools/MVAComputer framework to compute a discriminator from the impactParameterTagInfos and secondaryVertexTagInfos with an uptodate calibration from the the CMS conditions database, using a neural network instead of a likelihood ratio in case an actual secondary vertex was reconstructed'],
    '12':['ghostTrackBJetTags', '*', 'No documentation'] ,
    '13':['btagSoftElectrons','reco::Electron ','Electron candidates identified by the dedicated btagging SoftElectronProducer, starting from reco::Tracks'],
    '14':['softElectronCands', '*', 'No documentation'] ,
    '15':['softPFElectrons', '*', 'No documentation'] ,
    '16':['softElectronTagInfos','reco::SoftLeptonTagInfo ','soft electron dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft electron in the jet'],
    '17':['softElectronBJetTags','reco::JetTag ','results of b-tagging a jet using the SoftElectronTagInfo and the default soft electron tagger, which uses a neural network to combine most electron properties to improve rejection of non-b jets'],
    '18':['softElectronByIP3dBJetTags', '*', 'No documentation'] ,
    '19':['softElectronByPtBJetTags', '*', 'No documentation'] ,
    '20':['softMuonTagInfos','reco::SoftLeptonTagInfo ','soft muon dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft muon in the jet'],
    '21':['softMuonBJetTags','reco::JetTag ','results of b-tagging a jet using the SoftMuonTagInfo and the default soft muon tagger, which uses a neural network to combine most muon properties to improve rejection of non-b jets'],
    '22':['softMuonByIP3dBJetTags', '*', 'No documentation'] ,
    '23':['softMuonByPtBJetTags', '*', 'No documentation'] ,
    '24':['combinedMVABJetTags', '*', 'No documentation'] 
}

reco_title = "RecoBTag collections (in RECO only)"

reco = {
    '0':['impactParameterTagInfos', 'reco::TrackIPTagInfo', 'contains information used for btagging about track properties such as impact parameters, decay len, probability to originate from th primary vertex. Uses ak5JetTracksAssociatorAtVertex collection as input.'] ,
    '1':['trackCountingHighEffBJetTags', 'reco::JetTag', 'Result of track counting algorithm (requiring two tracks to have a significance above the discriminator). To be used for high efficiency selection (B eff > 50%, mistag rate > 1% )'] ,
    '2':['trackCountingHighPurBJetTags', 'reco::JetTag', 'Result of track counting algorithm (requiring three tracks to have a significance above the discriminator). To be used for high purity selection (B eff < 50%, mistag rate < 1% )'] ,
    '3':['jetProbabilityBJetTags','reco::JetTag ','result of jetProbability algorithm (based on TrackIPTagInfo).'],
    '4':['jetBProbabilityBJetTags','reco::JetTag ','result of jetProbability algorithm in the `jetBProbability` variant.'],
    '5':['secondaryVertexTagInfos','reco::SecondaryVertexTagInfo ','contains the reconstructed displaced secondary vertices in a jet and associated information, uses impactParameterTagInfos as input'],
    '6':['ghostTrackVertexTagInfos', '*', 'No documentation'] ,
    '7':['simpleSecondaryVertexBJetTags','reco::JetTag ','Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets'],
    '8':['simpleSecondaryVertexHighEffBJetTags','reco::JetTag ','Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with two or more tracks. Can be configured to return the value or significance in 2d and 3d, optionally corrected for the boost at the SV - works up to a maximum secondary vertex finding efficiency of ~70% in b-jets'],
    '9':['simpleSecondaryVertexHighPurBJetTags','reco::JetTag ','Uses the flight distance (i.e. distance between a reconstructed secondary vertex and the primary vertex in a jet) as b-tagging discriminator. Secondary vertex is reconstructed with three or more tracks.'],
    '10':['combinedSecondaryVertexBJetTags','reco::JetTag ','Result of application of a likelihood estimator to the tagging variables for the three possible algorithm outcomes (tracks only, pseudo vertex from at least two tracks or successful secondary vertex fit), obtained from impactParameterTagInfos and secondaryVertexTagInfos'],
    '11':['combinedSecondaryVertexMVABJetTags','reco::JetTag ','uses the PhysicsTools/MVAComputer framework to compute a discriminator from the impactParameterTagInfos and secondaryVertexTagInfos with an uptodate calibration from the the CMS conditions database, using a neural network instead of a likelihood ratio in case an actual secondary vertex was reconstructed'],
    '12':['ghostTrackBJetTags', '*', 'No documentation'] ,
    '13':['btagSoftElectrons','reco::Electron ','Electron candidates identified by the dedicated btagging SoftElectronProducer, starting from reco::Tracks'],
    '14':['softElectronCands', '*', 'No documentation'] ,
    '15':['softPFElectrons', '*', 'No documentation'] ,
    '16':['softElectronTagInfos','reco::SoftLeptonTagInfo ','soft electron dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft electron in the jet'],
    '17':['softElectronBJetTags','reco::JetTag ','results of b-tagging a jet using the SoftElectronTagInfo and the default soft electron tagger, which uses a neural network to combine most electron properties to improve rejection of non-b jets'],
    '18':['softElectronByIP3dBJetTags', '*', 'No documentation'] ,
    '19':['softElectronByPtBJetTags', '*', 'No documentation'] ,
    '20':['softMuonTagInfos','reco::SoftLeptonTagInfo ','soft muon dedicated TagInfo, containing informations used to b-tag jets due to the presence of a soft muon in the jet'],
    '21':['softMuonBJetTags','reco::JetTag ','results of b-tagging a jet using the SoftMuonTagInfo and the default soft muon tagger, which uses a neural network to combine most muon properties to improve rejection of non-b jets'],
    '22':['softMuonByIP3dBJetTags', '*', 'No documentation'] ,
    '23':['softMuonByPtBJetTags', '*', 'No documentation'] ,
    '24':['combinedMVABJetTags', '*', 'No documentation'] 
}

aod_title = "RecoBTag collections (in AOD only)"

aod = {
    '0':['trackCountingHighEffBJetTags', '*', 'No documentation'] ,
    '1':['trackCountingHighPurBJetTags', '*', 'No documentation'] ,
    '2':['jetProbabilityBJetTags', '*', 'No documentation'] ,
    '3':['jetBProbabilityBJetTags', '*', 'No documentation'] ,
    '4':['simpleSecondaryVertexBJetTags', '*', 'No documentation'] ,
    '5':['simpleSecondaryVertexHighEffBJetTags', '*', 'No documentation'] ,
    '6':['simpleSecondaryVertexHighPurBJetTags', '*', 'No documentation'] ,
    '7':['combinedSecondaryVertexBJetTags', '*', 'No documentation'] ,
    '8':['combinedSecondaryVertexMVABJetTags', '*', 'No documentation'] ,
    '9':['ghostTrackBJetTags', '*', 'No documentation'] ,
    '10':['softElectronBJetTags', '*', 'No documentation'] ,
    '11':['softElectronByIP3dBJetTags', '*', 'No documentation'] ,
    '12':['softElectronByPtBJetTags', '*', 'No documentation'] ,
    '13':['softMuonBJetTags', '*', 'No documentation'] ,
    '14':['softMuonByIP3dBJetTags', '*', 'No documentation'] ,
    '15':['softMuonByPtBJetTags', '*', 'No documentation'] ,
    '16':['combinedMVABJetTags', '*', 'No documentation'] 
}