
full_title = "RecoEgamma collections (in RECO and AOD)"

full = {
    '0':['gsfElectronCores','reco::GsfElectronCoreCollection','Offline electron object containing a minimal set of information: reference to GsfTrack, reference to SuperCluster (Egamma and/or PFlow cluster) and provenance information. The collection is unpreselected and contains electrons reconstructed either by Egamma or by PF algorithm.'],
    '1':['gsfElectrons','reco::GsfElectronCollection','Offline electron containing all the relevant observables and corrections. The collection is produced starting from GsfElectronCore and its modularity allows to easily recompute electron variables skipping the reconstruction step. Each electron has also a reference to the corresponding `core` partner. (standard offline electron collection)'],
    '2':['uncleanedOnlyGsfElectronCores', '*', 'No documentation'] ,
    '3':['uncleanedOnlyGsfElectrons', '*', 'No documentation'] ,
    '4':['eidRobustLoose', '*', 'No documentation'] ,
    '5':['eidRobustTight', '*', 'No documentation'] ,
    '6':['eidRobustHighEnergy', '*', 'No documentation'] ,
    '7':['eidLoose', '*', 'No documentation'] ,
    '8':['eidTight', '*', 'No documentation'] ,
    '9':['conversions', 'recoConversions', 'Converted photons'] ,
    '10':['photonCore','reco::PhotonCoreCollection','Photon objects containing reference to SuperCluster, Conversions and ElectronSeeds.'],
    '11':['photons','reco::PhotonCollection','Photons with all the related observables like vertex, shower shapes, isolation. Each photon has also a reference to the corresponding `core` partner. (standard reconstructed photon collection)'],
    '12':['allConversions', '*', 'No documentation'] ,
    '13':['ckfOutInTracksFromConversions', 'recoTracks', 'Conversion tracks from outside-in tracking.'] ,
    '14':['ckfInOutTracksFromConversions', 'recoTracks', 'Conversion tracks from inside-out tracking.'] ,
    '15':['uncleanedOnlyAllConversions', '*', 'No documentation'] ,
    '16':['uncleanedOnlyCkfOutInTracksFromConversions', '*', 'No documentation'] ,
    '17':['uncleanedOnlyCkfInOutTracksFromConversions', '*', 'No documentation'] ,
    '18':['PhotonIDProd', '*', 'No documentation'] ,
    '19':['hfRecoEcalCandidate', '*', 'No documentation'] ,
    '20':['hfEMClusters', '*', 'No documentation'],
      # Correction needed, because not matched with Event Content
    '21':['electronGsfTracks','reco::GsfTrackCollection','GsfTracks for GsfElectrons (produced in a dedicated module in the standard reconstruction sequence). They have been produced from the electronMergedSeeds described below.'],
    '22':['eidTYPE',' ','ValueMaps containing a reference to a CMS.GsfElectron and the result of the electron ID. The event has one Value Map per identification TYPE (eidLoose, eidRobust...) (standard electron ID computed in standard sequence)'],
    '23':['siStripElectronToTrackAssociator','reco::ElectronCollection','Electrons reconstructed using tracks seeded in the Si strip layers (not produced in standard sequence)'],
    '24':['egammaCTFFinalFitWithMaterial','reco::TrackCollection','Tracks for Si strip seeded electrons (not produced in standard sequence)'],
    '25':['siStripElectrons','reco::SiStripElectronCollection','Intermediate object for Si Strip electron reconstruction containing information about Si strip hits associated to the SuperCluster (not produced in standard sequence)'] 
}

reco_title = "RecoEgamma collections (in RECO only)"

reco = {
    '0':['gsfElectronCores','reco::GsfElectronCoreCollection','Offline electron object containing a minimal set of information: reference to GsfTrack, reference to SuperCluster (Egamma and/or PFlow cluster) and provenance information. The collection is unpreselected and contains electrons reconstructed either by Egamma or by PF algorithm.'],
    '1':['gsfElectrons','reco::GsfElectronCollection','Offline electron containing all the relevant observables and corrections. The collection is produced starting from GsfElectronCore and its modularity allows to easily recompute electron variables skipping the reconstruction step. Each electron has also a reference to the corresponding `core` partner. (standard offline electron collection)'],
    '2':['uncleanedOnlyGsfElectronCores', 'recoGsfElectronCores', 'No documentation'] ,
    '3':['uncleanedOnlyGsfElectrons', 'recoGsfElectrons', 'No documentation'] ,
    '4':['eidRobustLoose', 'floatedmValueMap', 'No documentation'] ,
    '5':['eidRobustTight', 'floatedmValueMap', 'No documentation'] ,
    '6':['eidRobustHighEnergy', 'floatedmValueMap', 'No documentation'] ,
    '7':['eidLoose', 'floatedmValueMap', 'No documentation'] ,
    '8':['eidTight', 'floatedmValueMap', 'No documentation'] ,
    '9':['photons','reco::PhotonCollection','Photons with all the related observables like vertex, shower shapes, isolation. Each photon has also a reference to the corresponding `core` partner. (standard reconstructed photon collection)'],
    '10':['photonCore','reco::PhotonCoreCollection','Photon objects containing reference to SuperCluster, Conversions and ElectronSeeds.'],
    '11':['conversions', 'recoConversions', 'Converted photons'] ,
    '12':['allConversions', 'recoConversions', 'No documentation'] ,
    '13':['ckfOutInTracksFromConversions', 'recoTracks', 'Conversion tracks from outside-in tracking.'] ,
    '14':['ckfInOutTracksFromConversions', 'recoTracks', 'Conversion tracks from inside-out tracking.'] ,
    '15':['ckfOutInTracksFromConversions', 'recoTrackExtras', 'TrackExtras and TrackingRecHits for conversion tracks from outside-in tracking.'] ,
    '16':['ckfInOutTracksFromConversions', 'recoTrackExtras', 'TrackExtras and TrackingRecHits for conversion tracks from inside-out tracking.'] ,
    '17':['ckfOutInTracksFromConversions', 'TrackingRecHitsOwned', 'No documentation'] ,
    '18':['ckfInOutTracksFromConversions', 'TrackingRecHitsOwned', 'No documentation'] ,
    '19':['uncleanedOnlyAllConversions', 'recoConversions', 'No documentation'] ,
    '20':['uncleanedOnlyCkfOutInTracksFromConversions', 'recoTracks', 'No documentation'] ,
    '21':['uncleanedOnlyCkfInOutTracksFromConversions', 'recoTracks', 'No documentation'] ,
    '22':['uncleanedOnlyCkfOutInTracksFromConversions', 'recoTrackExtras', 'No documentation'] ,
    '23':['uncleanedOnlyCkfInOutTracksFromConversions', 'recoTrackExtras', 'No documentation'] ,
    '24':['uncleanedOnlyCkfOutInTracksFromConversions', 'TrackingRecHitsOwned', 'No documentation'] ,
    '25':['uncleanedOnlyCkfInOutTracksFromConversions', 'TrackingRecHitsOwned', 'No documentation'] ,
    '26':['PhotonIDProd','reco::PhotonIDCollection','Photons identification variables, calculated for corresponding photons. (standard reconstructed photon collection)'],
    '27':['hfRecoEcalCandidate', '*', 'No documentation'] ,
    '28':['hfEMClusters', '*', 'No documentation'],
      # Correction needed, because not matched with Event Content
    '29':['gsfElectronsGsfFit','reco::GsfTrackExtraCollection reco::TrackExtraCollection TrackingRecHitCollection','GsfTrackExtras, TrackExtras, TrackingRecHits'],
    '30':['electronMergedSeeds','reco::ElectronSeedCollection','Mixed collection of ElectronSeeds coming from Egamma and/or PFlow algorithms, with their parent SuperClusters and/or tracks. They are used as input for the production of electronGsfTracks.'],
    '31':['ecalDrivenElectronSeeds','reco::ElectronSeedCollection','Collection of ElectronSeeds with their parent SuperClusters, using the SuperCluster driven pixel matching algorithm, made for electron track seeding. That`s one of the two collections which are mixed into electronMergedSeeds just above.'],
    '32':['egammaCTFFinalFitWithMaterial','reco::TrackExtraCollection','TrackExtras for Si strip seeded electrons (not produced in standard sequence)'] 
}

aod_title = "RecoEgamma collections (in AOD only)"

aod = {
    '0':['gsfElectronCores','reco::GsfElectronCoreCollection','Offline electron object containing a minimal set of information: reference to GsfTrack, reference to SuperCluster (Egamma and/or PFlow cluster) and provenance information. The collection is unpreselected and contains electrons reconstructed either by Egamma or by PF algorithm.'],
    '1':['gsfElectrons','reco::GsfElectronCollection','Offline electron containing all the relevant observables and corrections. The collection is produced starting from GsfElectronCore and its modularity allows to easily recompute electron variables skipping the reconstruction step. Each electron has also a reference to the corresponding `core` partner. (standard offline electron collection)'],
    '2':['uncleanedOnlyGsfElectronCores', 'recoGsfElectronCores', 'No documentation'] ,
    '3':['uncleanedOnlyGsfElectrons', 'recoGsfElectrons', 'No documentation'] ,
    '4':['eidRobustLoose', 'floatedmValueMap', 'No documentation'] ,
    '5':['eidRobustTight', 'floatedmValueMap', 'No documentation'] ,
    '6':['eidRobustHighEnergy', 'floatedmValueMap', 'No documentation'] ,
    '7':['eidLoose', 'floatedmValueMap', 'No documentation'] ,
    '8':['eidTight', 'floatedmValueMap', 'No documentation'] ,
    '9':['photonCore', 'recoPhotonCores', 'No documentation'] ,
    '10':['photons','reco::PhotonCollection','Photons with all the related observables like vertex, shower shapes, isolation. Each photon has also a reference to the corresponding `core` partner. (standard reconstructed photon collection)'],
    '11':['conversions', 'recoConversions', 'Converted photons'] ,
    '12':['allConversions', 'recoConversions', 'No documentation'] ,
    '13':['ckfOutInTracksFromConversions', 'recoTracks', 'Conversion tracks from outside-in tracking.'] ,
    '14':['ckfInOutTracksFromConversions', 'recoTracks', 'Conversion tracks from inside-out tracking.'] ,
    '15':['uncleanedOnlyAllConversions', 'recoConversions', 'No documentation'] ,
    '16':['uncleanedOnlyCkfOutInTracksFromConversions', 'recoTracks', 'No documentation'] ,
    '17':['uncleanedOnlyCkfInOutTracksFromConversions', 'recoTracks', 'No documentation'] ,
    '18':['PhotonIDProd','reco::PhotonIDCollection','Photons identification variables, calculated for corresponding photons. (standard reconstructed photon collection)'],
    '19':['hfRecoEcalCandidate', '*', 'No documentation'] ,
    '20':['hfEMClusters', '*', 'No documentation'] 
}

