'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoEgamma collections (in RECO and AOD)",
    "data": [
     {
      "instance": "egammaCTFFinalFitWithMaterial",
      "container": "reco::TrackCollection",
      "desc": "Tracks for Si strip seeded electrons (not produced in standard sequence)"
     },
     {
      "instance": "siStripElectrons",
      "container": "reco::SiStripElectronCollection",
      "desc": "Intermediate object for Si Strip electron reconstruction containing information about Si strip hits associated to the SuperCluster (not produced in standard sequence)"
     },
     {
      "instance": "hfEMClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "electronGsfTracks",
      "container": "reco::GsfTrackCollection",
      "desc": "GsfTracks for GsfElectrons (produced in a dedicated module in the standard reconstruction sequence). They have been produced from the electronMergedSeeds described below."
     },
     {
      "instance": "eidTYPE",
      "container": " ",
      "desc": "ValueMaps containing a reference to a CMS.GsfElectron and the result of the electron ID. The event has one Value Map per identification TYPE (eidLoose, eidRobust...) (standard electron ID computed in standard sequence)"
     },
     {
      "instance": "siStripElectronToTrackAssociator",
      "container": "reco::ElectronCollection",
      "desc": "Electrons reconstructed using tracks seeded in the Si strip layers (not produced in standard sequence)"
     },
     {
      "instance": "gsfElectrons",
      "container": "reco::GsfElectronCollection",
      "desc": "Offline electron containing all the relevant observables and corrections. The collection is produced starting from GsfElectronCore and its modularity allows to easily recompute electron variables skipping the reconstruction step. Each electron has also a reference to the corresponding `core` partner. (standard offline electron collection)"
     },
     {
      "instance": "gsfElectronCores",
      "container": "reco::GsfElectronCoreCollection",
      "desc": "Offline electron object containing a minimal set of information: reference to GsfTrack, reference to SuperCluster (Egamma and/or PFlow cluster) and provenance information. The collection is unpreselected and contains electrons reconstructed either by Egamma or by PF algorithm."
     },
     {
      "instance": "uncleanedOnlyGsfElectrons",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyGsfElectronCores",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustTight",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustLoose",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "eidLoose",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustHighEnergy",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "conversions",
      "container": "recoConversions",
      "desc": "Converted photons"
     },
     {
      "instance": "eidTight",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "photons",
      "container": "reco::PhotonCollection",
      "desc": "Photons with all the related observables like vertex, shower shapes, isolation. Each photon has also a reference to the corresponding `core` partner. (standard reconstructed photon collection)"
     },
     {
      "instance": "photonCore",
      "container": "reco::PhotonCoreCollection",
      "desc": "Photon objects containing reference to SuperCluster, Conversions and ElectronSeeds."
     },
     {
      "instance": "ckfOutInTracksFromConversions",
      "container": "recoTracks",
      "desc": "Conversion tracks from outside-in tracking."
     },
     {
      "instance": "allConversions",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyAllConversions",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "ckfInOutTracksFromConversions",
      "container": "recoTracks",
      "desc": "Conversion tracks from inside-out tracking."
     },
     {
      "instance": "uncleanedOnlyCkfInOutTracksFromConversions",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyCkfOutInTracksFromConversions",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hfRecoEcalCandidate",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "PhotonIDProd",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoEgamma collections (in AOD only)",
    "data": [
     {
      "instance": "conversions",
      "container": "recoConversions",
      "desc": "Converted photons"
     },
     {
      "instance": "photons",
      "container": "reco::PhotonCollection",
      "desc": "Photons with all the related observables like vertex, shower shapes, isolation. Each photon has also a reference to the corresponding `core` partner. (standard reconstructed photon collection)"
     },
     {
      "instance": "ckfOutInTracksFromConversions",
      "container": "recoTracks",
      "desc": "Conversion tracks from outside-in tracking."
     },
     {
      "instance": "allConversions",
      "container": "recoConversions",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyAllConversions",
      "container": "recoConversions",
      "desc": "No documentation"
     },
     {
      "instance": "ckfInOutTracksFromConversions",
      "container": "recoTracks",
      "desc": "Conversion tracks from inside-out tracking."
     },
     {
      "instance": "uncleanedOnlyCkfInOutTracksFromConversions",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyCkfOutInTracksFromConversions",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "hfRecoEcalCandidate",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "PhotonIDProd",
      "container": "reco::PhotonIDCollection",
      "desc": "Photons identification variables, calculated for corresponding photons. (standard reconstructed photon collection)"
     },
     {
      "instance": "hfEMClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "gsfElectrons",
      "container": "reco::GsfElectronCollection",
      "desc": "Offline electron containing all the relevant observables and corrections. The collection is produced starting from GsfElectronCore and its modularity allows to easily recompute electron variables skipping the reconstruction step. Each electron has also a reference to the corresponding `core` partner. (standard offline electron collection)"
     },
     {
      "instance": "gsfElectronCores",
      "container": "reco::GsfElectronCoreCollection",
      "desc": "Offline electron object containing a minimal set of information: reference to GsfTrack, reference to SuperCluster (Egamma and/or PFlow cluster) and provenance information. The collection is unpreselected and contains electrons reconstructed either by Egamma or by PF algorithm."
     },
     {
      "instance": "uncleanedOnlyGsfElectrons",
      "container": "recoGsfElectrons",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyGsfElectronCores",
      "container": "recoGsfElectronCores",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustTight",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustLoose",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "eidLoose",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustHighEnergy",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "photonCore",
      "container": "recoPhotonCores",
      "desc": "No documentation"
     },
     {
      "instance": "eidTight",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoEgamma collections (in RECO only)",
    "data": [
     {
      "instance": "uncleanedOnlyCkfOutInTracksFromConversions",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "electronMergedSeeds",
      "container": "reco::ElectronSeedCollection",
      "desc": "Mixed collection of ElectronSeeds coming from Egamma and/or PFlow algorithms, with their parent SuperClusters and/or tracks. They are used as input for the production of electronGsfTracks."
     },
     {
      "instance": "uncleanedOnlyCkfInOutTracksFromConversions",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyCkfOutInTracksFromConversions",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "PhotonIDProd",
      "container": "reco::PhotonIDCollection",
      "desc": "Photons identification variables, calculated for corresponding photons. (standard reconstructed photon collection)"
     },
     {
      "instance": "hfRecoEcalCandidate",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyCkfOutInTracksFromConversions",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "egammaCTFFinalFitWithMaterial",
      "container": "reco::TrackExtraCollection",
      "desc": "TrackExtras for Si strip seeded electrons (not produced in standard sequence)"
     },
     {
      "instance": "uncleanedOnlyCkfInOutTracksFromConversions",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "conversions",
      "container": "recoConversions",
      "desc": "Converted photons"
     },
     {
      "instance": "photonCore",
      "container": "reco::PhotonCoreCollection",
      "desc": "Photon objects containing reference to SuperCluster, Conversions and ElectronSeeds."
     },
     {
      "instance": "ckfOutInTracksFromConversions",
      "container": "recoTracks",
      "desc": "Conversion tracks from outside-in tracking."
     },
     {
      "instance": "allConversions",
      "container": "recoConversions",
      "desc": "No documentation"
     },
     {
      "instance": "ckfOutInTracksFromConversions",
      "container": "recoTrackExtras",
      "desc": "TrackExtras and TrackingRecHits for conversion tracks from outside-in tracking."
     },
     {
      "instance": "ckfInOutTracksFromConversions",
      "container": "recoTracks",
      "desc": "Conversion tracks from inside-out tracking."
     },
     {
      "instance": "ckfOutInTracksFromConversions",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "ckfInOutTracksFromConversions",
      "container": "recoTrackExtras",
      "desc": "TrackExtras and TrackingRecHits for conversion tracks from inside-out tracking."
     },
     {
      "instance": "uncleanedOnlyAllConversions",
      "container": "recoConversions",
      "desc": "No documentation"
     },
     {
      "instance": "ckfInOutTracksFromConversions",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "ecalDrivenElectronSeeds",
      "container": "reco::ElectronSeedCollection",
      "desc": "Collection of ElectronSeeds with their parent SuperClusters, using the SuperCluster driven pixel matching algorithm, made for electron track seeding. That`s one of the two collections which are mixed into electronMergedSeeds just above."
     },
     {
      "instance": "uncleanedOnlyCkfInOutTracksFromConversions",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "hfEMClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "gsfElectronsGsfFit",
      "container": "reco::GsfTrackExtraCollection reco::TrackExtraCollection TrackingRecHitCollection",
      "desc": "GsfTrackExtras, TrackExtras, TrackingRecHits"
     },
     {
      "instance": "gsfElectrons",
      "container": "reco::GsfElectronCollection",
      "desc": "Offline electron containing all the relevant observables and corrections. The collection is produced starting from GsfElectronCore and its modularity allows to easily recompute electron variables skipping the reconstruction step. Each electron has also a reference to the corresponding `core` partner. (standard offline electron collection)"
     },
     {
      "instance": "gsfElectronCores",
      "container": "reco::GsfElectronCoreCollection",
      "desc": "Offline electron object containing a minimal set of information: reference to GsfTrack, reference to SuperCluster (Egamma and/or PFlow cluster) and provenance information. The collection is unpreselected and contains electrons reconstructed either by Egamma or by PF algorithm."
     },
     {
      "instance": "uncleanedOnlyGsfElectrons",
      "container": "recoGsfElectrons",
      "desc": "No documentation"
     },
     {
      "instance": "uncleanedOnlyGsfElectronCores",
      "container": "recoGsfElectronCores",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustTight",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustLoose",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "eidLoose",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "eidRobustHighEnergy",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     },
     {
      "instance": "photons",
      "container": "reco::PhotonCollection",
      "desc": "Photons with all the related observables like vertex, shower shapes, isolation. Each photon has also a reference to the corresponding `core` partner. (standard reconstructed photon collection)"
     },
     {
      "instance": "eidTight",
      "container": "floatedmValueMap",
      "desc": "No documentation"
     }
    ]
  }
}
