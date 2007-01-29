#ifndef RecoAlgos_ObjectSelectors_h
#define RecoAlgos_ObjectSelectors_h
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"
#include "PhysicsTools/Utilities/interface/PtMinSelector.h"
#include "PhysicsTools/Utilities/interface/EtMinSelector.h"
#include "PhysicsTools/Utilities/interface/EtaRangeSelector.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"
#include "PhysicsTools/Utilities/interface/OrSelector.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"
#include "PhysicsTools/Utilities/interface/RangeObjectPairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectPairCollectionSelector.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/ElectronSelector.h"
#include "PhysicsTools/RecoAlgos/interface/PhotonSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/Utilities/interface/MinSelector.h"
#include "PhysicsTools/CandAlgos/interface/ObjectShallowCloneSelector.h"

 extern const std::string massParamPrefix( "mass" );

 /// select all tracks (just for test)
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::TrackCollection, 
             AnySelector<reco::Track> 
           > 
         > AnyTrackSelector;

 /// select tracks above a give pt
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::TrackCollection,
             PtMinSelector<reco::Track>
           >
         > PtMinTrackSelector;

 /// select electrons above a give pt
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::ElectronCollection,
             PtMinSelector<reco::Electron> 
           >
         > PtMinElectronSelector;

 /// select photons above a give Et
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::PhotonCollection,
             EtMinSelector<reco::Photon>
           >
         > EtMinPhotonSelector;

 /// select calo jets above a give Et
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::CaloJetCollection,
             EtMinSelector<reco::CaloJet>
           >
         > EtMinCaloJetSelector;

 /// select calo jets above a give Et 
 /// storing a polymorphic collection of 
 /// shallow clone candidates
 typedef ObjectShallowCloneSelector<
           SingleElementCollectionSelector<
             reco::CaloJetCollection,
             EtMinSelector<reco::CaloJet>,
             edm::RefVector<reco::CaloJetCollection>
           >
         > EtMinCaloJetShallowCloneSelector;

  /// select electrons above a give pt
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::ElectronCollection,
		AndSelector<
		  EtaRangeSelector<reco::Electron>,
               PtMinSelector<reco::Electron> 
             >
           >
         > EtaPtMinElectronSelector;

 /// configurable single track selector
 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::TrackCollection, 
             SingleObjectSelector<reco::Track> 
           >
         > ConfigTrackSelector;

 /// select the N tracks with highest pt
 typedef ObjectSelector<
           SortCollectionSelector<
             reco::TrackCollection, 
		PtInverseComparator<reco::Track> 
           > 
         > LargestPtTrackSelector;

 /// select the N calo jets with highest Et
 typedef ObjectSelector<
           SortCollectionSelector<
             reco::CaloJetCollection, 
		EtInverseComparator<reco::CaloJet> 
           > 
         > LargestEtCaloJetSelector;

 /// select the N calo jets with highest Et
 typedef ObjectShallowCloneSelector<
           SortCollectionSelector<
             reco::CaloJetCollection, 
             EtInverseComparator<reco::CaloJet>,
             edm::RefVector<reco::CaloJetCollection>
           > 
         > LargestEtCaloJetShallowCloneSelector;

 /// select track pairs within a given mass window
 typedef ObjectSelector<
           ObjectPairCollectionSelector<
             reco::TrackCollection, 
             RangeObjectPairSelector<
               reco::Track,
               MasslessInvariantMass<reco::Track>,
               massParamPrefix
             >
           >
         > MassWindowTrackSelector;

#endif
