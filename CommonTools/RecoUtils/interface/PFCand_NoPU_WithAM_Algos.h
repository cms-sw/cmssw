#ifndef PFCand_NoPU_WithAM_Algos_h
#define PFCand_NoPU_WithAM_Algos_h

// -*- C++ -*-
//
// Package:    PFCand_NoPU_WithAM
// Class:      PFCand_NoPU_WithAM
//
/**\class PF_PU_AssoMap PFCand_NoPU_WithAM.cc CommonTools/RecoUtils/plugins/PFCand_NoPU_WithAM.cc

 Description: Algorithms for the producer a collection of PFCandidates associated to the first vertex based on the association map

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//         Created:  Thu Dec  1 16:07:41 CET 2011
// $Id: PFCand_NoPU_WithAM_Algos.h,v 1.3 2012/06/06 09:06:59 mgeisler Exp $
//
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

class PFCand_NoPU_WithAM_Algos{
 public:
   typedef reco::BeamSpot BeamSpot;
   typedef reco::ConversionCollection ConversionCollection;
   typedef reco::PFCandidateCollection PFCandidateCollection;
   typedef reco::PFCandidateRef PFCandidateRef;
   typedef reco::PFDisplacedVertex PFDisplacedVertex;
   typedef reco::PFDisplacedVertexCollection PFDisplacedVertexCollection;
   typedef reco::RecoCandidate RecoCandidate;
   typedef reco::TrackCollection TrackCollection;
   typedef reco::VertexCollection VertexCollection;
   typedef reco::VertexCompositeCandidateCollection VertexCompositeCandidateCollection;
   typedef reco::VertexRef VertexRef;

   typedef edm::AssociationMap<edm::OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;
   typedef edm::AssociationMap<edm::OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;
   typedef std::pair<PFCandidateRef, float> PFCandQualityPair;
   typedef std::pair<VertexRef, PFCandQualityPair> VertexPfcQuality;
  
   //function to find the vertex with the highest TrackWeight for a certain track
   static VertexPfcQuality TrackWeightAssociation(const PFCandidateRef, edm::Handle<VertexCollection>); 

   //function to find the closest vertex in z for a certain point
   static VertexRef FindClosestInZ(double, edm::Handle<VertexCollection>);

   //function to associate the track to the closest vertex in z
   static VertexPfcQuality AssociateClosestInZ(const PFCandidateRef, edm::Handle<VertexCollection>);

   //function to compare two pfcandidates
   static bool Match(const PFCandidateRef, const RecoCandidate*);

   //function to find out if the track comes from a gamma conversion
   static bool ComesFromConversion(const PFCandidateRef, edm::Handle<ConversionCollection>, edm::Handle<VertexCollection>, VertexRef*);  

   //function to find the best vertex for a pfCandidate 
   static VertexRef FindPFCandVertex(const PFCandidateRef, edm::Handle<VertexCollection>);   

   //function to find out if the track comes from a V0 decay
   static bool ComesFromV0Decay(const PFCandidateRef, edm::Handle<VertexCompositeCandidateCollection>, 
	 	 	  	edm::Handle<VertexCompositeCandidateCollection>, edm::Handle<VertexCollection>, VertexRef*); 

   //function to find out if the track comes from a nuclear interaction
   static bool ComesFromNI(const PFCandidateRef, edm::Handle<PFDisplacedVertexCollection>, PFDisplacedVertex*, const edm::EventSetup&);
   
   static VertexRef FindNIVertex(const PFCandidateRef, PFDisplacedVertex, edm::Handle<VertexCollection>); 
   
   //function to check if a secondary is compatible with the BeamSpot
   static bool CheckBeamSpotCompability(const PFCandidateRef, edm::Handle<BeamSpot>); 

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   static std::auto_ptr<PFCandVertexAssMap> SortAssociationMap(PFCandVertexAssMap*);

 protected:
  //protected functions 

 private: 


};

#endif
