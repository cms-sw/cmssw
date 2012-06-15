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
// $Id: PFCand_NoPU_WithAM_Algos.h,v 1.1 2012/04/17 11:54:46 mgeisler Exp $
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

   
using namespace edm;
using namespace std;
using namespace reco;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;
  typedef AssociationMap<OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;

class PFCand_NoPU_WithAM_Algos{
 public:

   //function to find the closest vertex in z for a certain point
   static VertexRef FindClosestInZ(double, Handle<VertexCollection>);

   //function to compare two pfcandidates
   static bool Match(const PFCandidateRef, const RecoCandidate*);

   //function to find out if the track comes from a gamma conversion
   static bool ComesFromConversion(const PFCandidateRef, Handle<ConversionCollection>, Handle<VertexCollection>, VertexRef*);  

   //function to find the best vertex for a pfCandidate 
   static VertexRef FindPFCandVertex(const PFCandidateRef, Handle<VertexCollection>);   

   //function to find out if the track comes from a V0 decay
   static bool ComesFromV0Decay(const PFCandidateRef, Handle<VertexCompositeCandidateCollection>, 
	 	 	  	Handle<VertexCompositeCandidateCollection>, Handle<VertexCollection>, VertexRef*); 

   //function to find out if the track comes from a nuclear interaction
   static bool ComesFromNI(const PFCandidateRef, Handle<PFDisplacedVertexCollection>, PFDisplacedVertex*, const edm::EventSetup&);
   
   static VertexRef FindNIVertex(const PFCandidateRef, PFDisplacedVertex, Handle<VertexCollection>, bool, const edm::EventSetup&);  

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   static auto_ptr<PFCandVertexAssMap> SortAssociationMap(PFCandVertexAssMap*);

 protected:
  //protected functions 

 private: 


};

#endif
