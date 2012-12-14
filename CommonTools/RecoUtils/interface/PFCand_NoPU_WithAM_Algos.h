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
// $Id: PFCand_NoPU_WithAM_Algos.h,v 1.4 2012/06/21 22:34:15 wmtan Exp $
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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

typedef edm::AssociationMap<edm::OneToManyWithQuality< reco::VertexCollection, reco::PFCandidateCollection, float> > PFCandVertexAssMap;

class PFCand_NoPU_WithAM_Algos{
 public: 

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   static std::auto_ptr<PFCandVertexAssMap> SortAssociationMap(PFCandVertexAssMap*);

 protected:
  //protected functions 

 private: 
  //private methods for internal usage


};

#endif
