// -*- C++ -*-
//
// Package:    GoodVertexFilter
// Class:      GoodVertexFilter
// 
/**\class GoodVertexFilter GoodVertexFilter.cc DPGAnalysis/GoodVertexFilter/src/GoodVertexFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea RIZZI
//         Created:  Mon Dec  7 18:02:10 CET 2009
// $Id: GoodVertexFilter.cc,v 1.4 2010/02/28 20:10:01 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
// class declaration
//

class GoodVertexFilter : public edm::global::EDFilter<> {
   public:
      explicit GoodVertexFilter(const edm::ParameterSet&);
      ~GoodVertexFilter();

   private:
      virtual bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      const edm::EDGetTokenT<reco::VertexCollection> vertexSrc;        
      const unsigned int minNDOF;
      const double maxAbsZ;
      const double maxd0;
      // ----------member data ---------------------------
};

GoodVertexFilter::GoodVertexFilter(const edm::ParameterSet& iConfig) :
  vertexSrc{ consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection")) },
  minNDOF{ iConfig.getParameter<unsigned int>("minimumNDOF") },
  maxAbsZ{ iConfig.getParameter<double>("maxAbsZ") },
  maxd0 { iConfig.getParameter<double>("maxd0") }
{

}


GoodVertexFilter::~GoodVertexFilter()
{
}

bool
GoodVertexFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
 bool result = false; 
 edm::Handle<reco::VertexCollection> pvHandle; 
 iEvent.getByToken(vertexSrc,pvHandle);
 const reco::VertexCollection & vertices = *pvHandle.product();
 for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
  {
      if(it->ndof() > minNDOF && 
         ( (maxAbsZ <=0 ) || fabs(it->z()) <= maxAbsZ ) &&
         ( (maxd0 <=0 ) || fabs(it->position().rho()) <= maxd0 )
       ) result = true;
  }

   return result;
}


//define this as a plug-in
DEFINE_FWK_MODULE(GoodVertexFilter);
