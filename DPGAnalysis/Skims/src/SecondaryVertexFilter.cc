// -*- C++ -*-
//
// Package:    SecondaryVertexFilter
// Class:      SecondaryVertexFilter
// 
/**\class SecondaryVertexFilter SecondaryVertexFilter.cc DPGAnalysis/SecondaryVertexFilter/src/SecondaryVertexFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea RIZZI
//         Created:  Mon Dec  7 18:02:10 CET 2009
// $Id: SecondaryVertexFilter.cc,v 1.3 2013/02/27 20:17:14 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
//
// class declaration
//

class SecondaryVertexFilter : public edm::EDFilter {
   public:
      explicit SecondaryVertexFilter(const edm::ParameterSet&);
      ~SecondaryVertexFilter();

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      edm::InputTag vertexSrc;        
      unsigned int minNumTracks;
      double maxAbsZ;
      double maxd0;
      // ----------member data ---------------------------
};

SecondaryVertexFilter::SecondaryVertexFilter(const edm::ParameterSet& iConfig)
{
  vertexSrc = iConfig.getParameter<edm::InputTag>("vertexCollection");
  minNumTracks = iConfig.getParameter<unsigned int>("minimumNumberOfTracks");
  maxAbsZ = iConfig.getParameter<double>("maxAbsZ");
  maxd0 = iConfig.getParameter<double>("maxd0");

}


SecondaryVertexFilter::~SecondaryVertexFilter()
{
}

bool
SecondaryVertexFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 bool result = false; 
 edm::Handle<reco::SecondaryVertexTagInfoCollection> pvHandle; 
 iEvent.getByLabel(vertexSrc,pvHandle);
 const reco::SecondaryVertexTagInfoCollection & vertices = *pvHandle.product();
 for(reco::SecondaryVertexTagInfoCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
  {
    if(it->nVertices() > 0) result = true;   
//   if(it->tracksSize() > minNumTracks && 
 //      ( (maxAbsZ <=0 ) || fabs(it->z()) <= maxAbsZ ) &&
   //    ( (maxd0 <=0 ) || fabs(it->position().rho()) <= maxd0 )
     //) result = true;
  }

   return result;
}


//define this as a plug-in
DEFINE_FWK_MODULE(SecondaryVertexFilter);
