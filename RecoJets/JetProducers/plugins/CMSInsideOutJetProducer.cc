////////////////////////////////////////////////////////////////////////////////
//
// CMSInsideOutJetProducer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetProducers/plugins/CMSInsideOutJetProducer.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"


#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "fastjet/SISConePlugin.hh"
#include "fastjet/CMSIterativeConePlugin.hh"
#include "fastjet/ATLASConePlugin.hh"
#include "fastjet/CDFMidPointPlugin.hh"


#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;



////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
CMSInsideOutJetProducer::CMSInsideOutJetProducer(const edm::ParameterSet& iConfig)
  : VirtualJetProducer( iConfig ),
    alg_( iConfig.getParameter<double>("seedObjectPt"),
	  iConfig.getParameter<double>("growthParameter"), 
	  iConfig.getParameter<double>("maxSize"), 
	  iConfig.getParameter<double>("minSize") )
{
}


//______________________________________________________________________________
CMSInsideOutJetProducer::~CMSInsideOutJetProducer()
{
} 


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void CMSInsideOutJetProducer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  VirtualJetProducer::produce( iEvent, iSetup );
}

//______________________________________________________________________________
void CMSInsideOutJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup)
{

  fjJets_.clear();

  alg_.run( fjInputs_, fjJets_ );
}



////////////////////////////////////////////////////////////////////////////////
// define as cmssw plugin
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(CMSInsideOutJetProducer);



