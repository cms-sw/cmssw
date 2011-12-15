// -*- C++ -*-
//
// Package:    MultiplicityInvestigator
// Class:      MultiplicityInvestigator
// 
/**\class MultiplicityInvestigator MultiplicityInvestigator.cc myTKAnalyses/DigiInvestigator/src/MultiplicityInvestigator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
// $Id: MultiplicityInvestigator.cc,v 1.4 2011/02/02 11:05:50 venturia Exp $
//
//


// system include files
#include <memory>

// user include files

#include <vector>
#include <map>
#include <limits>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DPGAnalysis/SiStripTools/interface/DigiInvestigatorHistogramMaker.h"
#include "DPGAnalysis/SiStripTools/interface/DigiVertexCorrHistogramMaker.h"

//
// class decleration
//

class MultiplicityInvestigator : public edm::EDAnalyzer {
   public:
      explicit MultiplicityInvestigator(const edm::ParameterSet&);
      ~MultiplicityInvestigator();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void endJob() ;

      // ----------member data ---------------------------

  const bool _wantVtxCorrHist;
  DigiInvestigatorHistogramMaker _digiinvesthmevent;
  DigiVertexCorrHistogramMaker _digivtxcorrhmevent;

  edm::InputTag _multiplicityMap;
  edm::InputTag _vertexCollection;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MultiplicityInvestigator::MultiplicityInvestigator(const edm::ParameterSet& iConfig):
  //  _digiinvesthmevent(iConfig.getParameter<edm::ParameterSet>("digiInvestConfig")),  
  _wantVtxCorrHist(iConfig.getParameter<bool>("wantVtxCorrHist")),
  _digiinvesthmevent(iConfig),
  _digivtxcorrhmevent(iConfig.getParameter<edm::ParameterSet>("digiVtxCorrConfig")),
  _multiplicityMap(iConfig.getParameter<edm::InputTag>("multiplicityMap")),
  _vertexCollection(iConfig.getParameter<edm::InputTag>("vertexCollection"))
{
   //now do what ever initialization is needed


  _digiinvesthmevent.book("EventProcs");
  if(_wantVtxCorrHist) _digivtxcorrhmevent.book("VtxCorr");

}


MultiplicityInvestigator::~MultiplicityInvestigator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MultiplicityInvestigator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Handle<std::map<unsigned int, int> > mults;
  iEvent.getByLabel(_multiplicityMap,mults);
  
  _digiinvesthmevent.fill(iEvent.orbitNumber(),*mults);
  
  if(_wantVtxCorrHist) {
    Handle<reco::VertexCollection> vertices;
    iEvent.getByLabel(_vertexCollection,vertices);

    _digivtxcorrhmevent.fill(vertices->size(),*mults);
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
MultiplicityInvestigator::beginJob()
{

}

void
MultiplicityInvestigator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  _digiinvesthmevent.beginRun(iRun.run());

}

void
MultiplicityInvestigator::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
}
// ------------ method called once each job just after ending the event loop  ------------
void 
MultiplicityInvestigator::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(MultiplicityInvestigator);
