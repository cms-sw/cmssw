// -*- C++ -*-
//
// Package:    Vx3DHLTAnalyzer
// Class:      Vx3DHLTAnalyzer
// 
/**\class Vx3DHLTAnalyzer Vx3DHLTAnalyzer.cc Vx3DHLTAnalysis/Vx3DHLTAnalyzer/src/Vx3DHLTAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mauro Dinardo,28 S-020,+41227673777,
//         Created:  Tue Feb 23 13:15:31 CET 2010
// $Id: Vx3DHLTAnalyzer.cc,v 1.3 2010/02/24 21:25:07 ameyer Exp $
//
//


#include "DQM/BeamMonitor/plugins/Vx3DHLTAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


using namespace std;
using namespace reco;
using namespace edm;

Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const edm::ParameterSet& iConfig)
{

  vertexCollection  = iConfig.getParameter<edm::InputTag>("vertexCollection");

}


Vx3DHLTAnalyzer::~Vx3DHLTAnalyzer()
{
}


void Vx3DHLTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<VertexCollection> Vx3DCollection;
  iEvent.getByLabel(vertexCollection,Vx3DCollection);

  for (vector<Vertex>::const_iterator it3DVx = Vx3DCollection->begin(); it3DVx != Vx3DCollection->end(); it3DVx++) {

    if ((it3DVx->isValid() == true) && (it3DVx->isFake() == false))
      {
	Vx_X->Fill(it3DVx->x());
	Vx_Y->Fill(it3DVx->y());
	Vx_Z->Fill(it3DVx->z());
      }
  }
}


void Vx3DHLTAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
					 const edm::EventSetup& iSetup)
{
  // Put this in as a first-pass for figuring out the rate
  // each lumi block is 93 seconds in length
  Vx_X->Reset();
  Vx_Y->Reset();
  Vx_Z->Reset();
}


void Vx3DHLTAnalyzer::beginJob()
{
  DQMStore* dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
 
  if ( dbe ) 
    {
      dbe->setCurrentFolder("BeamPixel");

      Vx_X = dbe->book1D("Vertex_X", "Primary Vertex X Coordinate Distribution", 1000, -5.0, 5.0);
      Vx_Y = dbe->book1D("Vertex_Y", "Primary Vertex Y Coordinate Distribution", 1000, -5.0, 5.0);
      Vx_Z = dbe->book1D("Vertex_Z", "Primary Vertex Z Coordinate Distribution", 800, -20.0, 20.0);
      
      dbe->setCurrentFolder("BeamPixel/EventInfo");
      reportSummary = dbe->bookFloat("reportSummary");
      reportSummary->Fill(1.);
      reportSummaryMap = dbe->book2D("reportSummaryMap","Beam Pixel Summary Map", 1,0.,1.,1,0.,1.);
      dbe->setCurrentFolder("BeamPixel/EventInfo/reportSummaryContents");
    }
}


void Vx3DHLTAnalyzer::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(Vx3DHLTAnalyzer);
