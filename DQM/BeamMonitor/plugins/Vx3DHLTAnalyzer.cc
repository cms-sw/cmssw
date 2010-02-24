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
// $Id$
//
//


#include "DQM/BeamMonitor/plugins/Vx3DHLTAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


using namespace std;
using namespace reco;


Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const edm::ParameterSet& iConfig)
{
}


Vx3DHLTAnalyzer::~Vx3DHLTAnalyzer()
{
}


void Vx3DHLTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<VertexCollection> Vx3DCollection;
  iEvent.getByLabel("hlt3DPixelVertices",Vx3DCollection);

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
      dbe->setCurrentFolder("BeamMonitor");
      dbe->rmdir("BeamMonitor");
    }
 
  if ( dbe ) 
    {
      dbe->setCurrentFolder("BeamMonitor");

      Vx_X = dbe->book1D("Vertex_X", "Primary Vertex X Coordinate Distribution", 1000, -5.0, 5.0);
      Vx_Y = dbe->book1D("Vertex_Y", "Primary Vertex Y Coordinate Distribution", 1000, -5.0, 5.0);
      Vx_Z = dbe->book1D("Vertex_Z", "Primary Vertex Z Coordinate Distribution", 800, -20.0, 20.0);
    }
}


void Vx3DHLTAnalyzer::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(Vx3DHLTAnalyzer);
