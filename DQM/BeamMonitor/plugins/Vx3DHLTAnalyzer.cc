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
// $Id: Vx3DHLTAnalyzer.cc,v 1.5 2010/02/26 13:20:38 dinardo Exp $
//
//


#include "DQM/BeamMonitor/plugins/Vx3DHLTAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


using namespace std;
using namespace reco;
using namespace edm;


Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const ParameterSet& iConfig)
{
  vertexCollection = iConfig.getParameter<InputTag>("vertexCollection");
  nLumiReset       = iConfig.getParameter<unsigned int>("nLumiReset");
}


Vx3DHLTAnalyzer::~Vx3DHLTAnalyzer()
{
}


void Vx3DHLTAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  Handle<VertexCollection> Vx3DCollection;
  iEvent.getByLabel(vertexCollection,Vx3DCollection);

  for (vector<Vertex>::const_iterator it3DVx = Vx3DCollection->begin(); it3DVx != Vx3DCollection->end(); it3DVx++) {

    if ((it3DVx->isValid() == true) && (it3DVx->isFake() == false))
      {
	Vx_X->Fill(it3DVx->x());
	Vx_Y->Fill(it3DVx->y());
	Vx_Z->Fill(it3DVx->z());

	Vx_XZ->Fill(it3DVx->x(), it3DVx->z());
	Vx_YZ->Fill(it3DVx->y(), it3DVx->z());
	Vx_XY->Fill(it3DVx->x(), it3DVx->y());
      }
  }
}


void Vx3DHLTAnalyzer::endLuminosityBlock(const LuminosityBlock& lumiBlock,
					 const EventSetup& iSetup)
{
  lumiCounter++;

  if ((lumiCounter == nLumiReset) && (nLumiReset != 0))
    {
      Vx_X->Reset();
      Vx_Y->Reset();
      Vx_Z->Reset();

      Vx_XZ->Reset();
      Vx_YZ->Reset();
      Vx_XY->Reset();

      lumiCounter = 0;
    }
  else if (nLumiReset == 0) lumiCounter = 0;
}


void Vx3DHLTAnalyzer::beginJob()
{
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();
 
  if ( dbe ) 
    {
      dbe->setCurrentFolder("BeamPixel");

      Vx_X = dbe->book1D("Vertex_X", "Primary Vertex X Coordinate Distribution", 4000, -2.0, 2.0);
      Vx_Y = dbe->book1D("Vertex_Y", "Primary Vertex Y Coordinate Distribution", 4000, -2.0, 2.0);
      Vx_Z = dbe->book1D("Vertex_Z", "Primary Vertex Z Coordinate Distribution", 800, -20.0, 20.0);

      Vx_X->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_X->setAxisTitle("Entries [#]",2);
      Vx_Y->setAxisTitle("Primary Vertices Y [cm]",1);
      Vx_Y->setAxisTitle("Entries [#]",2);
      Vx_Z->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_Z->setAxisTitle("Entries [#]",2);
 
      Vx_XZ = dbe->book2D("Vertex_XZ", "Primary Vertex XZ Coordinate Distributions", 400, -2.0, 2.0, 80, -20.0, 20.0);
      Vx_YZ = dbe->book2D("Vertex_YZ", "Primary Vertex YZ Coordinate Distributions", 400, -2.0, 2.0, 80, -20.0, 20.0);
      Vx_XY = dbe->book2D("Vertex_XY", "Primary Vertex XY Coordinate Distributions", 400, -2.0, 2.0, 400, -2.0, 2.0);

      Vx_XZ->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_XZ->setAxisTitle("Primary Vertices Z [cm]",2);
      Vx_XZ->setAxisTitle("Entries [#]",3);
      Vx_YZ->setAxisTitle("Primary Vertices Y [cm]",1);
      Vx_YZ->setAxisTitle("Primary Vertices Z [cm]",2);
      Vx_YZ->setAxisTitle("Entries [#]",3);
      Vx_XY->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_XY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_XY->setAxisTitle("Entries [#]",3);

      dbe->setCurrentFolder("BeamPixel/EventInfo");
      reportSummary = dbe->bookFloat("reportSummary");
      reportSummary->Fill(1.);
      reportSummaryMap = dbe->book2D("reportSummaryMap","Beam Pixel Summary Map", 1, 0., 1., 1, 0., 1.);
      dbe->setCurrentFolder("BeamPixel/EventInfo/reportSummaryContents");
    }

  lumiCounter = 0;
}


void Vx3DHLTAnalyzer::endJob()
{
}


// Define this as a plug-in
DEFINE_FWK_MODULE(Vx3DHLTAnalyzer);
