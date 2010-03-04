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
// $Id: Vx3DHLTAnalyzer.cc,v 1.6 2010/02/26 15:32:04 dinardo Exp $
//
//


#include "DQM/BeamMonitor/plugins/Vx3DHLTAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


using namespace std;
using namespace reco;
using namespace edm;


Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const ParameterSet& iConfig)
{
  vertexCollection = edm::InputTag("pixelVertices");
  nLumiReset = 0;

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

	Vx_ZX->Fill(it3DVx->z(), it3DVx->x());
	Vx_ZY->Fill(it3DVx->z(), it3DVx->y());
	Vx_XY->Fill(it3DVx->x(), it3DVx->y());

	Vx_ZX_profile->Fill(it3DVx->z(), it3DVx->x());
	Vx_ZY_profile->Fill(it3DVx->z(), it3DVx->y());

	reportSummary->Fill(1.0);
      }
  }
}


void Vx3DHLTAnalyzer::endLuminosityBlock(const LuminosityBlock& lumiBlock,
					 const EventSetup& iSetup)
{
  lumiCounter++;

  if ((lumiCounter == nLumiReset) && (nLumiReset != 0))
    {
      mXlumi->ShiftFillLast(Vx_X->getTH1F()->GetMean(), Vx_X->getTH1F()->GetMeanError(), lumiBlock.luminosityBlock());
      mYlumi->ShiftFillLast(Vx_Y->getTH1F()->GetMean(), Vx_Y->getTH1F()->GetMeanError(), lumiBlock.luminosityBlock());
      mZlumi->ShiftFillLast(Vx_Z->getTH1F()->GetMean(), Vx_Z->getTH1F()->GetMeanError(), lumiBlock.luminosityBlock());

      sXlumi->ShiftFillLast(Vx_X->getTH1F()->GetRMS(), Vx_X->getTH1F()->GetRMSError(), lumiBlock.luminosityBlock());
      sYlumi->ShiftFillLast(Vx_Y->getTH1F()->GetRMS(), Vx_Y->getTH1F()->GetRMSError(), lumiBlock.luminosityBlock());
      sZlumi->ShiftFillLast(Vx_Z->getTH1F()->GetRMS(), Vx_Z->getTH1F()->GetRMSError(), lumiBlock.luminosityBlock());

      Vx_X->Reset();
      Vx_Y->Reset();
      Vx_Z->Reset();

      Vx_ZX->Reset();
      Vx_ZY->Reset();
      Vx_XY->Reset();

      Vx_ZX_profile->Reset();
      Vx_ZY_profile->Reset();

      reportSummary->Fill(0.5);

      lumiCounter = 0;
    }
  else if (nLumiReset == 0) { lumiCounter = 0; reportSummary->Fill(0.5); }
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
 
      mXlumi = dbe->book1D("muX vs lumi", "\\mu_{x} vs. Lumisection", 50, 0.5, 50.5);
      mYlumi = dbe->book1D("muY vs lumi", "\\mu_{y} vs. Lumisection", 50, 0.5, 50.5);
      mZlumi = dbe->book1D("muZ vs lumi", "\\mu_{z} vs. Lumisection", 50, 0.5, 50.5);

      mXlumi->setAxisTitle("Lumisection [#]",1);
      mXlumi->setAxisTitle("Entries [#]",2);
      mXlumi->getTH1()->SetOption("E1");
      mYlumi->setAxisTitle("Lumisection [#]",1);
      mYlumi->setAxisTitle("Entries [#]",2);
      mYlumi->getTH1()->SetOption("E1");
      mZlumi->setAxisTitle("Lumisection [#]",1);
      mZlumi->setAxisTitle("Entries [#]",2);
      mZlumi->getTH1()->SetOption("E1");

      sXlumi = dbe->book1D("sigmaX vs lumi", "\\sigma_{x} vs. Lumisection", 50, 0.5, 50.5);
      sYlumi = dbe->book1D("sigmaY vs lumi", "\\sigma_{y} vs. Lumisection", 50, 0.5, 50.5);
      sZlumi = dbe->book1D("sigmaZ vs lumi", "\\sigma_{z} vs. Lumisection", 50, 0.5, 50.5);

      sXlumi->setAxisTitle("Lumisection [#]",1);
      sXlumi->setAxisTitle("Entries [#]",2);
      sXlumi->getTH1()->SetOption("E1");
      sYlumi->setAxisTitle("Lumisection [#]",1);
      sYlumi->setAxisTitle("Entries [#]",2);
      sYlumi->getTH1()->SetOption("E1");
      sZlumi->setAxisTitle("Lumisection [#]",1);
      sZlumi->setAxisTitle("Entries [#]",2);
      sZlumi->getTH1()->SetOption("E1");

      Vx_ZX = dbe->book2D("Vertex_ZX", "Primary Vertex ZX Coordinate Distributions", 80, -20.0, 20.0, 400, -2.0, 2.0);
      Vx_ZY = dbe->book2D("Vertex_ZY", "Primary Vertex ZY Coordinate Distributions", 80, -20.0, 20.0, 400, -2.0, 2.0);
      Vx_XY = dbe->book2D("Vertex_XY", "Primary Vertex XY Coordinate Distributions", 400, -2.0, 2.0, 400, -2.0, 2.0);

      Vx_ZX->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZX->setAxisTitle("Primary Vertices X [cm]",2);
      Vx_ZX->setAxisTitle("Entries [#]",3);
      Vx_ZY->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_ZY->setAxisTitle("Entries [#]",3);
      Vx_XY->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_XY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_XY->setAxisTitle("Entries [#]",3);

      Vx_ZX_profile = dbe->bookProfile("ZX profile","ZX Profile", 40, -20.0, 20.0, 200, -2.0, 2.0, "");
      Vx_ZX_profile->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZX_profile->setAxisTitle("Primary Vertices X [cm]",2);

      Vx_ZY_profile = dbe->bookProfile("ZY profile","ZY Profile", 40, -20.0, 20.0, 200, -2.0, 2.0, "");
      Vx_ZY_profile->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZY_profile->setAxisTitle("Primary Vertices Y [cm]",2);

      dbe->setCurrentFolder("BeamPixel/EventInfo");
      reportSummary = dbe->bookFloat("reportSummary");
      reportSummary->Fill(0.);
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
