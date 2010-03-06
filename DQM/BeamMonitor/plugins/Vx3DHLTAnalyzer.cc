// -*- C++ -*-
//
// Package:    Vx3DHLTAnalyzer
// Class:      Vx3DHLTAnalyzer
// 
/**\class Vx3DHLTAnalyzer Vx3DHLTAnalyzer.cc plugins/Vx3DHLTAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mauro Dinardo,28 S-020,+41227673777,
//         Created:  Tue Feb 23 13:15:31 CET 2010
// $Id: Vx3DHLTAnalyzer.cc,v 1.13 2010/03/06 19:20:24 dinardo Exp $
//
//


#include "DQM/BeamMonitor/interface/Vx3DHLTAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <iostream>
#include <fstream>

#include <TF3.h>


using namespace std;
using namespace reco;
using namespace edm;


Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const ParameterSet& iConfig)
{
  vertexCollection = edm::InputTag("pixelVertices");
  nLumiReset       = 1;
  dataFromFit      = false;
  xRange           = 0.;
  xStep            = 1.;
  yRange           = 0.;
  yStep            = 1.;
  zRange           = 0.;
  zStep            = 1.;
  fileName         = "BeamSpot_3DVxPixels.txt";

  vertexCollection = iConfig.getParameter<InputTag>("vertexCollection");
  nLumiReset       = iConfig.getParameter<unsigned int>("nLumiReset");
  dataFromFit      = iConfig.getParameter<bool>("dataFromFit");
  xRange           = iConfig.getParameter<double>("xRange");
  xStep            = iConfig.getParameter<double>("xStep");
  yRange           = iConfig.getParameter<double>("yRange");
  yStep            = iConfig.getParameter<double>("yStep");
  zRange           = iConfig.getParameter<double>("zRange");
  zStep            = iConfig.getParameter<double>("zStep");
  fileName         = iConfig.getParameter<string>("fileName");
}


Vx3DHLTAnalyzer::~Vx3DHLTAnalyzer()
{
}


void Vx3DHLTAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  Handle<VertexCollection> Vx3DCollection;
  iEvent.getByLabel(vertexCollection,Vx3DCollection);

  if (beginTimeOfFit != 0)
    {
      for (vector<Vertex>::const_iterator it3DVx = Vx3DCollection->begin(); it3DVx != Vx3DCollection->end(); it3DVx++) {
	
	if ((it3DVx->isValid() == true) && (it3DVx->isFake() == false))
	  {
	    Distribution3D->Fill(it3DVx->x(), it3DVx->y(), it3DVx->z());

	    Vx_X->Fill(it3DVx->x());
	    Vx_Y->Fill(it3DVx->y());
	    Vx_Z->Fill(it3DVx->z());
	    
	    Vx_ZX->Fill(it3DVx->z(), it3DVx->x());
	    Vx_ZY->Fill(it3DVx->z(), it3DVx->y());
	    Vx_XY->Fill(it3DVx->x(), it3DVx->y());
	   
	    Vx_ZX_profile->Fill(it3DVx->z(), it3DVx->x());
	    Vx_ZY_profile->Fill(it3DVx->z(), it3DVx->y());
	   
	    reportSummary->Fill(1.0);
	    reportSummaryMap->Fill(0.5, 0.5, 1.0);
	  }
      }

      runNumber = iEvent.id().run();
    }
}


double Gauss3DFunc(double* x, double* par)
{ 
  double a,b,c,d,e,f;

  a = cos(par[1])*cos(par[1]) / (2.*par[2]*par[2]) + sin(par[1])*sin(par[1]) / (2.*par[4]*par[4]) +
      cos(par[3])*cos(par[3]) / (2.*par[2]*par[2]) + sin(par[3])*sin(par[3]) / (2.*par[6]*par[6]);
  b = cos(par[1])*cos(par[1]) / (2.*par[4]*par[4]) + sin(par[1])*sin(par[1]) / (2.*par[2]*par[2]) +
      cos(par[5])*cos(par[5]) / (2.*par[4]*par[4]) + sin(par[5])*sin(par[5]) / (2.*par[6]*par[6]);
  c = cos(par[3])*cos(par[3]) / (2.*par[6]*par[6]) + sin(par[3])*sin(par[3]) / (2.*par[2]*par[2]) +
      cos(par[5])*cos(par[5]) / (2.*par[6]*par[6]) + sin(par[5])*sin(par[5]) / (2.*par[4]*par[4]);
  d = -sin(2.*par[1]) / (4.*par[2]*par[2]) + sin(2.*par[1]) / (4.*par[4]*par[4]);
  e = -sin(2.*par[5]) / (4.*par[4]*par[4]) + sin(2.*par[5]) / (4.*par[6]*par[6]);
  f = -sin(2.*par[3]) / (4.*par[6]*par[6]) + sin(2.*par[3]) / (4.*par[2]*par[2]);

  return par[0] + exp(-(a*(x[0]-par[7])*(x[0]-par[7]) + b*(x[1]-par[8])*(x[1]-par[8]) + c*(x[2]-par[9])*(x[2]-par[9]) +
			2.*d*(x[0]-par[7])*(x[1]-par[8]) + 2.*e*(x[1]-par[8])*(x[2]-par[9]) + 2.*f*(x[0]-par[7])*(x[2]-par[9])));
}


void Vx3DHLTAnalyzer::writeToFile(vector<double>* vals,
				  string fileName,
				  edm::TimeValue_t BeginTimeOfFit,
				  edm::TimeValue_t EndTimeOfFit,
				  unsigned int BeginLumiOfFit,
				  unsigned int EndLumiOfFit)
{
  ofstream outputFile;
  outputFile.open(fileName.c_str(), ios::app); // To append: ios::app, to overwrite ios::out

  stringstream BufferString;
  BufferString.precision(5);
  
  if ((outputFile.is_open() == true) && (vals != NULL))
    {
      vector<double>::const_iterator it = vals->begin();

      outputFile << "Runnumber " << runNumber << endl;
      outputFile << "BeginTimeOfFit " << beginTimeOfFit << endl;
      outputFile << "EndTimeOfFit " << endTimeOfFit << endl;
      outputFile << "LumiRange " << beginLumiOfFit << " - " << endLumiOfFit << endl;
      outputFile << "Type 3" << endl; // 3D Vertexing with Pixel Tracks = type 3

      BufferString << *(it+0);
      outputFile << "X0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+1);
      outputFile << "Y0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+2);
      outputFile << "Z0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+5);
      outputFile << "sigmaZ0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+6);
      outputFile << "dxdz " << BufferString.str().c_str() << endl; // @@@@@@ To be completed @@@@@@
      BufferString.str("");

      BufferString << *(it+7);
      outputFile << "dydz " << BufferString.str().c_str() << endl; // @@@@@@ To be completed @@@@@@
      BufferString.str("");

      BufferString << *(it+3);
      outputFile << "BeamWidthX " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+4);
      outputFile << "BeamWidthY " << BufferString.str().c_str() << endl;
      BufferString.str("");

      outputFile << "Cov(0,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(1,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(2,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(3,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(4,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(5,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(6,j) 0.0 0.0 0.0 0.0 0.0 0.0 0.0" << endl;

      outputFile << "EmittanceX 0.0" << endl;
      outputFile << "EmittanceY 0.0" << endl;
      outputFile << "BetaStar 0.0" << endl;

      outputFile.close();
    }
}


void Vx3DHLTAnalyzer::beginLuminosityBlock(const LuminosityBlock& lumiBlock, 
					   const EventSetup& iSetup)
{
  if (lumiCounter == 0)
    {
      beginTimeOfFit = lumiBlock.beginTime().value();
      beginLumiOfFit = lumiBlock.luminosityBlock();
    }
}


void Vx3DHLTAnalyzer::endLuminosityBlock(const LuminosityBlock& lumiBlock,
					 const EventSetup& iSetup)
{
  // To know the lumisection: lumiBlock.luminosityBlock()
  lumiCounter++;

  if ((lumiCounter == nLumiReset) && (nLumiReset != 0) && (beginTimeOfFit != 0))
    {
      TF1* Gauss   = new TF1("Gauss", "gaus");
      TF3* Gauss3D = new TF3("Gauss3D", Gauss3DFunc, -xRange/2., xRange/2., -yRange/2., yRange/2., -zRange/2., zRange/2., 10);

      mXlumi->ShiftFillLast(Vx_X->getTH1F()->GetMean(), Vx_X->getTH1F()->GetMeanError(), nLumiReset);
      mYlumi->ShiftFillLast(Vx_Y->getTH1F()->GetMean(), Vx_Y->getTH1F()->GetMeanError(), nLumiReset);
      mZlumi->ShiftFillLast(Vx_Z->getTH1F()->GetMean(), Vx_Z->getTH1F()->GetMeanError(), nLumiReset);
      
      sXlumi->ShiftFillLast(Vx_X->getTH1F()->GetRMS(), Vx_X->getTH1F()->GetRMSError(), nLumiReset);
      sYlumi->ShiftFillLast(Vx_Y->getTH1F()->GetRMS(), Vx_Y->getTH1F()->GetRMSError(), nLumiReset);
      sZlumi->ShiftFillLast(Vx_Z->getTH1F()->GetRMS(), Vx_Z->getTH1F()->GetRMSError(), nLumiReset);
      
      Gauss->SetParameters(Vx_X->getTH1()->GetMaximum(), Vx_X->getTH1()->GetMean(), Vx_X->getTH1()->GetRMS());
      Vx_X->getTH1()->Fit("Gauss","QN0");
      RMSsigXlumi->ShiftFillLast(Vx_X->getTH1F()->GetRMS() / Gauss->GetParameter(2),
				 (Vx_X->getTH1F()->GetRMS() / Gauss->GetParameter(2)) * sqrt(powf(Vx_X->getTH1F()->GetRMSError() / Vx_X->getTH1F()->GetRMS(),2.) +
											     powf(Gauss->GetParError(2) / Gauss->GetParError(2),2.)), nLumiReset);
      Gauss->SetParameters(Vx_Y->getTH1()->GetMaximum(), Vx_Y->getTH1()->GetMean(), Vx_Y->getTH1()->GetRMS());
      Vx_Y->getTH1()->Fit("Gauss","QN0");
      RMSsigYlumi->ShiftFillLast(Vx_Y->getTH1F()->GetRMS() / Gauss->GetParameter(2),
				 (Vx_Y->getTH1F()->GetRMS() / Gauss->GetParameter(2)) * sqrt(powf(Vx_Y->getTH1F()->GetRMSError() / Vx_Y->getTH1F()->GetRMS(),2.) +
											     powf(Gauss->GetParError(2) / Gauss->GetParError(2),2.)), nLumiReset);
      Gauss->SetParameters(Vx_Z->getTH1()->GetMaximum(), Vx_Z->getTH1()->GetMean(), Vx_Z->getTH1()->GetRMS());
      Vx_Z->getTH1()->Fit("Gauss","QN0");
      RMSsigXlumi->ShiftFillLast(Vx_Z->getTH1F()->GetRMS() / Gauss->GetParameter(2),
				 (Vx_Z->getTH1F()->GetRMS() / Gauss->GetParameter(2)) * sqrt(powf(Vx_Z->getTH1F()->GetRMSError() / Vx_Z->getTH1F()->GetRMS(),2.) +
											     powf(Gauss->GetParError(2) / Gauss->GetParError(2),2.)), nLumiReset);

      // Write the 8 beam spot parameters into a file
      endTimeOfFit = lumiBlock.endTime().value();
      endLumiOfFit = lumiBlock.luminosityBlock();
      vector<double> vals;
      Gauss->SetParameters(Distribution3D->getTH1()->GetMaximum(),
			   0.0,
			   Distribution3D->getTH1()->GetRMS(1),
			   0.0,
			   Distribution3D->getTH1()->GetRMS(2),
			   0.0,
			   Distribution3D->getTH1()->GetRMS(3),
			   Distribution3D->getTH1()->GetMean(1),
			   Distribution3D->getTH1()->GetMean(2),
			   Distribution3D->getTH1()->GetMean(3));
      Distribution3D->getTH1()->Fit("Gauss3D","QN0");
      if (dataFromFit == true)
	{
	  vals.push_back(Gauss3D->GetParameter(7));
	  vals.push_back(Gauss3D->GetParameter(8));
	  vals.push_back(Gauss3D->GetParameter(9));
	  vals.push_back(Gauss3D->GetParameter(2));
	  vals.push_back(Gauss3D->GetParameter(4));
	  vals.push_back(Gauss3D->GetParameter(6));
	  vals.push_back(Gauss3D->GetParameter(3));
	  vals.push_back(Gauss3D->GetParameter(5));

	}
      else
	{
	  vals.push_back(Vx_X->getTH1F()->GetMean());
	  vals.push_back(Vx_Y->getTH1F()->GetMean());
	  vals.push_back(Vx_Z->getTH1F()->GetMean());
	  vals.push_back(Vx_X->getTH1F()->GetRMS());
	  vals.push_back(Vx_Y->getTH1F()->GetRMS());
	  vals.push_back(Vx_Z->getTH1F()->GetRMS());
	  vals.push_back(0.);
	  vals.push_back(0.);
	}
      writeToFile(&vals, fileName, beginTimeOfFit, endTimeOfFit, beginLumiOfFit, endLumiOfFit);
      vals.clear();

      Vx_X->Reset();
      Vx_Y->Reset();
      Vx_Z->Reset();

      Vx_ZX->Reset();
      Vx_ZY->Reset();
      Vx_XY->Reset();

      Vx_ZX_profile->Reset();
      Vx_ZY_profile->Reset();

      reportSummary->Fill(0.5);
      reportSummaryMap->Fill(0.5, 0.5, 0.5);

      lumiCounter = 0;

      delete Gauss;
      delete Gauss3D;
    }
  else if (nLumiReset == 0)
    {
      reportSummary->Fill(0.5);
      reportSummaryMap->Fill(0.5, 0.5, 0.5);

      lumiCounter = 0;
    }
}


void Vx3DHLTAnalyzer::beginJob()
{
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();
 
  if ( dbe ) 
    {
      dbe->setCurrentFolder("BeamPixel");
      Distribution3D = dbe->book3D("Distribution_3D", "Distribution3D of the Vertex Coordinates",
				   ((int)(xRange/xStep) < 40) ? (int)(xRange/xStep) : 40, -xRange/2., xRange/2.,
				   ((int)(yRange/yStep) < 40) ? (int)(yRange/yStep) : 40, -yRange/2., yRange/2.,
				   ((int)(zRange/zStep) <  8) ? (int)(zRange/zStep) :  8, -zRange/2., zRange/2.);

      Vx_X = dbe->book1D("Vertex_X", "Primary Vertex X Coordinate Distribution", (int)(xRange/xStep), -xRange/2., xRange/2.);
      Vx_Y = dbe->book1D("Vertex_Y", "Primary Vertex Y Coordinate Distribution", (int)(yRange/yStep), -yRange/2., yRange/2.);
      Vx_Z = dbe->book1D("Vertex_Z", "Primary Vertex Z Coordinate Distribution", (int)(zRange/zStep), -zRange/2., zRange/2.);

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

      RMSsigXlumi = dbe->book1D("RMS_over_sigma X vs lumi", "RMS_{x}/\\sigma_{x} vs. Lumisection", 50, 0.5, 50.5);
      RMSsigYlumi = dbe->book1D("RMS_over_sigma Y vs lumi", "RMS_{y}/\\sigma_{y} vs. Lumisection", 50, 0.5, 50.5);
      RMSsigZlumi = dbe->book1D("RMS_over_sigma Z vs lumi", "RMS_{z}/\\sigma_{z} vs. Lumisection", 50, 0.5, 50.5);

      RMSsigXlumi->setAxisTitle("Lumisection [#]",1);
      RMSsigXlumi->setAxisTitle("Entries [#]",2);
      RMSsigXlumi->getTH1()->SetOption("E1");
      RMSsigYlumi->setAxisTitle("Lumisection [#]",1);
      RMSsigYlumi->setAxisTitle("Entries [#]",2);
      RMSsigYlumi->getTH1()->SetOption("E1");
      RMSsigZlumi->setAxisTitle("Lumisection [#]",1);
      RMSsigZlumi->setAxisTitle("Entries [#]",2);
      RMSsigZlumi->getTH1()->SetOption("E1");

      Vx_ZX = dbe->book2D("Vertex_ZX", "Primary Vertex ZX Coordinate Distributions", (int)(zRange/zStep/10.), -zRange/2., zRange/2., (int)(xRange/xStep/10.), -xRange/2., xRange/2.);
      Vx_ZY = dbe->book2D("Vertex_ZY", "Primary Vertex ZY Coordinate Distributions", (int)(zRange/zStep/10.), -zRange/2., zRange/2., (int)(yRange/yStep/10.), -yRange/2., yRange/2.);
      Vx_XY = dbe->book2D("Vertex_XY", "Primary Vertex XY Coordinate Distributions", (int)(xRange/xStep/10.), -xRange/2., xRange/2., (int)(yRange/yStep/10.), -yRange/2., yRange/2.);

      Vx_ZX->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZX->setAxisTitle("Primary Vertices X [cm]",2);
      Vx_ZX->setAxisTitle("Entries [#]",3);
      Vx_ZY->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_ZY->setAxisTitle("Entries [#]",3);
      Vx_XY->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_XY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_XY->setAxisTitle("Entries [#]",3);

      Vx_ZX_profile = dbe->bookProfile("ZX profile","ZX Profile", (int)(zRange/zStep/20.), -zRange/2., zRange/2., (int)(xRange/xStep/20.), -xRange/2., xRange/2., "");
      Vx_ZX_profile->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZX_profile->setAxisTitle("Primary Vertices X [cm]",2);

      Vx_ZY_profile = dbe->bookProfile("ZY profile","ZY Profile", (int)(zRange/zStep/20.), -zRange/2., zRange/2., (int)(yRange/yStep/20.), -yRange/2., yRange/2., "");
      Vx_ZY_profile->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZY_profile->setAxisTitle("Primary Vertices Y [cm]",2);

      dbe->setCurrentFolder("BeamPixel/EventInfo");
      reportSummary = dbe->bookFloat("reportSummary");
      reportSummary->Fill(0.);
      reportSummaryMap = dbe->book2D("reportSummaryMap","Beam Pixel Summary Map", 1, 0., 1., 1, 0., 1.);
      reportSummaryMap->Fill(0.5, 0.5, 0.);
      dbe->setCurrentFolder("BeamPixel/EventInfo/reportSummaryContents");
    }

  runNumber = 0;
  lumiCounter = 0;
  beginTimeOfFit = 0;
  endTimeOfFit = 0;
  beginLumiOfFit = 0;
  endLumiOfFit = 0;
}


void Vx3DHLTAnalyzer::endJob()
{
}


// Define this as a plug-in
DEFINE_FWK_MODULE(Vx3DHLTAnalyzer);
