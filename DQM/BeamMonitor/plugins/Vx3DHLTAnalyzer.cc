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
// $Id: Vx3DHLTAnalyzer.cc,v 1.106 2012/12/07 10:03:22 eulisse Exp $


#include "DQM/BeamMonitor/plugins/Vx3DHLTAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TFitterMinuit.h>


using namespace std;
using namespace reco;
using namespace edm;


Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const ParameterSet& iConfig)
{
  vertexCollection = edm::InputTag("pixelVertices");
  debugMode        = true;
  nLumiReset       = 1;
  dataFromFit      = true;
  minNentries      = 35;
  xRange           = 2.;
  xStep            = 0.001;
  yRange           = 2.;
  yStep            = 0.001;
  zRange           = 30.;
  zStep            = 0.05;
  VxErrCorr        = 1.5;
  fileName         = "BeamPixelResults.txt";

  vertexCollection = iConfig.getParameter<InputTag>("vertexCollection");
  debugMode        = iConfig.getParameter<bool>("debugMode");
  nLumiReset       = iConfig.getParameter<unsigned int>("nLumiReset");
  dataFromFit      = iConfig.getParameter<bool>("dataFromFit");
  minNentries      = iConfig.getParameter<unsigned int>("minNentries");
  xRange           = iConfig.getParameter<double>("xRange");
  xStep            = iConfig.getParameter<double>("xStep");
  yRange           = iConfig.getParameter<double>("yRange");
  yStep            = iConfig.getParameter<double>("yStep");
  zRange           = iConfig.getParameter<double>("zRange");
  zStep            = iConfig.getParameter<double>("zStep");
  VxErrCorr        = iConfig.getParameter<double>("VxErrCorr");
  fileName         = iConfig.getParameter<string>("fileName");
}


Vx3DHLTAnalyzer::~Vx3DHLTAnalyzer()
{
}


void Vx3DHLTAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  Handle<VertexCollection> Vx3DCollection;
  iEvent.getByLabel(vertexCollection,Vx3DCollection);

  unsigned int i,j;
  double det;
  VertexType MyVertex;

  if (runNumber != iEvent.id().run())
    {
      reset("scratch");
      runNumber = iEvent.id().run();

      if (debugMode == true)
	{
	  stringstream debugFile;
	  string tmp(fileName);

	  if (outputDebugFile.is_open() == true) outputDebugFile.close();
	  tmp.erase(strlen(fileName.c_str())-4,4);
	  debugFile << tmp.c_str() << "_Run" << iEvent.id().run() << ".txt";
	  outputDebugFile.open(debugFile.str().c_str(), ios::out);
	  outputDebugFile.close();
	  outputDebugFile.open(debugFile.str().c_str(), ios::app);
	}

      beginLuminosityBlock(iEvent.getLuminosityBlock(),iSetup);
    }
  else if (beginTimeOfFit != 0)
    {
      totalHits += HitCounter(iEvent);

      for (vector<Vertex>::const_iterator it3DVx = Vx3DCollection->begin(); it3DVx != Vx3DCollection->end(); it3DVx++)
	{
	  if ((it3DVx->isValid() == true) &&
	      (it3DVx->isFake() == false) &&
	      (it3DVx->ndof() >= minVxDoF))
	    {
	      for (i = 0; i < DIM; i++)
		{
		  for (j = 0; j < DIM; j++)
		    {
		      MyVertex.Covariance[i][j] = it3DVx->covariance(i,j);
		      if (edm::isNotFinite(MyVertex.Covariance[i][j]) == true) break;
		    }
		  if (j != DIM) break;
		}
	      det = std::fabs(MyVertex.Covariance[0][0])*(std::fabs(MyVertex.Covariance[1][1])*std::fabs(MyVertex.Covariance[2][2]) - MyVertex.Covariance[1][2]*MyVertex.Covariance[1][2]) -
		MyVertex.Covariance[0][1]*(MyVertex.Covariance[0][1]*std::fabs(MyVertex.Covariance[2][2]) - MyVertex.Covariance[0][2]*MyVertex.Covariance[1][2]) +
		MyVertex.Covariance[0][2]*(MyVertex.Covariance[0][1]*MyVertex.Covariance[1][2] - MyVertex.Covariance[0][2]*std::fabs(MyVertex.Covariance[1][1]));
	      if ((i == DIM) && (det > 0.))
		{
		  MyVertex.x = it3DVx->x();
		  MyVertex.y = it3DVx->y();
		  MyVertex.z = it3DVx->z();
		  Vertices.push_back(MyVertex);
		}
	      else if (internalDebug == true)
		{
		  cout << "Vertex discarded !" << endl;
		  for (i = 0; i < DIM; i++)
		    for (j = 0; j < DIM; j++)
		      cout << "(i,j) --> " << i << "," << j << " --> " << MyVertex.Covariance[i][j] << endl;
		}
	      
	      Vx_X->Fill(it3DVx->x());
	      Vx_Y->Fill(it3DVx->y());
	      Vx_Z->Fill(it3DVx->z());
	      
	      Vx_ZX->Fill(it3DVx->z(), it3DVx->x());
	      Vx_ZY->Fill(it3DVx->z(), it3DVx->y());
	      Vx_XY->Fill(it3DVx->x(), it3DVx->y());
	    }
	}
    }
}


unsigned int Vx3DHLTAnalyzer::HitCounter(const Event& iEvent)
{
  edm::Handle<SiPixelRecHitCollection> rechitspixel;
  iEvent.getByLabel("siPixelRecHits",rechitspixel);

  unsigned int counter = 0;
  
  for (SiPixelRecHitCollection::const_iterator j = rechitspixel->begin(); j != rechitspixel->end(); j++)
    for (edmNew::DetSet<SiPixelRecHit>::const_iterator h = j->begin(); h != j->end(); h++) counter += h->cluster()->size();	  
  
  return counter;
}


char* Vx3DHLTAnalyzer::formatTime (const time_t& t)
{
  static char ts[25];
  strftime(ts, sizeof(ts), "%Y.%m.%d %H:%M:%S %Z", gmtime(&t));

  return ts;
}


void Gauss3DFunc(int& /*npar*/, double* /*gin*/, double& fval, double* par, int /*iflag*/)
{
  double K[DIM][DIM]; // Covariance Matrix
  double M[DIM][DIM]; // K^-1
  double det;
  double sumlog = 0.;

//   par[0] = K(0,0) --> Var[X]
//   par[1] = K(1,1) --> Var[Y]
//   par[2] = K(2,2) --> Var[Z]
//   par[3] = K(0,1) = K(1,0) --> Cov[X,Y]
//   par[4] = K(1,2) = K(2,1) --> Cov[Y,Z] --> dy/dz
//   par[5] = K(0,2) = K(2,0) --> Cov[X,Z] --> dx/dz
//   par[6] = mean x
//   par[7] = mean y
//   par[8] = mean z

  counterVx = 0;
  for (unsigned int i = 0; i < Vertices.size(); i++)
    {
      if ((std::sqrt((Vertices[i].x-xPos)*(Vertices[i].x-xPos) + (Vertices[i].y-yPos)*(Vertices[i].y-yPos)) <= maxTransRadius) &&
	  (std::fabs(Vertices[i].z-zPos) <= maxLongLength))
	{
	  if (considerVxCovariance == true)
	    {
	      K[0][0] = std::fabs(par[0]) + VxErrCorr*VxErrCorr * std::fabs(Vertices[i].Covariance[0][0]);
	      K[1][1] = std::fabs(par[1]) + VxErrCorr*VxErrCorr * std::fabs(Vertices[i].Covariance[1][1]);
	      K[2][2] = std::fabs(par[2]) + VxErrCorr*VxErrCorr * std::fabs(Vertices[i].Covariance[2][2]);
	      K[0][1] = K[1][0] = par[3] + VxErrCorr*VxErrCorr * Vertices[i].Covariance[0][1];
	      K[1][2] = K[2][1] = par[4]*(std::fabs(par[2])-std::fabs(par[1])) - par[5]*par[3] + VxErrCorr*VxErrCorr * Vertices[i].Covariance[1][2];
	      K[0][2] = K[2][0] = par[5]*(std::fabs(par[2])-std::fabs(par[0])) - par[4]*par[3] + VxErrCorr*VxErrCorr * Vertices[i].Covariance[0][2];
	    }
	  else
	    {
	      K[0][0] = std::fabs(par[0]);
	      K[1][1] = std::fabs(par[1]);
	      K[2][2] = std::fabs(par[2]);
	      K[0][1] = K[1][0] = par[3];
	      K[1][2] = K[2][1] = par[4]*(std::fabs(par[2])-std::fabs(par[1])) - par[5]*par[3];
	      K[0][2] = K[2][0] = par[5]*(std::fabs(par[2])-std::fabs(par[0])) - par[4]*par[3];
	    }

	  det = K[0][0]*(K[1][1]*K[2][2] - K[1][2]*K[1][2]) -
		K[0][1]*(K[0][1]*K[2][2] - K[0][2]*K[1][2]) +
		K[0][2]*(K[0][1]*K[1][2] - K[0][2]*K[1][1]);

	  M[0][0] = (K[1][1]*K[2][2] - K[1][2]*K[1][2]) / det;
	  M[1][1] = (K[0][0]*K[2][2] - K[0][2]*K[0][2]) / det;
	  M[2][2] = (K[0][0]*K[1][1] - K[0][1]*K[0][1]) / det;
	  M[0][1] = M[1][0] = (K[0][2]*K[1][2] - K[0][1]*K[2][2]) / det;
	  M[1][2] = M[2][1] = (K[0][2]*K[0][1] - K[1][2]*K[0][0]) / det;
	  M[0][2] = M[2][0] = (K[0][1]*K[1][2] - K[0][2]*K[1][1]) / det;
	  
	  sumlog += double(DIM)*std::log(2.*pi) + std::log(std::fabs(det)) +
	    (M[0][0]*(Vertices[i].x-par[6])*(Vertices[i].x-par[6]) +
	     M[1][1]*(Vertices[i].y-par[7])*(Vertices[i].y-par[7]) +
	     M[2][2]*(Vertices[i].z-par[8])*(Vertices[i].z-par[8]) +
	     2.*M[0][1]*(Vertices[i].x-par[6])*(Vertices[i].y-par[7]) +
	     2.*M[1][2]*(Vertices[i].y-par[7])*(Vertices[i].z-par[8]) +
	     2.*M[0][2]*(Vertices[i].x-par[6])*(Vertices[i].z-par[8]));
	  
	  counterVx++;
	}
    }
  
  fval = sumlog;
}


int Vx3DHLTAnalyzer::MyFit(vector<double>* vals)
{
  // RETURN CODE:
  //  0 == OK
  // -2 == NO OK - not enough "minNentries"
  // Any other number == NO OK
  unsigned int nParams = 9;
 
  if ((vals != NULL) && (vals->size() == nParams*2))
    {
      double nSigmaXY       = 100.;
      double nSigmaZ        = 100.;
      double varFactor      = 4./25.; // Take into account the difference between the RMS and sigma (RMS usually greater than sigma)
      double parDistanceXY  = 0.005;  // Unit: [cm]
      double parDistanceZ   = 0.5;    // Unit: [cm]
      double parDistanceddZ = 1e-3;   // Unit: [rad]
      double parDistanceCxy = 1e-5;   // Unit: [cm^2]
      double bestEdm        = 1e-1;

      const unsigned int trials = 4;
      double largerDist[trials] = {0.1, 5., 10., 100.};

      double covxz,covyz,det;
      double deltaMean;
      int bestMovementX = 1;
      int bestMovementY = 1;
      int bestMovementZ = 1;
      int goodData;

      double arglist[2];
      double amin,errdef,edm;
      int nvpar,nparx;
      
      vector<double>::const_iterator it = vals->begin();

      TFitterMinuit* Gauss3D = new TFitterMinuit(nParams);
      if (internalDebug == true) Gauss3D->SetPrintLevel(3);
      else Gauss3D->SetPrintLevel(0);
      Gauss3D->SetFCN(Gauss3DFunc);
      arglist[0] = 10000; // Max number of function calls
      arglist[1] = 1e-9;  // Tolerance on likelihood

      if (internalDebug == true) cout << "\n@@@ START FITTING @@@" << endl;

      // @@@ Fit at X-deltaMean | X | X+deltaMean @@@
      bestEdm = 1.;
      for (int i = 0; i < 3; i++)
	{
	  deltaMean = (double(i)-1.)*std::sqrt((*(it+0))*varFactor);
	  if (internalDebug == true) cout << "deltaMean --> " << deltaMean << endl;

	  Gauss3D->Clear();

	  // arg3 - first guess of parameter value
	  // arg4 - step of the parameter
	  Gauss3D->SetParameter(0,"var x ", *(it+0)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(1,"var y ", *(it+1)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(2,"var z ", *(it+2), parDistanceZ*parDistanceZ, 0., 0.);
	  Gauss3D->SetParameter(3,"cov xy", *(it+3), parDistanceCxy, 0., 0.);
	  Gauss3D->SetParameter(4,"dydz  ", *(it+4), parDistanceddZ, 0., 0.);
	  Gauss3D->SetParameter(5,"dxdz  ", *(it+5), parDistanceddZ, 0., 0.);
	  Gauss3D->SetParameter(6,"mean x", *(it+6)+deltaMean, parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(7,"mean y", *(it+7), parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(8,"mean z", *(it+8), parDistanceZ, 0., 0.);

	  // Set the central positions of the centroid for vertex rejection
	  xPos = Gauss3D->GetParameter(6);
	  yPos = Gauss3D->GetParameter(7);
	  zPos = Gauss3D->GetParameter(8);

	  // Set dimensions of the centroid for vertex rejection
	  maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->GetParameter(0)) + std::fabs(Gauss3D->GetParameter(1))) / 2.;
	  maxLongLength  = nSigmaZ  * std::sqrt(std::fabs(Gauss3D->GetParameter(2)));

	  goodData = Gauss3D->ExecuteCommand("MIGRAD",arglist,2);
	  Gauss3D->GetStats(amin, edm, errdef, nvpar, nparx);

	  if (counterVx < minNentries) goodData = -2;
	  else if (edm::isNotFinite(edm) == true) goodData = -1;
	  else for (unsigned int j = 0; j < nParams; j++) if (edm::isNotFinite(Gauss3D->GetParError(j)) == true) { goodData = -1; break; }
	  if (goodData == 0)
	    {
	      covyz = Gauss3D->GetParameter(4)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(1))) - Gauss3D->GetParameter(5)*Gauss3D->GetParameter(3);
	      covxz = Gauss3D->GetParameter(5)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(0))) - Gauss3D->GetParameter(4)*Gauss3D->GetParameter(3);
	      
	      det = std::fabs(Gauss3D->GetParameter(0)) * (std::fabs(Gauss3D->GetParameter(1))*std::fabs(Gauss3D->GetParameter(2)) - covyz*covyz) -
		Gauss3D->GetParameter(3) * (Gauss3D->GetParameter(3)*std::fabs(Gauss3D->GetParameter(2)) - covxz*covyz) +
		covxz * (Gauss3D->GetParameter(3)*covyz - covxz*std::fabs(Gauss3D->GetParameter(1)));
	      if (det < 0.) { goodData = -1; if (internalDebug == true) cout << "Negative determinant !" << endl; }
	    }

	  if ((goodData == 0) && (std::fabs(edm) < bestEdm)) { bestEdm = edm; bestMovementX = i; }
	}
      if (internalDebug == true) cout << "Found bestMovementX --> " << bestMovementX << endl;

      // @@@ Fit at Y-deltaMean | Y | Y+deltaMean @@@
      bestEdm = 1.;
      for (int i = 0; i < 3; i++)
	{
	  deltaMean = (double(i)-1.)*std::sqrt((*(it+1))*varFactor);
	  if (internalDebug == true)
	    {
	      cout << "deltaMean --> " << deltaMean << endl;
	      cout << "deltaMean X --> " << (double(bestMovementX)-1.)*std::sqrt((*(it+0))*varFactor) << endl;
	    }

	  Gauss3D->Clear();

	  // arg3 - first guess of parameter value
	  // arg4 - step of the parameter
	  Gauss3D->SetParameter(0,"var x ", *(it+0)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(1,"var y ", *(it+1)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(2,"var z ", *(it+2), parDistanceZ*parDistanceZ, 0., 0.);
	  Gauss3D->SetParameter(3,"cov xy", *(it+3), parDistanceCxy, 0., 0.);
	  Gauss3D->SetParameter(4,"dydz  ", *(it+4), parDistanceddZ, 0., 0.);
	  Gauss3D->SetParameter(5,"dxdz  ", *(it+5), parDistanceddZ, 0., 0.);
	  Gauss3D->SetParameter(6,"mean x", *(it+6)+(double(bestMovementX)-1.)*std::sqrt((*(it+0))*varFactor), parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(7,"mean y", *(it+7)+deltaMean, parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(8,"mean z", *(it+8), parDistanceZ, 0., 0.);

	  // Set the central positions of the centroid for vertex rejection
	  xPos = Gauss3D->GetParameter(6);
	  yPos = Gauss3D->GetParameter(7);
	  zPos = Gauss3D->GetParameter(8);

	  // Set dimensions of the centroid for vertex rejection
	  maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->GetParameter(0)) + std::fabs(Gauss3D->GetParameter(1))) / 2.;
	  maxLongLength  = nSigmaZ  * std::sqrt(std::fabs(Gauss3D->GetParameter(2)));

	  goodData = Gauss3D->ExecuteCommand("MIGRAD",arglist,2);
	  Gauss3D->GetStats(amin, edm, errdef, nvpar, nparx);

	  if (counterVx < minNentries) goodData = -2;
	  else if (edm::isNotFinite(edm) == true) goodData = -1;
	  else for (unsigned int j = 0; j < nParams; j++) if (edm::isNotFinite(Gauss3D->GetParError(j)) == true) { goodData = -1; break; }
	  if (goodData == 0)
	    {
	      covyz = Gauss3D->GetParameter(4)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(1))) - Gauss3D->GetParameter(5)*Gauss3D->GetParameter(3);
	      covxz = Gauss3D->GetParameter(5)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(0))) - Gauss3D->GetParameter(4)*Gauss3D->GetParameter(3);
	      
	      det = std::fabs(Gauss3D->GetParameter(0)) * (std::fabs(Gauss3D->GetParameter(1))*std::fabs(Gauss3D->GetParameter(2)) - covyz*covyz) -
		Gauss3D->GetParameter(3) * (Gauss3D->GetParameter(3)*std::fabs(Gauss3D->GetParameter(2)) - covxz*covyz) +
		covxz * (Gauss3D->GetParameter(3)*covyz - covxz*std::fabs(Gauss3D->GetParameter(1)));
	      if (det < 0.) { goodData = -1; if (internalDebug == true) cout << "Negative determinant !" << endl; }
	    }
	  
	  if ((goodData == 0) && (std::fabs(edm) < bestEdm)) { bestEdm = edm; bestMovementY = i; }
	}
      if (internalDebug == true) cout << "Found bestMovementY --> " << bestMovementY << endl;

      // @@@ Fit at Z-deltaMean | Z | Z+deltaMean @@@
      bestEdm = 1.;
      for (int i = 0; i < 3; i++)
	{
	  deltaMean = (double(i)-1.)*std::sqrt(*(it+2));
	  if (internalDebug == true)
	    {
	      cout << "deltaMean --> " << deltaMean << endl;
	      cout << "deltaMean X --> " << (double(bestMovementX)-1.)*std::sqrt((*(it+0))*varFactor) << endl;
	      cout << "deltaMean Y --> " << (double(bestMovementY)-1.)*std::sqrt((*(it+1))*varFactor) << endl;
	    }

	  Gauss3D->Clear();

	  // arg3 - first guess of parameter value
	  // arg4 - step of the parameter
	  Gauss3D->SetParameter(0,"var x ", *(it+0)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(1,"var y ", *(it+1)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(2,"var z ", *(it+2), parDistanceZ*parDistanceZ, 0., 0.);
	  Gauss3D->SetParameter(3,"cov xy", *(it+3), parDistanceCxy, 0., 0.);
	  Gauss3D->SetParameter(4,"dydz  ", *(it+4), parDistanceddZ, 0., 0.);
	  Gauss3D->SetParameter(5,"dxdz  ", *(it+5), parDistanceddZ, 0., 0.);
	  Gauss3D->SetParameter(6,"mean x", *(it+6)+(double(bestMovementX)-1.)*std::sqrt((*(it+0))*varFactor), parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(7,"mean y", *(it+7)+(double(bestMovementY)-1.)*std::sqrt((*(it+1))*varFactor), parDistanceXY, 0., 0.);
	  Gauss3D->SetParameter(8,"mean z", *(it+8)+deltaMean, parDistanceZ, 0., 0.);

	  // Set the central positions of the centroid for vertex rejection
	  xPos = Gauss3D->GetParameter(6);
	  yPos = Gauss3D->GetParameter(7);
	  zPos = Gauss3D->GetParameter(8);

	  // Set dimensions of the centroid for vertex rejection
	  maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->GetParameter(0)) + std::fabs(Gauss3D->GetParameter(1))) / 2.;
	  maxLongLength  = nSigmaZ  * std::sqrt(std::fabs(Gauss3D->GetParameter(2)));

	  goodData = Gauss3D->ExecuteCommand("MIGRAD",arglist,2);
	  Gauss3D->GetStats(amin, edm, errdef, nvpar, nparx);

	  if (counterVx < minNentries) goodData = -2;
	  else if (edm::isNotFinite(edm) == true) goodData = -1;
	  else for (unsigned int j = 0; j < nParams; j++) if (edm::isNotFinite(Gauss3D->GetParError(j)) == true) { goodData = -1; break; }
	  if (goodData == 0)
	    {
	      covyz = Gauss3D->GetParameter(4)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(1))) - Gauss3D->GetParameter(5)*Gauss3D->GetParameter(3);
	      covxz = Gauss3D->GetParameter(5)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(0))) - Gauss3D->GetParameter(4)*Gauss3D->GetParameter(3);
	      
	      det = std::fabs(Gauss3D->GetParameter(0)) * (std::fabs(Gauss3D->GetParameter(1))*std::fabs(Gauss3D->GetParameter(2)) - covyz*covyz) -
		Gauss3D->GetParameter(3) * (Gauss3D->GetParameter(3)*std::fabs(Gauss3D->GetParameter(2)) - covxz*covyz) +
		covxz * (Gauss3D->GetParameter(3)*covyz - covxz*std::fabs(Gauss3D->GetParameter(1)));
	      if (det < 0.) { goodData = -1; if (internalDebug == true) cout << "Negative determinant !" << endl; }
	    }
	  
	  if ((goodData == 0) && (std::fabs(edm) < bestEdm)) { bestEdm = edm; bestMovementZ = i; }
	}
      if (internalDebug == true) cout << "Found bestMovementZ --> " << bestMovementZ << endl;

      Gauss3D->Clear();

      // @@@ FINAL FIT @@@
      // arg3 - first guess of parameter value
      // arg4 - step of the parameter
      Gauss3D->SetParameter(0,"var x ", *(it+0)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
      Gauss3D->SetParameter(1,"var y ", *(it+1)*varFactor, parDistanceXY*parDistanceXY, 0., 0.);
      Gauss3D->SetParameter(2,"var z ", *(it+2), parDistanceZ*parDistanceZ, 0., 0.);
      Gauss3D->SetParameter(3,"cov xy", *(it+3), parDistanceCxy, 0., 0.);
      Gauss3D->SetParameter(4,"dydz  ", *(it+4), parDistanceddZ, 0., 0.);
      Gauss3D->SetParameter(5,"dxdz  ", *(it+5), parDistanceddZ, 0., 0.);
      Gauss3D->SetParameter(6,"mean x", *(it+6)+(double(bestMovementX)-1.)*std::sqrt((*(it+0))*varFactor), parDistanceXY, 0., 0.);
      Gauss3D->SetParameter(7,"mean y", *(it+7)+(double(bestMovementY)-1.)*std::sqrt((*(it+1))*varFactor), parDistanceXY, 0., 0.);
      Gauss3D->SetParameter(8,"mean z", *(it+8)+(double(bestMovementZ)-1.)*std::sqrt(*(it+2)), parDistanceZ, 0., 0.);

      // Set the central positions of the centroid for vertex rejection
      xPos = Gauss3D->GetParameter(6);
      yPos = Gauss3D->GetParameter(7);
      zPos = Gauss3D->GetParameter(8);
      
      // Set dimensions of the centroid for vertex rejection
      maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->GetParameter(0)) + std::fabs(Gauss3D->GetParameter(1))) / 2.;
      maxLongLength  = nSigmaZ  * std::sqrt(std::fabs(Gauss3D->GetParameter(2)));

      goodData = Gauss3D->ExecuteCommand("MIGRAD",arglist,2);      
      Gauss3D->GetStats(amin, edm, errdef, nvpar, nparx);
      
      if (counterVx < minNentries) goodData = -2;
      else if (edm::isNotFinite(edm) == true) goodData = -1;
      else for (unsigned int j = 0; j < nParams; j++) if (edm::isNotFinite(Gauss3D->GetParError(j)) == true) { goodData = -1; break; }
      if (goodData == 0)
	{
	  covyz = Gauss3D->GetParameter(4)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(1))) - Gauss3D->GetParameter(5)*Gauss3D->GetParameter(3);
	  covxz = Gauss3D->GetParameter(5)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(0))) - Gauss3D->GetParameter(4)*Gauss3D->GetParameter(3);
	  
	  det = std::fabs(Gauss3D->GetParameter(0)) * (std::fabs(Gauss3D->GetParameter(1))*std::fabs(Gauss3D->GetParameter(2)) - covyz*covyz) -
	    Gauss3D->GetParameter(3) * (Gauss3D->GetParameter(3)*std::fabs(Gauss3D->GetParameter(2)) - covxz*covyz) +
	    covxz * (Gauss3D->GetParameter(3)*covyz - covxz*std::fabs(Gauss3D->GetParameter(1)));
	  if (det < 0.) { goodData = -1; if (internalDebug == true) cout << "Negative determinant !" << endl; }
	}

      // @@@ FIT WITH DIFFERENT PARAMETER DISTANCES@@@
      // arg3 - first guess of parameter value
      // arg4 - step of the parameter
      for (unsigned int i = 0; i < trials; i++)
	{
	  if ((goodData != 0) && (goodData != -2))
	    {
	      Gauss3D->Clear();
	  
	      if (internalDebug == true) cout << "FIT WITH DIFFERENT PARAMETER DISTANCES - STEP " << i+1 << endl;      

	      Gauss3D->SetParameter(0,"var x ", *(it+0)*varFactor, parDistanceXY*parDistanceXY * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(1,"var y ", *(it+1)*varFactor, parDistanceXY*parDistanceXY * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(2,"var z ", *(it+2), parDistanceZ*parDistanceZ * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(3,"cov xy", *(it+3), parDistanceCxy * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(4,"dydz  ", *(it+4), parDistanceddZ * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(5,"dxdz  ", *(it+5), parDistanceddZ * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(6,"mean x", *(it+6)+(double(bestMovementX)-1.)*std::sqrt((*(it+0))*varFactor), parDistanceXY * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(7,"mean y", *(it+7)+(double(bestMovementY)-1.)*std::sqrt((*(it+1))*varFactor), parDistanceXY * largerDist[i], 0, 0);
	      Gauss3D->SetParameter(8,"mean z", *(it+8)+(double(bestMovementZ)-1.)*std::sqrt(*(it+2)), parDistanceZ * largerDist[i], 0, 0);

	      // Set the central positions of the centroid for vertex rejection
	      xPos = Gauss3D->GetParameter(6);
	      yPos = Gauss3D->GetParameter(7);
	      zPos = Gauss3D->GetParameter(8);

	      // Set dimensions of the centroid for vertex rejection
	      maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->GetParameter(0)) + std::fabs(Gauss3D->GetParameter(1))) / 2.;
	      maxLongLength  = nSigmaZ  * std::sqrt(std::fabs(Gauss3D->GetParameter(2)));

	      goodData = Gauss3D->ExecuteCommand("MIGRAD",arglist,2);
	      Gauss3D->GetStats(amin, edm, errdef, nvpar, nparx);
      
	      if (counterVx < minNentries) goodData = -2;
	      else if (edm::isNotFinite(edm) == true) goodData = -1;
	      else for (unsigned int j = 0; j < nParams; j++) if (edm::isNotFinite(Gauss3D->GetParError(j)) == true) { goodData = -1; break; }
	      if (goodData == 0)
		{
		  covyz = Gauss3D->GetParameter(4)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(1))) - Gauss3D->GetParameter(5)*Gauss3D->GetParameter(3);
		  covxz = Gauss3D->GetParameter(5)*(std::fabs(Gauss3D->GetParameter(2))-std::fabs(Gauss3D->GetParameter(0))) - Gauss3D->GetParameter(4)*Gauss3D->GetParameter(3);
	      
		  det = std::fabs(Gauss3D->GetParameter(0)) * (std::fabs(Gauss3D->GetParameter(1))*std::fabs(Gauss3D->GetParameter(2)) - covyz*covyz) -
		    Gauss3D->GetParameter(3) * (Gauss3D->GetParameter(3)*std::fabs(Gauss3D->GetParameter(2)) - covxz*covyz) +
		    covxz * (Gauss3D->GetParameter(3)*covyz - covxz*std::fabs(Gauss3D->GetParameter(1)));
		  if (det < 0.) { goodData = -1; if (internalDebug == true) cout << "Negative determinant !" << endl; }
		}
	    } else break;
	}

      if (goodData == 0)
	for (unsigned int i = 0; i < nParams; i++)
	  {
	    vals->operator[](i) = Gauss3D->GetParameter(i);
	    vals->operator[](i+nParams) = Gauss3D->GetParError(i);
	  }
      
      delete Gauss3D;
      return goodData;
    }
  
  return -1;
}


void Vx3DHLTAnalyzer::reset(string ResetType)
{
  if (ResetType.compare("scratch") == 0)
    {
      runNumber        = 0;
      numberGoodFits   = 0;
      numberFits       = 0;
      lastLumiOfFit    = 0;
      
      Vx_X->Reset();
      Vx_Y->Reset();
      Vx_Z->Reset();
      
      Vx_ZX->Reset();
      Vx_ZY->Reset();
      Vx_XY->Reset();
      
      mXlumi->Reset();
      mYlumi->Reset();
      mZlumi->Reset();

      sXlumi->Reset();
      sYlumi->Reset();
      sZlumi->Reset();

      dxdzlumi->Reset();
      dydzlumi->Reset();

      hitCounter->Reset();
      hitCountHistory->Reset();
      goodVxCounter->Reset();
      goodVxCountHistory->Reset();
      fitResults->Reset();

      reportSummary->Fill(0.);
      reportSummaryMap->Fill(0.5, 0.5, 0.);

      Vertices.clear();
      
      lumiCounter      = 0;
      lumiCounterHisto = 0;
      totalHits        = 0;
      beginTimeOfFit   = 0;
      endTimeOfFit     = 0;
      beginLumiOfFit   = 0;
      endLumiOfFit     = 0;
    }
  else if (ResetType.compare("whole") == 0)
    {
      Vx_X->Reset();
      Vx_Y->Reset();
      Vx_Z->Reset();
      
      Vx_ZX->Reset();
      Vx_ZY->Reset();
      Vx_XY->Reset();
      
      Vertices.clear();
      
      lumiCounter      = 0;
      lumiCounterHisto = 0;
      totalHits        = 0;
      beginTimeOfFit   = 0;
      endTimeOfFit     = 0;
      beginLumiOfFit   = 0;
      endLumiOfFit     = 0;
    }
  else if (ResetType.compare("partial") == 0)
    {
      Vx_X->Reset();
      Vx_Y->Reset();
      Vx_Z->Reset();
      
      Vertices.clear();
      
      lumiCounter      = 0;
      totalHits        = 0;
      beginTimeOfFit   = 0;
      endTimeOfFit     = 0;
      beginLumiOfFit   = 0;
      endLumiOfFit     = 0;
    }
  else if (ResetType.compare("nohisto") == 0)
    {
      Vertices.clear();
      
      lumiCounter      = 0;
      lumiCounterHisto = 0;
      totalHits        = 0;
      beginTimeOfFit   = 0;
      endTimeOfFit     = 0;
      beginLumiOfFit   = 0;
      endLumiOfFit     = 0;
    }
  else if (ResetType.compare("hitCounter") == 0)
    totalHits          = 0;
}


void Vx3DHLTAnalyzer::writeToFile(vector<double>* vals,
				  edm::TimeValue_t BeginTimeOfFit,
				  edm::TimeValue_t EndTimeOfFit,
				  unsigned int BeginLumiOfFit,
				  unsigned int EndLumiOfFit,
				  int dataType)
{
  stringstream BufferString;
  BufferString.precision(5);

  outputFile.open(fileName.c_str(), ios::out);

  if ((outputFile.is_open() == true) && (vals != NULL) && (vals->size() == 8*2))
    {
      vector<double>::const_iterator it = vals->begin();

      outputFile << "Runnumber " << runNumber << endl;
      outputFile << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " " << (beginTimeOfFit >> 32) << endl;
      outputFile << "EndTimeOfFit " << formatTime(endTimeOfFit >> 32) << " " << (endTimeOfFit >> 32) << endl;
      outputFile << "LumiRange " << beginLumiOfFit << " - " << endLumiOfFit << endl;
      outputFile << "Type " << dataType << endl;
      // 3D Vertexing with Pixel Tracks:
      // Good data = Type  3
      // Bad data  = Type -1

      BufferString << *(it+0);
      outputFile << "X0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+1);
      outputFile << "Y0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+2);
      outputFile << "Z0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+3);
      outputFile << "sigmaZ0 " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+4);
      outputFile << "dxdz " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+5);
      outputFile << "dydz " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+6);
      outputFile << "BeamWidthX " << BufferString.str().c_str() << endl;
      BufferString.str("");

      BufferString << *(it+7);
      outputFile << "BeamWidthY " << BufferString.str().c_str() << endl;
      BufferString.str("");

      outputFile << "Cov(0,j) " << *(it+8) << " 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(1,j) 0.0 " << *(it+9) << " 0.0 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(2,j) 0.0 0.0 " << *(it+10) << " 0.0 0.0 0.0 0.0" << endl;
      outputFile << "Cov(3,j) 0.0 0.0 0.0 " << *(it+11) << " 0.0 0.0 0.0" << endl;
      outputFile << "Cov(4,j) 0.0 0.0 0.0 0.0 " << *(it+12) << " 0.0 0.0" << endl;
      outputFile << "Cov(5,j) 0.0 0.0 0.0 0.0 0.0 " << *(it+13) << " 0.0" << endl;
      outputFile << "Cov(6,j) 0.0 0.0 0.0 0.0 0.0 0.0 " << ((*(it+14)) + (*(it+15)) + 2.*std::sqrt((*(it+14))*(*(it+15)))) / 4. << endl;

      outputFile << "EmittanceX 0.0" << endl;
      outputFile << "EmittanceY 0.0" << endl;
      outputFile << "BetaStar 0.0" << endl;
    }
  outputFile.close();

  if ((debugMode == true) && (outputDebugFile.is_open() == true) && (vals != NULL) && (vals->size() == 8*2))
    {
      vector<double>::const_iterator it = vals->begin();

      outputDebugFile << "Runnumber " << runNumber << endl;
      outputDebugFile << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " " << (beginTimeOfFit >> 32) << endl;
      outputDebugFile << "EndTimeOfFit " << formatTime(endTimeOfFit >> 32) << " " << (endTimeOfFit >> 32) << endl;
      outputDebugFile << "LumiRange " << beginLumiOfFit << " - " << endLumiOfFit << endl;
      outputDebugFile << "Type " << dataType << endl;
      // 3D Vertexing with Pixel Tracks:
      // Good data = Type  3
      // Bad data  = Type -1
	  
      BufferString << *(it+0);
      outputDebugFile << "X0 " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+1);
      outputDebugFile << "Y0 " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+2);
      outputDebugFile << "Z0 " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+3);
      outputDebugFile << "sigmaZ0 " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+4);
      outputDebugFile << "dxdz " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+5);
      outputDebugFile << "dydz " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+6);
      outputDebugFile << "BeamWidthX " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      BufferString << *(it+7);
      outputDebugFile << "BeamWidthY " << BufferString.str().c_str() << endl;
      BufferString.str("");
	  
      outputDebugFile << "Cov(0,j) " << *(it+8) << " 0.0 0.0 0.0 0.0 0.0 0.0" << endl;
      outputDebugFile << "Cov(1,j) 0.0 " << *(it+9) << " 0.0 0.0 0.0 0.0 0.0" << endl;
      outputDebugFile << "Cov(2,j) 0.0 0.0 " << *(it+10) << " 0.0 0.0 0.0 0.0" << endl;
      outputDebugFile << "Cov(3,j) 0.0 0.0 0.0 " << *(it+11) << " 0.0 0.0 0.0" << endl;
      outputDebugFile << "Cov(4,j) 0.0 0.0 0.0 0.0 " << *(it+12) << " 0.0 0.0" << endl;
      outputDebugFile << "Cov(5,j) 0.0 0.0 0.0 0.0 0.0 " << *(it+13) << " 0.0" << endl;
      outputDebugFile << "Cov(6,j) 0.0 0.0 0.0 0.0 0.0 0.0 " << ((*(it+14)) + (*(it+15)) + 2.*std::sqrt((*(it+14))*(*(it+15)))) / 4. << endl;
	  
      outputDebugFile << "EmittanceX 0.0" << endl;
      outputDebugFile << "EmittanceY 0.0" << endl;
      outputDebugFile << "BetaStar 0.0" << endl;
    }
}


void Vx3DHLTAnalyzer::beginLuminosityBlock(const LuminosityBlock& lumiBlock, 
					   const EventSetup& iSetup)
{
  if ((lumiCounter == 0) && (lumiBlock.luminosityBlock() > lastLumiOfFit))
    {
      beginTimeOfFit = lumiBlock.beginTime().value();
      beginLumiOfFit = lumiBlock.luminosityBlock();
      lumiCounter++;
      lumiCounterHisto++;
    }
  else if ((lumiCounter != 0) && (lumiBlock.luminosityBlock() >= (beginLumiOfFit+lumiCounter))) { lumiCounter++; lumiCounterHisto++; }
}


void Vx3DHLTAnalyzer::endLuminosityBlock(const LuminosityBlock& lumiBlock,
					 const EventSetup& iSetup)
{
  stringstream histTitle;
  int goodData;
  unsigned int nParams = 9;

  if ((lumiCounter%nLumiReset == 0) && (nLumiReset != 0) && (beginTimeOfFit != 0) && (runNumber != 0))
    {
      endTimeOfFit  = lumiBlock.endTime().value();
      endLumiOfFit  = lumiBlock.luminosityBlock();
      lastLumiOfFit = endLumiOfFit;
      vector<double> vals;

      hitCounter->ShiftFillLast((double)totalHits, std::sqrt((double)totalHits), nLumiReset);

      if (lastLumiOfFit % prescaleHistory == 0)
	{
	  hitCountHistory->getTH1()->SetBinContent(lastLumiOfFit, (double)totalHits);
	  hitCountHistory->getTH1()->SetBinError(lastLumiOfFit, std::sqrt((double)totalHits));
	}

      if (dataFromFit == true)
	{
	  vector<double> fitResults;

	  fitResults.push_back(Vx_X->getTH1()->GetRMS()*Vx_X->getTH1()->GetRMS());
	  fitResults.push_back(Vx_Y->getTH1()->GetRMS()*Vx_Y->getTH1()->GetRMS());
	  fitResults.push_back(Vx_Z->getTH1()->GetRMS()*Vx_Z->getTH1()->GetRMS());
	  fitResults.push_back(0.0);
	  fitResults.push_back(0.0);
	  fitResults.push_back(0.0);
	  fitResults.push_back(Vx_X->getTH1()->GetMean());
	  fitResults.push_back(Vx_Y->getTH1()->GetMean());
	  fitResults.push_back(Vx_Z->getTH1()->GetMean());
	  for (unsigned int i = 0; i < nParams; i++) fitResults.push_back(0.0);
	  
	  goodData = MyFit(&fitResults);	  	      

	  if (internalDebug == true) 
	    {
	      cout << "goodData --> " << goodData << endl;
	      cout << "Used vertices --> " << counterVx << endl;
	      cout << "var x -->  " << fitResults[0] << " +/- " << fitResults[0+nParams] << endl;
	      cout << "var y -->  " << fitResults[1] << " +/- " << fitResults[1+nParams] << endl;
	      cout << "var z -->  " << fitResults[2] << " +/- " << fitResults[2+nParams] << endl;
	      cout << "cov xy --> " << fitResults[3] << " +/- " << fitResults[3+nParams] << endl;
	      cout << "dydz   --> " << fitResults[4] << " +/- " << fitResults[4+nParams] << endl;
	      cout << "dxdz   --> " << fitResults[5] << " +/- " << fitResults[5+nParams] << endl;
	      cout << "mean x --> " << fitResults[6] << " +/- " << fitResults[6+nParams] << endl;
	      cout << "mean y --> " << fitResults[7] << " +/- " << fitResults[7+nParams] << endl;
	      cout << "mean z --> " << fitResults[8] << " +/- " << fitResults[8+nParams] << endl;
	    }

	  if (goodData == 0)
	    {		 
	      vals.push_back(fitResults[6]);
	      vals.push_back(fitResults[7]);
	      vals.push_back(fitResults[8]);
	      vals.push_back(std::sqrt(std::fabs(fitResults[2])));
	      vals.push_back(fitResults[5]);
	      vals.push_back(fitResults[4]);
	      vals.push_back(std::sqrt(std::fabs(fitResults[0])));
	      vals.push_back(std::sqrt(std::fabs(fitResults[1])));

	      vals.push_back(std::pow(fitResults[6+nParams],2.));
	      vals.push_back(std::pow(fitResults[7+nParams],2.));
	      vals.push_back(std::pow(fitResults[8+nParams],2.));
	      vals.push_back(std::pow(std::fabs(fitResults[2+nParams]) / (2.*std::sqrt(std::fabs(fitResults[2]))),2.));
	      vals.push_back(std::pow(fitResults[5+nParams],2.));
	      vals.push_back(std::pow(fitResults[4+nParams],2.));
	      vals.push_back(std::pow(std::fabs(fitResults[0+nParams]) / (2.*std::sqrt(std::fabs(fitResults[0]))),2.));
	      vals.push_back(std::pow(std::fabs(fitResults[1+nParams]) / (2.*std::sqrt(std::fabs(fitResults[1]))),2.));
	    }
	  else for (unsigned int i = 0; i < 8*2; i++) vals.push_back(0.0);

	  fitResults.clear();
	}
      else
	{
	  counterVx = Vx_X->getTH1F()->GetEntries();
	    
	  if (Vx_X->getTH1F()->GetEntries() >= minNentries)
	    {
	    goodData = 0;
	    
	    vals.push_back(Vx_X->getTH1F()->GetMean());
	    vals.push_back(Vx_Y->getTH1F()->GetMean());
	    vals.push_back(Vx_Z->getTH1F()->GetMean());
	    vals.push_back(Vx_Z->getTH1F()->GetRMS());
	    vals.push_back(0.0);
	    vals.push_back(0.0);
	    vals.push_back(Vx_X->getTH1F()->GetRMS());
	    vals.push_back(Vx_Y->getTH1F()->GetRMS());
	    
	    vals.push_back(std::pow(Vx_X->getTH1F()->GetMeanError(),2.));
	    vals.push_back(std::pow(Vx_Y->getTH1F()->GetMeanError(),2.));
	    vals.push_back(std::pow(Vx_Z->getTH1F()->GetMeanError(),2.));
	    vals.push_back(std::pow(Vx_Z->getTH1F()->GetRMSError(),2.));
	    vals.push_back(0.0);
	    vals.push_back(0.0);
	    vals.push_back(std::pow(Vx_X->getTH1F()->GetRMSError(),2.));
	    vals.push_back(std::pow(Vx_Y->getTH1F()->GetRMSError(),2.));
	    }
	  else
	    {
	      goodData = -2;
	      for (unsigned int i = 0; i < 8*2; i++) vals.push_back(0.0);
	    }
	}

      // vals[0]  = X0
      // vals[1]  = Y0
      // vals[2]  = Z0
      // vals[3]  = sigmaZ0
      // vals[4]  = dxdz
      // vals[5]  = dydz
      // vals[6]  = BeamWidthX
      // vals[7]  = BeamWidthY

      // vals[8]  = err^2 X0
      // vals[9]  = err^2 Y0
      // vals[10] = err^2 Z0
      // vals[11] = err^2 sigmaZ0
      // vals[12] = err^2 dxdz
      // vals[13] = err^2 dydz
      // vals[14] = err^2 BeamWidthX
      // vals[15] = err^2 BeamWidthY

      // "goodData" CODE:
      //  0 == OK --> Reset
      // -2 == NO OK - not enough "minNentries" --> Wait for more lumisections
      // Any other number == NO OK --> Reset

      numberFits++;
      if (goodData == 0)
	{
	  writeToFile(&vals, beginTimeOfFit, endTimeOfFit, beginLumiOfFit, endLumiOfFit, 3);
	  if ((internalDebug == true) && (outputDebugFile.is_open() == true)) outputDebugFile << "Used vertices: " << counterVx << endl;

	  numberGoodFits++;

	  histTitle << "Fitted Beam Spot [cm] (Lumi start: " << beginLumiOfFit << " - Lumi end: " << endLumiOfFit << ")";
	  if (lumiCounterHisto >= maxLumiIntegration) reset("whole");
	  else reset("partial");
	}
      else
	{
	  writeToFile(&vals, beginTimeOfFit, endTimeOfFit, beginLumiOfFit, endLumiOfFit, -1);
	  if ((internalDebug == true) && (outputDebugFile.is_open() == true)) outputDebugFile << "Used vertices: " << counterVx << endl;

	  if (goodData == -2)
	    {
	      histTitle << "Fitted Beam Spot [cm] (not enough statistics)";
	      if (lumiCounter >= maxLumiIntegration) reset("whole");
	      else reset("hitCounter");
	    }
	  else
	    {
	      histTitle << "Fitted Beam Spot [cm] (problems)";
	      if (lumiCounterHisto >= maxLumiIntegration) reset("whole");
	      else reset("partial");

	      counterVx = 0;
	    }
	}

      reportSummary->Fill(numberFits != 0 ? (double)numberGoodFits/(double)numberFits : 0.0);
      reportSummaryMap->Fill(0.5, 0.5, numberFits != 0 ? (double)numberGoodFits/(double)numberFits : 0.0);

      fitResults->setAxisTitle(histTitle.str().c_str(), 1);
      
      fitResults->setBinContent(1, 9, vals[0]);
      fitResults->setBinContent(1, 8, vals[1]);
      fitResults->setBinContent(1, 7, vals[2]);
      fitResults->setBinContent(1, 6, vals[3]);
      fitResults->setBinContent(1, 5, vals[4]);
      fitResults->setBinContent(1, 4, vals[5]);
      fitResults->setBinContent(1, 3, vals[6]);
      fitResults->setBinContent(1, 2, vals[7]);
      fitResults->setBinContent(1, 1, counterVx);
      
      fitResults->setBinContent(2, 9, std::sqrt(vals[8]));
      fitResults->setBinContent(2, 8, std::sqrt(vals[9]));
      fitResults->setBinContent(2, 7, std::sqrt(vals[10]));
      fitResults->setBinContent(2, 6, std::sqrt(vals[11]));
      fitResults->setBinContent(2, 5, std::sqrt(vals[12]));
      fitResults->setBinContent(2, 4, std::sqrt(vals[13]));
      fitResults->setBinContent(2, 3, std::sqrt(vals[14]));
      fitResults->setBinContent(2, 2, std::sqrt(vals[15]));
      fitResults->setBinContent(2, 1, std::sqrt(counterVx));

      // Linear fit to the historical plots
      TF1* myLinFit = new TF1("myLinFit", "[0] + [1]*x", mXlumi->getTH1()->GetXaxis()->GetXmin(), mXlumi->getTH1()->GetXaxis()->GetXmax());
      myLinFit->SetLineColor(2);
      myLinFit->SetLineWidth(2);
      myLinFit->SetParName(0,"Intercept");
      myLinFit->SetParName(1,"Slope");

      mXlumi->ShiftFillLast(vals[0], std::sqrt(vals[8]), nLumiReset);
      myLinFit->SetParameter(0, mXlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      mXlumi->getTH1()->Fit("myLinFit","QR");

      mYlumi->ShiftFillLast(vals[1], std::sqrt(vals[9]), nLumiReset);
      myLinFit->SetParameter(0, mYlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      mYlumi->getTH1()->Fit("myLinFit","QR");

      mZlumi->ShiftFillLast(vals[2], std::sqrt(vals[10]), nLumiReset);
      myLinFit->SetParameter(0, mZlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      mZlumi->getTH1()->Fit("myLinFit","QR");

      sXlumi->ShiftFillLast(vals[6], std::sqrt(vals[14]), nLumiReset);
      myLinFit->SetParameter(0, sXlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      sXlumi->getTH1()->Fit("myLinFit","QR");

      sYlumi->ShiftFillLast(vals[7], std::sqrt(vals[15]), nLumiReset);
      myLinFit->SetParameter(0, sYlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      sYlumi->getTH1()->Fit("myLinFit","QR");

      sZlumi->ShiftFillLast(vals[3], std::sqrt(vals[11]), nLumiReset);
      myLinFit->SetParameter(0, sZlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      sZlumi->getTH1()->Fit("myLinFit","QR");

      dxdzlumi->ShiftFillLast(vals[4], std::sqrt(vals[12]), nLumiReset);
      myLinFit->SetParameter(0, dxdzlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      dxdzlumi->getTH1()->Fit("myLinFit","QR");

      dydzlumi->ShiftFillLast(vals[5], std::sqrt(vals[13]), nLumiReset);
      myLinFit->SetParameter(0, dydzlumi->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      dydzlumi->getTH1()->Fit("myLinFit","QR");
      
      goodVxCounter->ShiftFillLast((double)counterVx, std::sqrt((double)counterVx), nLumiReset);      
      myLinFit->SetParameter(0, goodVxCounter->getTH1()->GetMean(2));
      myLinFit->SetParameter(1, 0.0);
      goodVxCounter->getTH1()->Fit("myLinFit","QR");

      if (lastLumiOfFit % prescaleHistory == 0)
	{
	  goodVxCountHistory->getTH1()->SetBinContent(lastLumiOfFit, (double)counterVx);
	  goodVxCountHistory->getTH1()->SetBinError(lastLumiOfFit, std::sqrt((double)counterVx));
	}

      delete myLinFit;

      vals.clear();
    }
  else if (nLumiReset == 0)
    {
      histTitle << "Fitted Beam Spot [cm] (no ongoing fits)";
      fitResults->setAxisTitle(histTitle.str().c_str(), 1);
      reportSummaryMap->Fill(0.5, 0.5, 1.0);
      hitCounter->ShiftFillLast(totalHits, std::sqrt(totalHits), 1);
      reset("nohisto");
    }
}


void Vx3DHLTAnalyzer::beginJob()
{
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();
 
  // ### Set internal variables ###
  nBinsHistoricalPlot = 80;
  nBinsWholeHistory   = 3000; // Corresponds to about 20h of data taking: 20h * 60min * 60s / 23s per lumi-block = 3130
  // ##############################

  if ( dbe ) 
    {
      dbe->setCurrentFolder("BeamPixel");

      Vx_X = dbe->book1D("vertex x", "Primary Vertex X Coordinate Distribution", int(rint(xRange/xStep)), -xRange/2., xRange/2.);
      Vx_Y = dbe->book1D("vertex y", "Primary Vertex Y Coordinate Distribution", int(rint(yRange/yStep)), -yRange/2., yRange/2.);
      Vx_Z = dbe->book1D("vertex z", "Primary Vertex Z Coordinate Distribution", int(rint(zRange/zStep)), -zRange/2., zRange/2.);
      Vx_X->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_X->setAxisTitle("Entries [#]",2);
      Vx_Y->setAxisTitle("Primary Vertices Y [cm]",1);
      Vx_Y->setAxisTitle("Entries [#]",2);
      Vx_Z->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_Z->setAxisTitle("Entries [#]",2);
 
      mXlumi = dbe->book1D("muX vs lumi", "\\mu_{x} vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      mYlumi = dbe->book1D("muY vs lumi", "\\mu_{y} vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      mZlumi = dbe->book1D("muZ vs lumi", "\\mu_{z} vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      mXlumi->setAxisTitle("Lumisection [#]",1);
      mXlumi->setAxisTitle("\\mu_{x} [cm]",2);
      mXlumi->getTH1()->SetOption("E1");
      mYlumi->setAxisTitle("Lumisection [#]",1);
      mYlumi->setAxisTitle("\\mu_{y} [cm]",2);
      mYlumi->getTH1()->SetOption("E1");
      mZlumi->setAxisTitle("Lumisection [#]",1);
      mZlumi->setAxisTitle("\\mu_{z} [cm]",2);
      mZlumi->getTH1()->SetOption("E1");

      sXlumi = dbe->book1D("sigmaX vs lumi", "\\sigma_{x} vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      sYlumi = dbe->book1D("sigmaY vs lumi", "\\sigma_{y} vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      sZlumi = dbe->book1D("sigmaZ vs lumi", "\\sigma_{z} vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      sXlumi->setAxisTitle("Lumisection [#]",1);
      sXlumi->setAxisTitle("\\sigma_{x} [cm]",2);
      sXlumi->getTH1()->SetOption("E1");
      sYlumi->setAxisTitle("Lumisection [#]",1);
      sYlumi->setAxisTitle("\\sigma_{y} [cm]",2);
      sYlumi->getTH1()->SetOption("E1");
      sZlumi->setAxisTitle("Lumisection [#]",1);
      sZlumi->setAxisTitle("\\sigma_{z} [cm]",2);
      sZlumi->getTH1()->SetOption("E1");

      dxdzlumi = dbe->book1D("dxdz vs lumi", "dX/dZ vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      dydzlumi = dbe->book1D("dydz vs lumi", "dY/dZ vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      dxdzlumi->setAxisTitle("Lumisection [#]",1);
      dxdzlumi->setAxisTitle("dX/dZ [rad]",2);
      dxdzlumi->getTH1()->SetOption("E1");
      dydzlumi->setAxisTitle("Lumisection [#]",1);
      dydzlumi->setAxisTitle("dY/dZ [rad]",2);
      dydzlumi->getTH1()->SetOption("E1");

      Vx_ZX = dbe->book2D("vertex zx", "Primary Vertex ZX Coordinate Distribution", int(rint(zRange/zStep/5.)), -zRange/2., zRange/2., int(rint(xRange/xStep/5.)), -xRange/2., xRange/2.);
      Vx_ZY = dbe->book2D("vertex zy", "Primary Vertex ZY Coordinate Distribution", int(rint(zRange/zStep/5.)), -zRange/2., zRange/2., int(rint(yRange/yStep/5.)), -yRange/2., yRange/2.);
      Vx_XY = dbe->book2D("vertex xy", "Primary Vertex XY Coordinate Distribution", int(rint(xRange/xStep/5.)), -xRange/2., xRange/2., int(rint(yRange/yStep/5.)), -yRange/2., yRange/2.);
      Vx_ZX->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZX->setAxisTitle("Primary Vertices X [cm]",2);
      Vx_ZX->setAxisTitle("Entries [#]",3);
      Vx_ZY->setAxisTitle("Primary Vertices Z [cm]",1);
      Vx_ZY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_ZY->setAxisTitle("Entries [#]",3);
      Vx_XY->setAxisTitle("Primary Vertices X [cm]",1);
      Vx_XY->setAxisTitle("Primary Vertices Y [cm]",2);
      Vx_XY->setAxisTitle("Entries [#]",3);

      hitCounter = dbe->book1D("pixelHits vs lumi", "# Pixel-Hits vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      hitCounter->setAxisTitle("Lumisection [#]",1);
      hitCounter->setAxisTitle("Pixel-Hits [#]",2);
      hitCounter->getTH1()->SetOption("E1");

      hitCountHistory = dbe->book1D("hist pixelHits vs lumi", "History: # Pixel-Hits vs. Lumi", nBinsWholeHistory, 0.5, (double)nBinsWholeHistory+0.5);
      hitCountHistory->setAxisTitle("Lumisection [#]",1);
      hitCountHistory->setAxisTitle("Pixel-Hits [#]",2);
      hitCountHistory->getTH1()->SetOption("E1");

      goodVxCounter = dbe->book1D("good vertices vs lumi", "# Good vertices vs. Lumisection", nBinsHistoricalPlot, 0.5, (double)nBinsHistoricalPlot+0.5);
      goodVxCounter->setAxisTitle("Lumisection [#]",1);
      goodVxCounter->setAxisTitle("Good vertices [#]",2);
      goodVxCounter->getTH1()->SetOption("E1");

      goodVxCountHistory = dbe->book1D("hist good vx vs lumi", "History: # Good vx vs. Lumi", nBinsWholeHistory, 0.5, (double)nBinsWholeHistory+0.5);
      goodVxCountHistory->setAxisTitle("Lumisection [#]",1);
      goodVxCountHistory->setAxisTitle("Good vertices [#]",2);
      goodVxCountHistory->getTH1()->SetOption("E1");

      fitResults = dbe->book2D("fit results","Results of Beam Spot Fit", 2, 0., 2., 9, 0., 9.);
      fitResults->setAxisTitle("Fitted Beam Spot [cm]", 1);
      fitResults->setBinLabel(9, "X", 2);
      fitResults->setBinLabel(8, "Y", 2);
      fitResults->setBinLabel(7, "Z", 2);
      fitResults->setBinLabel(6, "\\sigma_{Z}", 2);
      fitResults->setBinLabel(5, "#frac{dX}{dZ}[rad]", 2);
      fitResults->setBinLabel(4, "#frac{dY}{dZ}[rad]", 2);
      fitResults->setBinLabel(3, "\\sigma_{X}", 2);
      fitResults->setBinLabel(2, "\\sigma_{Y}", 2);
      fitResults->setBinLabel(1, "Vertices", 2);
      fitResults->setBinLabel(1, "Value", 1);
      fitResults->setBinLabel(2, "Stat. Error", 1);
      fitResults->getTH1()->SetOption("text");

      dbe->setCurrentFolder("BeamPixel/EventInfo");
      reportSummary = dbe->bookFloat("reportSummary");
      reportSummary->Fill(0.);
      reportSummaryMap = dbe->book2D("reportSummaryMap","Pixel-Vertices Beam Spot: % Good Fits", 1, 0., 1., 1, 0., 1.);
      reportSummaryMap->Fill(0.5, 0.5, 0.);
      dbe->setCurrentFolder("BeamPixel/EventInfo/reportSummaryContents");

      // Convention for reportSummary and reportSummaryMap:
      // - 0%  at the moment of creation of the histogram
      // - n%  numberGoodFits / numberFits
    }

  // ### Set internal variables ###
  reset("scratch");
  prescaleHistory      = 1;
  maxLumiIntegration   = 15;
  minVxDoF             = 10.;
  // For vertex fitter without track-weight: d.o.f. = 2*NTracks - 3
  // For vertex fitter with track-weight:    d.o.f. = sum_NTracks(2*track_weight) - 3
  internalDebug        = false;
  considerVxCovariance = true;
  pi = 3.141592653589793238;
  // ##############################
}


void Vx3DHLTAnalyzer::endJob() { reset("scratch"); }


// Define this as a plug-in
DEFINE_FWK_MODULE(Vx3DHLTAnalyzer);
