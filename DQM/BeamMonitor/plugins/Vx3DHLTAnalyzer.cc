/*
  \File Vx3DHLTAnalyzer.cc
  \Display Beam-spot monitor entirely based on pixel detector information
           the monitoring is based on a 3D fit to the vertex cloud
  \Author Mauro Dinardo
  \Version $ Revision: 3.5 $
  \Date $ Date: 2010/23/02 13:15:00 $
*/

#include "DQM/BeamMonitor/plugins/Vx3DHLTAnalyzer.h"

#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include <Math/Minimizer.h>
#include <Math/Factory.h>
#include <Math/Functor.h>

// ### Calling namespaces ###
using namespace std;
using namespace edm;
using namespace reco;

Vx3DHLTAnalyzer::Vx3DHLTAnalyzer(const ParameterSet& iConfig) {
  debugMode = true;
  nLumiFit = 2;  // Number of integrated lumis to perform the fit
  maxLumiIntegration =
      15;  // If failing fits, this is the maximum number of integrated lumis after which a reset is issued
  nLumiXaxisRange = 5000;  // Correspond to about 32h of data taking: 32h * 60min * 60s / 23s per lumi-block = 5009
  dataFromFit = true;      // The Beam Spot data can be either taken from the histograms or from the fit results
  minNentries = 20;        // Minimum number of good vertices to perform the fit
  xRange = 0.8;            // [cm]
  xStep = 0.001;           // [cm]
  yRange = 0.8;            // [cm]
  yStep = 0.001;           // [cm]
  zRange = 30.;            // [cm]
  zStep = 0.04;            // [cm]
  VxErrCorr = 1.3;
  minVxDoF = 10.;  // Good-vertex selection cut
  // For vertex fitter without track-weight: d.o.f. = 2*NTracks - 3
  // For vertex fitter with track-weight:    d.o.f. = sum_NTracks(2*track_weight) - 3
  minVxWgt = 0.5;  // Good-vertex selection cut
  fileName = "BeamPixelResults.txt";

  vertexCollection = consumes<VertexCollection>(
      iConfig.getUntrackedParameter<InputTag>("vertexCollection", InputTag("pixelVertices")));
  pixelHitCollection = consumes<SiPixelRecHitCollection>(
      iConfig.getUntrackedParameter<InputTag>("pixelHitCollection", InputTag("siPixelRecHits")));

  debugMode = iConfig.getParameter<bool>("debugMode");
  nLumiFit = iConfig.getParameter<unsigned int>("nLumiFit");
  maxLumiIntegration = iConfig.getParameter<unsigned int>("maxLumiIntegration");
  nLumiXaxisRange = iConfig.getParameter<unsigned int>("nLumiXaxisRange");
  dataFromFit = iConfig.getParameter<bool>("dataFromFit");
  minNentries = iConfig.getParameter<unsigned int>("minNentries");
  xRange = iConfig.getParameter<double>("xRange");
  xStep = iConfig.getParameter<double>("xStep");
  yRange = iConfig.getParameter<double>("yRange");
  yStep = iConfig.getParameter<double>("yStep");
  zRange = iConfig.getParameter<double>("zRange");
  zStep = iConfig.getParameter<double>("zStep");
  VxErrCorr = iConfig.getParameter<double>("VxErrCorr");
  minVxDoF = iConfig.getParameter<double>("minVxDoF");
  minVxWgt = iConfig.getParameter<double>("minVxWgt");
  fileName = iConfig.getParameter<string>("fileName");

  // ### Set internal variables ###
  nParams = 9;  // Number of free parameters in the fit
  internalDebug = false;
  considerVxCovariance = true;  // Deconvolute vertex covariance matrix
  pi = 3.141592653589793238;
  // ##############################
}

Vx3DHLTAnalyzer::~Vx3DHLTAnalyzer() {}

void Vx3DHLTAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
  Handle<VertexCollection> Vx3DCollection;
  iEvent.getByToken(vertexCollection, Vx3DCollection);

  unsigned int i, j;
  double det;
  VertexType MyVertex;

  if (runNumber != iEvent.id().run()) {
    reset("scratch");
    runNumber = iEvent.id().run();

    if (debugMode == true) {
      stringstream debugFile;
      string tmp(fileName);

      if (outputDebugFile.is_open() == true)
        outputDebugFile.close();
      tmp.erase(strlen(fileName.c_str()) - 4, 4);
      debugFile << tmp.c_str() << "_Run" << iEvent.id().run() << ".txt";
      outputDebugFile.open(debugFile.str().c_str(), ios::out);
      outputDebugFile.close();
      outputDebugFile.open(debugFile.str().c_str(), ios::app);
    }

    dqmBeginLuminosityBlock(iEvent.getLuminosityBlock(), iSetup);
  } else if (beginTimeOfFit != 0) {
    totalHits += HitCounter(iEvent);

    if (internalDebug == true) {
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tI found " << totalHits << " pixel hits until now";
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tIn this event there are " << Vx3DCollection->size() << " vertex cadidates";
    }

    for (vector<Vertex>::const_iterator it3DVx = Vx3DCollection->begin(); it3DVx != Vx3DCollection->end(); it3DVx++) {
      if (internalDebug == true) {
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tVertex selections:";
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tEvent ID = " << iEvent.id();
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tVertex number = " << it3DVx - Vx3DCollection->begin();
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tisValid = " << it3DVx->isValid();
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tisFake = " << it3DVx->isFake();
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tnodof = " << it3DVx->ndof();
        edm::LogInfo("Vx3DHLTAnalyzer") << "\ttracksSize = " << it3DVx->tracksSize();
      }

      if ((it3DVx->isValid() == true) && (it3DVx->isFake() == false) && (it3DVx->ndof() >= minVxDoF) &&
          (it3DVx->tracksSize() > 0) && ((it3DVx->ndof() + 3.) / ((double)it3DVx->tracksSize()) >= 2. * minVxWgt)) {
        for (i = 0; i < DIM; i++) {
          for (j = 0; j < DIM; j++) {
            MyVertex.Covariance[i][j] = it3DVx->covariance(i, j);
            if (isNotFinite(MyVertex.Covariance[i][j]) == true)
              break;
          }

          if (j != DIM)
            break;
        }

        if (i == DIM)
          det = std::fabs(MyVertex.Covariance[0][0]) *
                    (std::fabs(MyVertex.Covariance[1][1]) * std::fabs(MyVertex.Covariance[2][2]) -
                     MyVertex.Covariance[1][2] * MyVertex.Covariance[1][2]) -
                MyVertex.Covariance[0][1] * (MyVertex.Covariance[0][1] * std::fabs(MyVertex.Covariance[2][2]) -
                                             MyVertex.Covariance[0][2] * MyVertex.Covariance[1][2]) +
                MyVertex.Covariance[0][2] * (MyVertex.Covariance[0][1] * MyVertex.Covariance[1][2] -
                                             MyVertex.Covariance[0][2] * std::fabs(MyVertex.Covariance[1][1]));

        if ((i == DIM) && (det > 0.)) {
          if (internalDebug == true)
            edm::LogInfo("Vx3DHLTAnalyzer") << "\tVertex accepted !";

          MyVertex.x = it3DVx->x();
          MyVertex.y = it3DVx->y();
          MyVertex.z = it3DVx->z();
          Vertices.push_back(MyVertex);

          Vx_X->Fill(it3DVx->x());
          Vx_Y->Fill(it3DVx->y());
          Vx_Z->Fill(it3DVx->z());

          Vx_ZX->Fill(it3DVx->z(), it3DVx->x());
          Vx_ZY->Fill(it3DVx->z(), it3DVx->y());
          Vx_XY->Fill(it3DVx->x(), it3DVx->y());

          Vx_X_Cum->Fill(it3DVx->x());
          Vx_Y_Cum->Fill(it3DVx->y());
          Vx_Z_Cum->Fill(it3DVx->z());

          Vx_ZX_Cum->Fill(it3DVx->z(), it3DVx->x());
          Vx_ZY_Cum->Fill(it3DVx->z(), it3DVx->y());
          Vx_XY_Cum->Fill(it3DVx->x(), it3DVx->y());
        } else if (internalDebug == true) {
          edm::LogInfo("Vx3DHLTAnalyzer") << "\tVertex discarded !";

          for (i = 0; i < DIM; i++)
            for (j = 0; j < DIM; j++)
              edm::LogInfo("Vx3DHLTAnalyzer") << "(i,j) --> " << i << "," << j << " --> " << MyVertex.Covariance[i][j];
        }
      } else if (internalDebug == true)
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tVertex discarded !";
    }
  }
}

unsigned int Vx3DHLTAnalyzer::HitCounter(const Event& iEvent) {
  Handle<SiPixelRecHitCollection> rechitspixel;
  iEvent.getByToken(pixelHitCollection, rechitspixel);

  unsigned int counter = 0;

  for (SiPixelRecHitCollection::const_iterator j = rechitspixel->begin(); j != rechitspixel->end(); j++)
    for (edmNew::DetSet<SiPixelRecHit>::const_iterator h = j->begin(); h != j->end(); h++)
      counter += h->cluster()->size();

  return counter;
}

string Vx3DHLTAnalyzer::formatTime(const time_t& t) {
  char ts[25];
  strftime(ts, sizeof(ts), "%Y.%m.%d %H:%M:%S %Z", gmtime(&t));

  string ts_string(ts);

  return ts_string;
}

double Vx3DHLTAnalyzer::Gauss3DFunc(const double* par) {
  double K[DIM][DIM];  // Covariance Matrix
  double M[DIM][DIM];  // K^-1
  double det;
  double sumlog = 0.;

  //  par[0] = K(0,0) --> Var[X]
  //  par[1] = K(1,1) --> Var[Y]
  //  par[2] = K(2,2) --> Var[Z]
  //  par[3] = K(0,1) = K(1,0) --> Cov[X,Y]
  //  par[4] = K(1,2) = K(2,1) --> Cov[Y,Z] --> dy/dz
  //  par[5] = K(0,2) = K(2,0) --> Cov[X,Z] --> dx/dz
  //  par[6] = mean x
  //  par[7] = mean y
  //  par[8] = mean z

  counterVx = 0;
  for (unsigned int i = 0; i < Vertices.size(); i++) {
    if ((std::sqrt((Vertices[i].x - xPos) * (Vertices[i].x - xPos) + (Vertices[i].y - yPos) * (Vertices[i].y - yPos)) <=
         maxTransRadius) &&
        (std::fabs(Vertices[i].z - zPos) <= maxLongLength)) {
      if (considerVxCovariance == true) {
        K[0][0] = std::fabs(par[0]) + VxErrCorr * VxErrCorr * std::fabs(Vertices[i].Covariance[0][0]);
        K[1][1] = std::fabs(par[1]) + VxErrCorr * VxErrCorr * std::fabs(Vertices[i].Covariance[1][1]);
        K[2][2] = std::fabs(par[2]) + VxErrCorr * VxErrCorr * std::fabs(Vertices[i].Covariance[2][2]);
        K[0][1] = K[1][0] = par[3] + VxErrCorr * VxErrCorr * Vertices[i].Covariance[0][1];
        K[1][2] = K[2][1] = par[4] * (std::fabs(par[2]) - std::fabs(par[1])) - par[5] * par[3] +
                            VxErrCorr * VxErrCorr * Vertices[i].Covariance[1][2];
        K[0][2] = K[2][0] = par[5] * (std::fabs(par[2]) - std::fabs(par[0])) - par[4] * par[3] +
                            VxErrCorr * VxErrCorr * Vertices[i].Covariance[0][2];
      } else {
        K[0][0] = std::fabs(par[0]);
        K[1][1] = std::fabs(par[1]);
        K[2][2] = std::fabs(par[2]);
        K[0][1] = K[1][0] = par[3];
        K[1][2] = K[2][1] = par[4] * (std::fabs(par[2]) - std::fabs(par[1])) - par[5] * par[3];
        K[0][2] = K[2][0] = par[5] * (std::fabs(par[2]) - std::fabs(par[0])) - par[4] * par[3];
      }

      det = K[0][0] * (K[1][1] * K[2][2] - K[1][2] * K[1][2]) - K[0][1] * (K[0][1] * K[2][2] - K[0][2] * K[1][2]) +
            K[0][2] * (K[0][1] * K[1][2] - K[0][2] * K[1][1]);

      M[0][0] = (K[1][1] * K[2][2] - K[1][2] * K[1][2]) / det;
      M[1][1] = (K[0][0] * K[2][2] - K[0][2] * K[0][2]) / det;
      M[2][2] = (K[0][0] * K[1][1] - K[0][1] * K[0][1]) / det;
      M[0][1] = M[1][0] = (K[0][2] * K[1][2] - K[0][1] * K[2][2]) / det;
      M[1][2] = M[2][1] = (K[0][2] * K[0][1] - K[1][2] * K[0][0]) / det;
      M[0][2] = M[2][0] = (K[0][1] * K[1][2] - K[0][2] * K[1][1]) / det;

      sumlog += double(DIM) * std::log(2. * pi) + std::log(std::fabs(det)) +
                (M[0][0] * (Vertices[i].x - par[6]) * (Vertices[i].x - par[6]) +
                 M[1][1] * (Vertices[i].y - par[7]) * (Vertices[i].y - par[7]) +
                 M[2][2] * (Vertices[i].z - par[8]) * (Vertices[i].z - par[8]) +
                 2. * M[0][1] * (Vertices[i].x - par[6]) * (Vertices[i].y - par[7]) +
                 2. * M[1][2] * (Vertices[i].y - par[7]) * (Vertices[i].z - par[8]) +
                 2. * M[0][2] * (Vertices[i].x - par[6]) * (Vertices[i].z - par[8]));

      counterVx++;
    }
  }

  return sumlog;
}

int Vx3DHLTAnalyzer::MyFit(vector<double>* vals) {
  // ############################################
  // # RETURN CODE:                             #
  // # >0 == NO OK - fit status (MINUIT manual) #
  // #  0 == OK                                 #
  // # -1 == NO OK - not finite edm             #
  // # -2 == NO OK - not enough "minNentries"   #
  // # -3 == NO OK - not finite errors          #
  // # -4 == NO OK - negative determinant       #
  // # -5 == NO OK - maxLumiIntegration reached #
  // ############################################

  if ((vals != nullptr) && (vals->size() == nParams * 2)) {
    double nSigmaXY = 10.;
    double nSigmaZ = 10.;
    double parDistanceXY = 1e-3;   // Unit: [cm]
    double parDistanceZ = 1e-2;    // Unit: [cm]
    double parDistanceddZ = 1e-3;  // Unit: [rad]
    double parDistanceCxy = 1e-5;  // Unit: [cm^2]
    double bestEdm;

    const unsigned int trials = 4;
    double largerDist[trials] = {0.1, 5., 10., 100.};

    double covxz, covyz, det;
    double deltaMean;
    int bestMovementX = 1;
    int bestMovementY = 1;
    int bestMovementZ = 1;
    int goodData;

    double edm;

    vector<double>::const_iterator it = vals->begin();

    ROOT::Math::Minimizer* Gauss3D = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
    Gauss3D->SetErrorDef(1.0);
    if (internalDebug == true)
      Gauss3D->SetPrintLevel(3);
    else
      Gauss3D->SetPrintLevel(0);

    ROOT::Math::Functor _Gauss3DFunc(this, &Vx3DHLTAnalyzer::Gauss3DFunc, nParams);
    Gauss3D->SetFunction(_Gauss3DFunc);

    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\t@@@ START FITTING @@@";

    // @@@ Fit at X-deltaMean | X | X+deltaMean @@@
    bestEdm = 1.;
    for (int i = 0; i < 3; i++) {
      deltaMean = (double(i) - 1.) * std::sqrt(*(it + 0));
      if (internalDebug == true)
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tdeltaMean --> " << deltaMean;

      Gauss3D->Clear();

      Gauss3D->SetVariable(0, "var x ", *(it + 0), parDistanceXY * parDistanceXY);
      Gauss3D->SetVariable(1, "var y ", *(it + 1), parDistanceXY * parDistanceXY);
      Gauss3D->SetVariable(2, "var z ", *(it + 2), parDistanceZ * parDistanceZ);
      Gauss3D->SetVariable(3, "cov xy", *(it + 3), parDistanceCxy);
      Gauss3D->SetVariable(4, "dydz  ", *(it + 4), parDistanceddZ);
      Gauss3D->SetVariable(5, "dxdz  ", *(it + 5), parDistanceddZ);
      Gauss3D->SetVariable(6, "mean x", *(it + 6) + deltaMean, parDistanceXY);
      Gauss3D->SetVariable(7, "mean y", *(it + 7), parDistanceXY);
      Gauss3D->SetVariable(8, "mean z", *(it + 8), parDistanceZ);

      // Set the central positions of the centroid for vertex rejection
      xPos = (*vals)[6];
      yPos = (*vals)[7];
      zPos = (*vals)[8];

      // Set dimensions of the centroid for vertex rejection
      maxTransRadius = nSigmaXY * std::sqrt(std::fabs((*vals)[0]) + std::fabs((*vals)[1])) / 2.;
      maxLongLength = nSigmaZ * std::sqrt(std::fabs((*vals)[2]));

      try {
        Gauss3D->Minimize();
      } catch (cms::Exception& er) {
        edm::LogError("Vx3DHLTAnalyzer") << "\tCaught Minuit2 exception: " << er.what();
      }
      goodData = Gauss3D->Status();
      edm = Gauss3D->Edm();

      if (counterVx < minNentries)
        goodData = -2;
      else if (isNotFinite(edm) == true) {
        goodData = -1;
        if (internalDebug == true)
          edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite edm !";
      } else
        for (unsigned int j = 0; j < nParams; j++)
          if (isNotFinite(Gauss3D->Errors()[j]) == true) {
            goodData = -3;
            if (internalDebug == true)
              edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite errors !";
            break;
          }
      if (goodData == 0) {
        covyz = Gauss3D->X()[4] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[1])) -
                Gauss3D->X()[5] * Gauss3D->X()[3];
        covxz = Gauss3D->X()[5] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[0])) -
                Gauss3D->X()[4] * Gauss3D->X()[3];

        det = std::fabs(Gauss3D->X()[0]) * (std::fabs(Gauss3D->X()[1]) * std::fabs(Gauss3D->X()[2]) - covyz * covyz) -
              Gauss3D->X()[3] * (Gauss3D->X()[3] * std::fabs(Gauss3D->X()[2]) - covxz * covyz) +
              covxz * (Gauss3D->X()[3] * covyz - covxz * std::fabs(Gauss3D->X()[1]));
        if (det < 0.) {
          goodData = -4;
          if (internalDebug == true)
            edm::LogInfo("Vx3DHLTAnalyzer") << "\tNegative determinant !";
        }
      }

      if ((goodData == 0) && (std::fabs(edm) < bestEdm)) {
        bestEdm = edm;
        bestMovementX = i;
      }
    }
    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tFound bestMovementX --> " << bestMovementX;

    // @@@ Fit at Y-deltaMean | Y | Y+deltaMean @@@
    bestEdm = 1.;
    for (int i = 0; i < 3; i++) {
      deltaMean = (double(i) - 1.) * std::sqrt(*(it + 1));
      if (internalDebug == true) {
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tdeltaMean --> " << deltaMean;
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tdeltaMean X --> " << (double(bestMovementX) - 1.) * std::sqrt(*(it + 0));
      }

      Gauss3D->Clear();

      Gauss3D->SetVariable(0, "var x ", *(it + 0), parDistanceXY * parDistanceXY);
      Gauss3D->SetVariable(1, "var y ", *(it + 1), parDistanceXY * parDistanceXY);
      Gauss3D->SetVariable(2, "var z ", *(it + 2), parDistanceZ * parDistanceZ);
      Gauss3D->SetVariable(3, "cov xy", *(it + 3), parDistanceCxy);
      Gauss3D->SetVariable(4, "dydz  ", *(it + 4), parDistanceddZ);
      Gauss3D->SetVariable(5, "dxdz  ", *(it + 5), parDistanceddZ);
      Gauss3D->SetVariable(6, "mean x", *(it + 6) + (double(bestMovementX) - 1.) * std::sqrt(*(it + 0)), parDistanceXY);
      Gauss3D->SetVariable(7, "mean y", *(it + 7) + deltaMean, parDistanceXY);
      Gauss3D->SetVariable(8, "mean z", *(it + 8), parDistanceZ);

      // Set the central positions of the centroid for vertex rejection
      xPos = Gauss3D->X()[6];
      yPos = Gauss3D->X()[7];
      zPos = Gauss3D->X()[8];

      // Set dimensions of the centroid for vertex rejection
      maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->X()[0]) + std::fabs(Gauss3D->X()[1])) / 2.;
      maxLongLength = nSigmaZ * std::sqrt(std::fabs(Gauss3D->X()[2]));

      try {
        Gauss3D->Minimize();
      } catch (cms::Exception& er) {
        edm::LogError("Vx3DHLTAnalyzer") << "\tCaught Minuit2 exception: " << er.what();
      }
      goodData = Gauss3D->Status();
      edm = Gauss3D->Edm();

      if (counterVx < minNentries)
        goodData = -2;
      else if (isNotFinite(edm) == true) {
        goodData = -1;
        if (internalDebug == true)
          edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite edm !";
      } else
        for (unsigned int j = 0; j < nParams; j++)
          if (isNotFinite(Gauss3D->Errors()[j]) == true) {
            goodData = -3;
            if (internalDebug == true)
              edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite errors !";
            break;
          }
      if (goodData == 0) {
        covyz = Gauss3D->X()[4] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[1])) -
                Gauss3D->X()[5] * Gauss3D->X()[3];
        covxz = Gauss3D->X()[5] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[0])) -
                Gauss3D->X()[4] * Gauss3D->X()[3];

        det = std::fabs(Gauss3D->X()[0]) * (std::fabs(Gauss3D->X()[1]) * std::fabs(Gauss3D->X()[2]) - covyz * covyz) -
              Gauss3D->X()[3] * (Gauss3D->X()[3] * std::fabs(Gauss3D->X()[2]) - covxz * covyz) +
              covxz * (Gauss3D->X()[3] * covyz - covxz * std::fabs(Gauss3D->X()[1]));
        if (det < 0.) {
          goodData = -4;
          if (internalDebug == true)
            edm::LogInfo("Vx3DHLTAnalyzer") << "\tNegative determinant !";
        }
      }

      if ((goodData == 0) && (std::fabs(edm) < bestEdm)) {
        bestEdm = edm;
        bestMovementY = i;
      }
    }
    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tFound bestMovementY --> " << bestMovementY;

    // @@@ Fit at Z-deltaMean | Z | Z+deltaMean @@@
    bestEdm = 1.;
    for (int i = 0; i < 3; i++) {
      deltaMean = (double(i) - 1.) * std::sqrt(*(it + 2));
      if (internalDebug == true) {
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tdeltaMean --> " << deltaMean;
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tdeltaMean X --> " << (double(bestMovementX) - 1.) * std::sqrt(*(it + 0));
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tdeltaMean Y --> " << (double(bestMovementY) - 1.) * std::sqrt(*(it + 1));
      }

      Gauss3D->Clear();

      Gauss3D->SetVariable(0, "var x ", *(it + 0), parDistanceXY * parDistanceXY);
      Gauss3D->SetVariable(1, "var y ", *(it + 1), parDistanceXY * parDistanceXY);
      Gauss3D->SetVariable(2, "var z ", *(it + 2), parDistanceZ * parDistanceZ);
      Gauss3D->SetVariable(3, "cov xy", *(it + 3), parDistanceCxy);
      Gauss3D->SetVariable(4, "dydz  ", *(it + 4), parDistanceddZ);
      Gauss3D->SetVariable(5, "dxdz  ", *(it + 5), parDistanceddZ);
      Gauss3D->SetVariable(6, "mean x", *(it + 6) + (double(bestMovementX) - 1.) * std::sqrt(*(it + 0)), parDistanceXY);
      Gauss3D->SetVariable(7, "mean y", *(it + 7) + (double(bestMovementY) - 1.) * std::sqrt(*(it + 1)), parDistanceXY);
      Gauss3D->SetVariable(8, "mean z", *(it + 8) + deltaMean, parDistanceZ);

      // Set the central positions of the centroid for vertex rejection
      xPos = Gauss3D->X()[6];
      yPos = Gauss3D->X()[7];
      zPos = Gauss3D->X()[8];

      // Set dimensions of the centroid for vertex rejection
      maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->X()[0]) + std::fabs(Gauss3D->X()[1])) / 2.;
      maxLongLength = nSigmaZ * std::sqrt(std::fabs(Gauss3D->X()[2]));

      try {
        Gauss3D->Minimize();
      } catch (cms::Exception& er) {
        edm::LogError("Vx3DHLTAnalyzer") << "\tCaught Minuit2 exception: " << er.what();
      }
      goodData = Gauss3D->Status();
      edm = Gauss3D->Edm();

      if (counterVx < minNentries)
        goodData = -2;
      else if (isNotFinite(edm) == true) {
        goodData = -1;
        if (internalDebug == true)
          edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite edm !";
      } else
        for (unsigned int j = 0; j < nParams; j++)
          if (isNotFinite(Gauss3D->Errors()[j]) == true) {
            goodData = -3;
            if (internalDebug == true)
              edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite errors !";
            break;
          }
      if (goodData == 0) {
        covyz = Gauss3D->X()[4] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[1])) -
                Gauss3D->X()[5] * Gauss3D->X()[3];
        covxz = Gauss3D->X()[5] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[0])) -
                Gauss3D->X()[4] * Gauss3D->X()[3];

        det = std::fabs(Gauss3D->X()[0]) * (std::fabs(Gauss3D->X()[1]) * std::fabs(Gauss3D->X()[2]) - covyz * covyz) -
              Gauss3D->X()[3] * (Gauss3D->X()[3] * std::fabs(Gauss3D->X()[2]) - covxz * covyz) +
              covxz * (Gauss3D->X()[3] * covyz - covxz * std::fabs(Gauss3D->X()[1]));
        if (det < 0.) {
          goodData = -4;
          if (internalDebug == true)
            edm::LogInfo("Vx3DHLTAnalyzer") << "\tNegative determinant !";
        }
      }

      if ((goodData == 0) && (std::fabs(edm) < bestEdm)) {
        bestEdm = edm;
        bestMovementZ = i;
      }
    }
    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tFound bestMovementZ --> " << bestMovementZ;

    Gauss3D->Clear();

    // @@@ FINAL FIT @@@
    Gauss3D->SetVariable(0, "var x ", *(it + 0), parDistanceXY * parDistanceXY);
    Gauss3D->SetVariable(1, "var y ", *(it + 1), parDistanceXY * parDistanceXY);
    Gauss3D->SetVariable(2, "var z ", *(it + 2), parDistanceZ * parDistanceZ);
    Gauss3D->SetVariable(3, "cov xy", *(it + 3), parDistanceCxy);
    Gauss3D->SetVariable(4, "dydz  ", *(it + 4), parDistanceddZ);
    Gauss3D->SetVariable(5, "dxdz  ", *(it + 5), parDistanceddZ);
    Gauss3D->SetVariable(6, "mean x", *(it + 6) + (double(bestMovementX) - 1.) * std::sqrt(*(it + 0)), parDistanceXY);
    Gauss3D->SetVariable(7, "mean y", *(it + 7) + (double(bestMovementY) - 1.) * std::sqrt(*(it + 1)), parDistanceXY);
    Gauss3D->SetVariable(8, "mean z", *(it + 8) + (double(bestMovementZ) - 1.) * std::sqrt(*(it + 2)), parDistanceZ);

    // Set the central positions of the centroid for vertex rejection
    xPos = (*vals)[6];
    yPos = (*vals)[7];
    zPos = (*vals)[8];

    // Set dimensions of the centroid for vertex rejection
    maxTransRadius = nSigmaXY * std::sqrt(std::fabs((*vals)[0]) + std::fabs((*vals)[1])) / 2.;
    maxLongLength = nSigmaZ * std::sqrt(std::fabs((*vals)[2]));

    try {
      Gauss3D->Minimize();
    } catch (cms::Exception& er) {
      edm::LogError("Vx3DHLTAnalyzer") << "\tCaught Minuit2 exception: " << er.what();
    }
    goodData = Gauss3D->Status();
    edm = Gauss3D->Edm();

    if (counterVx < minNentries)
      goodData = -2;
    else if (isNotFinite(edm) == true) {
      goodData = -1;
      if (internalDebug == true)
        edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite edm !";
    } else
      for (unsigned int j = 0; j < nParams; j++)
        if (isNotFinite(Gauss3D->Errors()[j]) == true) {
          goodData = -3;
          if (internalDebug == true)
            edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite errors !";
          break;
        }
    if (goodData == 0) {
      covyz = Gauss3D->X()[4] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[1])) -
              Gauss3D->X()[5] * Gauss3D->X()[3];
      covxz = Gauss3D->X()[5] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[0])) -
              Gauss3D->X()[4] * Gauss3D->X()[3];

      det = std::fabs(Gauss3D->X()[0]) * (std::fabs(Gauss3D->X()[1]) * std::fabs(Gauss3D->X()[2]) - covyz * covyz) -
            Gauss3D->X()[3] * (Gauss3D->X()[3] * std::fabs(Gauss3D->X()[2]) - covxz * covyz) +
            covxz * (Gauss3D->X()[3] * covyz - covxz * std::fabs(Gauss3D->X()[1]));
      if (det < 0.) {
        goodData = -4;
        if (internalDebug == true)
          edm::LogInfo("Vx3DHLTAnalyzer") << "\tNegative determinant !";
      }
    }

    // @@@ FIT WITH DIFFERENT PARAMETER DISTANCES @@@
    for (unsigned int i = 0; i < trials; i++) {
      if ((goodData != 0) && (goodData != -2)) {
        Gauss3D->Clear();

        if (internalDebug == true)
          edm::LogInfo("Vx3DHLTAnalyzer") << "\tFIT WITH DIFFERENT PARAMETER DISTANCES - STEP " << i + 1;

        Gauss3D->SetVariable(0, "var x ", *(it + 0), parDistanceXY * parDistanceXY * largerDist[i]);
        Gauss3D->SetVariable(1, "var y ", *(it + 1), parDistanceXY * parDistanceXY * largerDist[i]);
        Gauss3D->SetVariable(2, "var z ", *(it + 2), parDistanceZ * parDistanceZ * largerDist[i]);
        Gauss3D->SetVariable(3, "cov xy", *(it + 3), parDistanceCxy * largerDist[i]);
        Gauss3D->SetVariable(4, "dydz  ", *(it + 4), parDistanceddZ * largerDist[i]);
        Gauss3D->SetVariable(5, "dxdz  ", *(it + 5), parDistanceddZ * largerDist[i]);
        Gauss3D->SetVariable(6,
                             "mean x",
                             *(it + 6) + (double(bestMovementX) - 1.) * std::sqrt(*(it + 0)),
                             parDistanceXY * largerDist[i]);
        Gauss3D->SetVariable(7,
                             "mean y",
                             *(it + 7) + (double(bestMovementY) - 1.) * std::sqrt(*(it + 1)),
                             parDistanceXY * largerDist[i]);
        Gauss3D->SetVariable(
            8, "mean z", *(it + 8) + (double(bestMovementZ) - 1.) * std::sqrt(*(it + 2)), parDistanceZ * largerDist[i]);

        // Set the central positions of the centroid for vertex rejection
        xPos = Gauss3D->X()[6];
        yPos = Gauss3D->X()[7];
        zPos = Gauss3D->X()[8];

        // Set dimensions of the centroid for vertex rejection
        maxTransRadius = nSigmaXY * std::sqrt(std::fabs(Gauss3D->X()[0]) + std::fabs(Gauss3D->X()[1])) / 2.;
        maxLongLength = nSigmaZ * std::sqrt(std::fabs(Gauss3D->X()[2]));

        try {
          Gauss3D->Minimize();
        } catch (cms::Exception& er) {
          edm::LogError("Vx3DHLTAnalyzer") << "\tCaught Minuit2 exception: " << er.what();
        }
        goodData = Gauss3D->Status();
        edm = Gauss3D->Edm();

        if (counterVx < minNentries)
          goodData = -2;
        else if (isNotFinite(edm) == true) {
          goodData = -1;
          if (internalDebug == true)
            edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite edm !";
        } else
          for (unsigned int j = 0; j < nParams; j++)
            if (isNotFinite(Gauss3D->Errors()[j]) == true) {
              goodData = -3;
              if (internalDebug == true)
                edm::LogInfo("Vx3DHLTAnalyzer") << "\tNot finite errors !";
              break;
            }
        if (goodData == 0) {
          covyz = Gauss3D->X()[4] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[1])) -
                  Gauss3D->X()[5] * Gauss3D->X()[3];
          covxz = Gauss3D->X()[5] * (std::fabs(Gauss3D->X()[2]) - std::fabs(Gauss3D->X()[0])) -
                  Gauss3D->X()[4] * Gauss3D->X()[3];

          det = std::fabs(Gauss3D->X()[0]) * (std::fabs(Gauss3D->X()[1]) * std::fabs(Gauss3D->X()[2]) - covyz * covyz) -
                Gauss3D->X()[3] * (Gauss3D->X()[3] * std::fabs(Gauss3D->X()[2]) - covxz * covyz) +
                covxz * (Gauss3D->X()[3] * covyz - covxz * std::fabs(Gauss3D->X()[1]));
          if (det < 0.) {
            goodData = -4;
            if (internalDebug == true)
              edm::LogInfo("Vx3DHLTAnalyzer") << "\tNegative determinant !";
          }
        }
      } else
        break;
    }

    if (goodData == 0)
      for (unsigned int i = 0; i < nParams; i++) {
        vals->operator[](i) = Gauss3D->X()[i];
        vals->operator[](i + nParams) = Gauss3D->Errors()[i];
      }

    delete Gauss3D;
    return goodData;
  }

  return -1;
}

void Vx3DHLTAnalyzer::reset(string ResetType) {
  if ((debugMode == true) && (outputDebugFile.is_open() == true)) {
    outputDebugFile << "Runnumber " << runNumber << endl;
    outputDebugFile << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " " << (beginTimeOfFit >> 32) << endl;
    outputDebugFile << "BeginLumiRange " << beginLumiOfFit << endl;
    outputDebugFile << "EndTimeOfFit " << formatTime(endTimeOfFit >> 32) << " " << (endTimeOfFit >> 32) << endl;
    outputDebugFile << "EndLumiRange " << endLumiOfFit << endl;
    outputDebugFile << "LumiCounter " << lumiCounter << endl;
    outputDebugFile << "LastLumiOfFit " << lastLumiOfFit << endl;
  }

  if (ResetType == "scratch") {
    runNumber = 0;
    numberGoodFits = 0;
    numberFits = 0;
    lastLumiOfFit = 0;

    Vx_X->Reset();
    Vx_Y->Reset();
    Vx_Z->Reset();

    Vx_X_Fit->Reset();
    Vx_Y_Fit->Reset();
    Vx_Z_Fit->Reset();

    Vx_ZX->Reset();
    Vx_ZY->Reset();
    Vx_XY->Reset();

    Vx_X_Cum->Reset();
    Vx_Y_Cum->Reset();
    Vx_Z_Cum->Reset();

    Vx_ZX_Cum->Reset();
    Vx_ZY_Cum->Reset();
    Vx_XY_Cum->Reset();

    mXlumi->Reset();
    mYlumi->Reset();
    mZlumi->Reset();

    sXlumi->Reset();
    sYlumi->Reset();
    sZlumi->Reset();

    dxdzlumi->Reset();
    dydzlumi->Reset();

    hitCounter->Reset();
    goodVxCounter->Reset();
    statusCounter->Reset();
    fitResults->Reset();

    reportSummary->Fill(-1);
    reportSummaryMap->getTH1()->SetBinContent(1, 1, -1);

    Vertices.clear();

    lumiCounter = 0;
    totalHits = 0;
    beginTimeOfFit = 0;
    endTimeOfFit = 0;
    beginLumiOfFit = 0;
    endLumiOfFit = 0;

    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tReset issued: scratch";
    if ((debugMode == true) && (outputDebugFile.is_open() == true))
      outputDebugFile << "Reset -scratch- issued\n" << endl;
  } else if (ResetType == "whole") {
    Vx_X->Reset();
    Vx_Y->Reset();
    Vx_Z->Reset();

    Vx_ZX->Reset();
    Vx_ZY->Reset();
    Vx_XY->Reset();

    Vertices.clear();

    lumiCounter = 0;
    totalHits = 0;
    beginTimeOfFit = 0;
    endTimeOfFit = 0;
    beginLumiOfFit = 0;
    endLumiOfFit = 0;

    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tReset issued: whole";
    if ((debugMode == true) && (outputDebugFile.is_open() == true))
      outputDebugFile << "Reset -whole- issued\n" << endl;
  } else if (ResetType == "fit") {
    Vx_X_Fit->Reset();
    Vx_Y_Fit->Reset();
    Vx_Z_Fit->Reset();

    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tReset issued: fit";
    if ((debugMode == true) && (outputDebugFile.is_open() == true))
      outputDebugFile << "Reset -fit- issued\n" << endl;
  } else if (ResetType == "hitCounter") {
    totalHits = 0;

    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tReset issued: hitCounter";
    if ((debugMode == true) && (outputDebugFile.is_open() == true))
      outputDebugFile << "Reset -hitCounter- issued\n" << endl;
  }
}

void Vx3DHLTAnalyzer::writeToFile(vector<double>* vals,
                                  TimeValue_t BeginTimeOfFit,
                                  TimeValue_t EndTimeOfFit,
                                  unsigned int BeginLumiOfFit,
                                  unsigned int EndLumiOfFit,
                                  int dataType) {
  stringstream BufferString;
  BufferString.precision(5);

  outputFile.open(fileName.c_str(), ios::out);

  if ((outputFile.is_open() == true) && (vals != nullptr) && (vals->size() == (nParams - 1) * 2)) {
    vector<double>::const_iterator it = vals->begin();

    outputFile << "Runnumber " << runNumber << endl;
    outputFile << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " " << (beginTimeOfFit >> 32) << endl;
    outputFile << "EndTimeOfFit " << formatTime(endTimeOfFit >> 32) << " " << (endTimeOfFit >> 32) << endl;
    outputFile << "LumiRange " << beginLumiOfFit << " - " << endLumiOfFit << endl;
    outputFile << "Type " << dataType << endl;
    // 3D Vertexing with Pixel Tracks:
    // Good data = Type  3
    // Bad data  = Type -1

    BufferString << *(it + 0);
    outputFile << "X0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 1);
    outputFile << "Y0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 2);
    outputFile << "Z0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 5);
    outputFile << "sigmaZ0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 6);
    outputFile << "dxdz " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 7);
    outputFile << "dydz " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 3);
    outputFile << "BeamWidthX " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 4);
    outputFile << "BeamWidthY " << BufferString.str().c_str() << endl;
    BufferString.str("");

    outputFile << "Cov(0,j) " << *(it + 8) << " 0 0 0 0 0 0" << endl;
    outputFile << "Cov(1,j) 0 " << *(it + 9) << " 0 0 0 0 0" << endl;
    outputFile << "Cov(2,j) 0 0 " << *(it + 10) << " 0 0 0 0" << endl;
    outputFile << "Cov(3,j) 0 0 0 " << *(it + 13) << " 0 0 0" << endl;
    outputFile << "Cov(4,j) 0 0 0 0 " << *(it + 14) << " 0 0" << endl;
    outputFile << "Cov(5,j) 0 0 0 0 0 " << *(it + 15) << " 0" << endl;
    outputFile << "Cov(6,j) 0 0 0 0 0 0 "
               << ((*(it + 11)) + (*(it + 12)) + 2. * std::sqrt((*(it + 11)) * (*(it + 12)))) / 4. << endl;

    outputFile << "EmittanceX 0" << endl;
    outputFile << "EmittanceY 0" << endl;
    outputFile << "BetaStar 0" << endl;
  }
  outputFile.close();

  if ((debugMode == true) && (outputDebugFile.is_open() == true) && (vals != nullptr) &&
      (vals->size() == (nParams - 1) * 2)) {
    vector<double>::const_iterator it = vals->begin();

    outputDebugFile << "Runnumber " << runNumber << endl;
    outputDebugFile << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " " << (beginTimeOfFit >> 32) << endl;
    outputDebugFile << "EndTimeOfFit " << formatTime(endTimeOfFit >> 32) << " " << (endTimeOfFit >> 32) << endl;
    outputDebugFile << "LumiRange " << beginLumiOfFit << " - " << endLumiOfFit << endl;
    outputDebugFile << "Type " << dataType << endl;
    // 3D Vertexing with Pixel Tracks:
    // Good data = Type  3
    // Bad data  = Type -1

    BufferString << *(it + 0);
    outputDebugFile << "X0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 1);
    outputDebugFile << "Y0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 2);
    outputDebugFile << "Z0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 5);
    outputDebugFile << "sigmaZ0 " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 6);
    outputDebugFile << "dxdz " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 7);
    outputDebugFile << "dydz " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 3);
    outputDebugFile << "BeamWidthX " << BufferString.str().c_str() << endl;
    BufferString.str("");

    BufferString << *(it + 4);
    outputDebugFile << "BeamWidthY " << BufferString.str().c_str() << endl;
    BufferString.str("");

    outputDebugFile << "Cov(0,j) " << *(it + 8) << " 0 0 0 0 0 0" << endl;
    outputDebugFile << "Cov(1,j) 0 " << *(it + 9) << " 0 0 0 0 0" << endl;
    outputDebugFile << "Cov(2,j) 0 0 " << *(it + 10) << " 0 0 0 0" << endl;
    outputDebugFile << "Cov(3,j) 0 0 0 " << *(it + 13) << " 0 0 0" << endl;
    outputDebugFile << "Cov(4,j) 0 0 0 0 " << *(it + 14) << " 0 0" << endl;
    outputDebugFile << "Cov(5,j) 0 0 0 0 0 " << *(it + 15) << " 0" << endl;
    outputDebugFile << "Cov(6,j) 0 0 0 0 0 0 "
                    << ((*(it + 11)) + (*(it + 12)) + 2. * std::sqrt((*(it + 11)) * (*(it + 12)))) / 4. << endl;

    outputDebugFile << "EmittanceX 0" << endl;
    outputDebugFile << "EmittanceY 0" << endl;
    outputDebugFile << "BetaStar 0" << endl;

    outputDebugFile << "\nUsed vertices: " << counterVx << "\n" << endl;
  }
}

void Vx3DHLTAnalyzer::printFitParams(const vector<double>& fitResults) {
  edm::LogInfo("Vx3DHLTAnalyzer") << "var x -->  " << fitResults[0] << " +/- " << fitResults[0 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "var y -->  " << fitResults[1] << " +/- " << fitResults[1 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "var z -->  " << fitResults[2] << " +/- " << fitResults[2 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "cov xy --> " << fitResults[3] << " +/- " << fitResults[3 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "dydz   --> " << fitResults[4] << " +/- " << fitResults[4 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "dxdz   --> " << fitResults[5] << " +/- " << fitResults[5 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "mean x --> " << fitResults[6] << " +/- " << fitResults[6 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "mean y --> " << fitResults[7] << " +/- " << fitResults[7 + nParams];
  edm::LogInfo("Vx3DHLTAnalyzer") << "mean z --> " << fitResults[8] << " +/- " << fitResults[8 + nParams];
}

void Vx3DHLTAnalyzer::dqmBeginLuminosityBlock(const LuminosityBlock& lumiBlock, const EventSetup& iSetup) {
  // @@@ If statement to avoid problems with non-sequential lumisections @@@
  if ((lumiCounter == 0) && (lumiBlock.luminosityBlock() > lastLumiOfFit)) {
    beginTimeOfFit = lumiBlock.beginTime().value();
    beginLumiOfFit = lumiBlock.luminosityBlock();
    lumiCounter++;
  } else if ((lumiCounter != 0) && (lumiBlock.luminosityBlock() >= (beginLumiOfFit + lumiCounter)))
    lumiCounter++;
  else
    reset("scratch");
}

void Vx3DHLTAnalyzer::dqmEndLuminosityBlock(const LuminosityBlock& lumiBlock, const EventSetup& iSetup) {
  stringstream histTitle;
  double minXfit, maxXfit;
  int goodData;

  if ((nLumiFit != 0) && (lumiCounter % nLumiFit == 0) && (beginTimeOfFit != 0) && (runNumber != 0)) {
    endTimeOfFit = lumiBlock.endTime().value();
    endLumiOfFit = lumiBlock.luminosityBlock();
    lastLumiOfFit = endLumiOfFit;
    vector<double> vals;

    hitCounter->getTH1()->SetBinContent(lastLumiOfFit, (double)totalHits);
    hitCounter->getTH1()->SetBinError(
        lastLumiOfFit,
        (totalHits != 0 ? 1.
                        : 0.));  // It's not sqrt(n) because we want to weight all entries in the same way for the fit

    if (dataFromFit == true) {
      vector<double> fitResults;

      fitResults.push_back(Vx_X->getTH1()->GetRMS() * Vx_X->getTH1()->GetRMS());
      fitResults.push_back(Vx_Y->getTH1()->GetRMS() * Vx_Y->getTH1()->GetRMS());
      fitResults.push_back(Vx_Z->getTH1()->GetRMS() * Vx_Z->getTH1()->GetRMS());
      fitResults.push_back(0.0);
      fitResults.push_back(0.0);
      fitResults.push_back(0.0);
      fitResults.push_back(Vx_X->getTH1()->GetMean());
      fitResults.push_back(Vx_Y->getTH1()->GetMean());
      fitResults.push_back(Vx_Z->getTH1()->GetMean());
      for (unsigned int i = 0; i < nParams; i++)
        fitResults.push_back(0.0);

      if (internalDebug == true) {
        edm::LogInfo("Vx3DHLTAnalyzer") << "\t@@@ Beam Spot parameters - prefit @@@";

        printFitParams(fitResults);

        edm::LogInfo("Vx3DHLTAnalyzer") << "Runnumber " << runNumber;
        edm::LogInfo("Vx3DHLTAnalyzer") << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " "
                                        << (beginTimeOfFit >> 32);
        edm::LogInfo("Vx3DHLTAnalyzer") << "EndTimeOfFit " << formatTime(endTimeOfFit >> 32) << " "
                                        << (endTimeOfFit >> 32);
        edm::LogInfo("Vx3DHLTAnalyzer") << "LumiRange " << beginLumiOfFit << " - " << endLumiOfFit;
      }

      goodData = MyFit(&fitResults);

      if (internalDebug == true) {
        edm::LogInfo("Vx3DHLTAnalyzer") << "\t@@@ Beam Spot parameters - postfit @@@";

        printFitParams(fitResults);

        edm::LogInfo("Vx3DHLTAnalyzer") << "goodData --> " << goodData;
        edm::LogInfo("Vx3DHLTAnalyzer") << "Used vertices --> " << counterVx;
      }

      if (goodData == 0) {
        vals.push_back(fitResults[6]);
        vals.push_back(fitResults[7]);
        vals.push_back(fitResults[8]);
        vals.push_back(std::sqrt(std::fabs(fitResults[0])));
        vals.push_back(std::sqrt(std::fabs(fitResults[1])));
        vals.push_back(std::sqrt(std::fabs(fitResults[2])));
        vals.push_back(fitResults[5]);
        vals.push_back(fitResults[4]);

        vals.push_back(std::pow(fitResults[6 + nParams], 2.));
        vals.push_back(std::pow(fitResults[7 + nParams], 2.));
        vals.push_back(std::pow(fitResults[8 + nParams], 2.));
        vals.push_back(std::pow(std::fabs(fitResults[0 + nParams]) / (2. * std::sqrt(std::fabs(fitResults[0]))), 2.));
        vals.push_back(std::pow(std::fabs(fitResults[1 + nParams]) / (2. * std::sqrt(std::fabs(fitResults[1]))), 2.));
        vals.push_back(std::pow(std::fabs(fitResults[2 + nParams]) / (2. * std::sqrt(std::fabs(fitResults[2]))), 2.));
        vals.push_back(std::pow(fitResults[5 + nParams], 2.));
        vals.push_back(std::pow(fitResults[4 + nParams], 2.));
      } else
        for (unsigned int i = 0; i < (nParams - 1) * 2; i++)
          vals.push_back(0.0);

      fitResults.clear();
    } else {
      counterVx = Vx_X->getTH1F()->GetEntries();

      if (Vx_X->getTH1F()->GetEntries() >= minNentries) {
        goodData = 0;

        vals.push_back(Vx_X->getTH1F()->GetMean());
        vals.push_back(Vx_Y->getTH1F()->GetMean());
        vals.push_back(Vx_Z->getTH1F()->GetMean());
        vals.push_back(Vx_X->getTH1F()->GetRMS());
        vals.push_back(Vx_Y->getTH1F()->GetRMS());
        vals.push_back(Vx_Z->getTH1F()->GetRMS());
        vals.push_back(0.0);
        vals.push_back(0.0);

        vals.push_back(std::pow(Vx_X->getTH1F()->GetMeanError(), 2.));
        vals.push_back(std::pow(Vx_Y->getTH1F()->GetMeanError(), 2.));
        vals.push_back(std::pow(Vx_Z->getTH1F()->GetMeanError(), 2.));
        vals.push_back(std::pow(Vx_X->getTH1F()->GetRMSError(), 2.));
        vals.push_back(std::pow(Vx_Y->getTH1F()->GetRMSError(), 2.));
        vals.push_back(std::pow(Vx_Z->getTH1F()->GetRMSError(), 2.));
        vals.push_back(0.0);
        vals.push_back(0.0);
      } else {
        goodData = -2;
        for (unsigned int i = 0; i < (nParams - 1) * 2; i++)
          vals.push_back(0.0);
      }
    }

    // vals[0]  = X0
    // vals[1]  = Y0
    // vals[2]  = Z0
    // vals[3]  = sigmaX0
    // vals[4]  = sigmaY0
    // vals[5]  = sigmaZ0
    // vals[6]  = dxdz
    // vals[7]  = dydz

    // vals[8]  = err^2 X0
    // vals[9]  = err^2 Y0
    // vals[10] = err^2 Z0
    // vals[11] = err^2 sigmaX0
    // vals[12] = err^2 sigmaY0
    // vals[13] = err^2 sigmaZ0
    // vals[14] = err^2 dxdz
    // vals[15] = err^2 dydz

    numberFits++;
    writeToFile(&vals, beginTimeOfFit, endTimeOfFit, beginLumiOfFit, endLumiOfFit, 3);
    if (internalDebug == true)
      edm::LogInfo("Vx3DHLTAnalyzer") << "\tUsed vertices: " << counterVx;

    statusCounter->getTH1()->SetBinContent(lastLumiOfFit, (double)goodData);
    statusCounter->getTH1()->SetBinError(lastLumiOfFit, 1e-3);

    // Copy vertex position histograms into to-fit histograms
    if (goodData == 0)
      reset("fit");
    else if (lumiCounter >= maxLumiIntegration) {
      reset("fit");
      reset("whole");
    }

    for (int i = 0; i < Vx_X_Fit->getTH1()->GetNbinsX(); i++) {
      Vx_X_Fit->getTH1()->SetBinContent(
          i + 1, Vx_X_Fit->getTH1()->GetBinContent(i + 1) + Vx_X->getTH1()->GetBinContent(i + 1));
      Vx_X_Fit->getTH1()->SetBinError(i + 1, sqrt(Vx_X_Fit->getTH1()->GetBinContent(i + 1)));
    }

    for (int i = 0; i < Vx_Y_Fit->getTH1()->GetNbinsX(); i++) {
      Vx_Y_Fit->getTH1()->SetBinContent(
          i + 1, Vx_Y_Fit->getTH1()->GetBinContent(i + 1) + Vx_Y->getTH1()->GetBinContent(i + 1));
      Vx_Y_Fit->getTH1()->SetBinError(i + 1, sqrt(Vx_Y_Fit->getTH1()->GetBinContent(i + 1)));
    }

    for (int i = 0; i < Vx_Z_Fit->getTH1()->GetNbinsX(); i++) {
      Vx_Z_Fit->getTH1()->SetBinContent(
          i + 1, Vx_Z_Fit->getTH1()->GetBinContent(i + 1) + Vx_Z->getTH1()->GetBinContent(i + 1));
      Vx_Z_Fit->getTH1()->SetBinError(i + 1, sqrt(Vx_Z_Fit->getTH1()->GetBinContent(i + 1)));
    }

    // Check data quality
    if (goodData == 0) {
      numberGoodFits++;

      histTitle << "Ongoing: fitted lumis " << beginLumiOfFit << " - " << endLumiOfFit;
      reset("whole");
    } else {
      if (goodData == -2)
        histTitle << "Ongoing: not enough evts (" << lumiCounter << " - " << maxLumiIntegration << " lumis)";
      else
        histTitle << "Ongoing: temporary problems (" << lumiCounter << " - " << maxLumiIntegration << " lumis)";

      if (lumiCounter >= maxLumiIntegration) {
        statusCounter->getTH1()->SetBinContent(lastLumiOfFit, -5);
        statusCounter->getTH1()->SetBinError(lastLumiOfFit, 1e-3);
      } else
        reset("hitCounter");
    }

    reportSummary->Fill((numberFits != 0 ? ((double)numberGoodFits) / ((double)numberFits) : -1));
    reportSummaryMap->getTH1()->SetBinContent(
        1, 1, (numberFits != 0 ? ((double)numberGoodFits) / ((double)numberFits) : -1));

    fitResults->setAxisTitle(histTitle.str(), 1);

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
    TF1* myLinFit = new TF1(
        "myLinFit", "[0] + [1]*x", mXlumi->getTH1()->GetXaxis()->GetXmin(), mXlumi->getTH1()->GetXaxis()->GetXmax());
    myLinFit->SetLineColor(2);
    myLinFit->SetLineWidth(2);
    myLinFit->SetParName(0, "Inter.");
    myLinFit->SetParName(1, "Slope");

    mXlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[0]);
    mXlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[8]));
    myLinFit->SetParameter(0, mXlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    mXlumi->getTH1()->Fit(myLinFit, "QR");

    mYlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[1]);
    mYlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[9]));
    myLinFit->SetParameter(0, mYlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    mYlumi->getTH1()->Fit(myLinFit, "QR");

    mZlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[2]);
    mZlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[10]));
    myLinFit->SetParameter(0, mZlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    mZlumi->getTH1()->Fit(myLinFit, "QR");

    sXlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[3]);
    sXlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[11]));
    myLinFit->SetParameter(0, sXlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    sXlumi->getTH1()->Fit(myLinFit, "QR");

    sYlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[4]);
    sYlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[12]));
    myLinFit->SetParameter(0, sYlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    sYlumi->getTH1()->Fit(myLinFit, "QR");

    sZlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[5]);
    sZlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[13]));
    myLinFit->SetParameter(0, sZlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    sZlumi->getTH1()->Fit(myLinFit, "QR");

    dxdzlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[6]);
    dxdzlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[14]));
    myLinFit->SetParameter(0, dxdzlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    dxdzlumi->getTH1()->Fit(myLinFit, "QR");

    dydzlumi->getTH1()->SetBinContent(lastLumiOfFit, vals[7]);
    dydzlumi->getTH1()->SetBinError(lastLumiOfFit, std::sqrt(vals[15]));
    myLinFit->SetParameter(0, dydzlumi->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    dydzlumi->getTH1()->Fit(myLinFit, "QR");

    myLinFit->SetParameter(0, hitCounter->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    hitCounter->getTH1()->Fit(myLinFit, "QR");

    goodVxCounter->getTH1()->SetBinContent(lastLumiOfFit, (double)counterVx);
    goodVxCounter->getTH1()->SetBinError(
        lastLumiOfFit,
        (counterVx != 0 ? 1.
                        : 0.));  // It's not sqrt(n) because we want to weight all entries in the same way for the fit
    myLinFit->SetParameter(0, goodVxCounter->getTH1()->GetMean(2));
    myLinFit->SetParameter(1, 0.0);
    goodVxCounter->getTH1()->Fit(myLinFit, "QR");

    delete myLinFit;
    vals.clear();

    // Gaussian fit to 1D vertex coordinate distributions
    TF1* myGaussFit = new TF1("myGaussFit",
                              "[0]*exp(-(x-[1])*(x-[1])/(2*[2]*[2]))",
                              Vx_Z_Fit->getTH1()->GetXaxis()->GetXmin(),
                              Vx_Z_Fit->getTH1()->GetXaxis()->GetXmax());
    myGaussFit->SetLineColor(2);
    myGaussFit->SetLineWidth(2);
    myGaussFit->SetParName(0, "Ampl.");
    myGaussFit->SetParName(1, "#mu");
    myGaussFit->SetParName(2, "#sigma");

    myGaussFit->SetParameter(0, Vx_X_Fit->getTH1()->GetMaximum());
    myGaussFit->SetParameter(1, Vx_X_Fit->getTH1()->GetMean());
    myGaussFit->SetParameter(2, Vx_X_Fit->getTH1()->GetRMS());
    minXfit = Vx_X_Fit->getTH1()->GetBinLowEdge(1);
    for (int i = 0; i < Vx_X_Fit->getTH1()->GetNbinsX(); i++) {
      if (Vx_X_Fit->getTH1()->GetBinContent(i + 1) > 0) {
        minXfit = Vx_X_Fit->getTH1()->GetBinLowEdge(i + 1);
        break;
      }
    }
    maxXfit = Vx_X_Fit->getTH1()->GetBinLowEdge(Vx_X_Fit->getTH1()->GetNbinsX());
    for (int i = Vx_X_Fit->getTH1()->GetNbinsX(); i > 0; i--) {
      if (Vx_X_Fit->getTH1()->GetBinContent(i) > 0) {
        maxXfit = Vx_X_Fit->getTH1()->GetBinLowEdge(i);
        break;
      }
    }
    myGaussFit->SetRange(minXfit - (maxXfit - minXfit) / 2., maxXfit + (maxXfit - minXfit) / 2.);
    if (Vx_X_Fit->getTH1()->GetEntries() > 0)
      Vx_X_Fit->getTH1()->Fit(myGaussFit, "QRL");

    myGaussFit->SetParameter(0, Vx_Y_Fit->getTH1()->GetMaximum());
    myGaussFit->SetParameter(1, Vx_Y_Fit->getTH1()->GetMean());
    myGaussFit->SetParameter(2, Vx_Y_Fit->getTH1()->GetRMS());
    minXfit = Vx_Y_Fit->getTH1()->GetBinLowEdge(1);
    for (int i = 0; i < Vx_Y_Fit->getTH1()->GetNbinsX(); i++) {
      if (Vx_Y_Fit->getTH1()->GetBinContent(i + 1) > 0) {
        minXfit = Vx_Y_Fit->getTH1()->GetBinLowEdge(i + 1);
        break;
      }
    }
    maxXfit = Vx_Y_Fit->getTH1()->GetBinLowEdge(Vx_Y_Fit->getTH1()->GetNbinsX());
    for (int i = Vx_Y_Fit->getTH1()->GetNbinsX(); i > 0; i--) {
      if (Vx_Y_Fit->getTH1()->GetBinContent(i) > 0) {
        maxXfit = Vx_Y_Fit->getTH1()->GetBinLowEdge(i);
        break;
      }
    }
    myGaussFit->SetRange(minXfit - (maxXfit - minXfit) / 2., maxXfit + (maxXfit - minXfit) / 2.);
    if (Vx_Y_Fit->getTH1()->GetEntries() > 0)
      Vx_Y_Fit->getTH1()->Fit(myGaussFit, "QRL");

    myGaussFit->SetParameter(0, Vx_Z_Fit->getTH1()->GetMaximum());
    myGaussFit->SetParameter(1, Vx_Z_Fit->getTH1()->GetMean());
    myGaussFit->SetParameter(2, Vx_Z_Fit->getTH1()->GetRMS());
    minXfit = Vx_Z_Fit->getTH1()->GetBinLowEdge(1);
    for (int i = 0; i < Vx_Z_Fit->getTH1()->GetNbinsX(); i++) {
      if (Vx_Z_Fit->getTH1()->GetBinContent(i + 1) > 0) {
        minXfit = Vx_Z_Fit->getTH1()->GetBinLowEdge(i + 1);
        break;
      }
    }
    maxXfit = Vx_Z_Fit->getTH1()->GetBinLowEdge(Vx_Z_Fit->getTH1()->GetNbinsX());
    for (int i = Vx_Z_Fit->getTH1()->GetNbinsX(); i > 0; i--) {
      if (Vx_Z_Fit->getTH1()->GetBinContent(i) > 0) {
        maxXfit = Vx_Z_Fit->getTH1()->GetBinLowEdge(i);
        break;
      }
    }
    myGaussFit->SetRange(minXfit - (maxXfit - minXfit) / 2., maxXfit + (maxXfit - minXfit) / 2.);
    if (Vx_Z_Fit->getTH1()->GetEntries() > 0)
      Vx_Z_Fit->getTH1()->Fit(myGaussFit, "QRL");

    delete myGaussFit;
  } else if ((nLumiFit != 0) && (lumiCounter % nLumiFit != 0) && (beginTimeOfFit != 0) && (runNumber != 0)) {
    histTitle << "Ongoing: accumulating evts (" << lumiCounter % nLumiFit << " - " << nLumiFit << " in " << lumiCounter
              << " - " << maxLumiIntegration << " lumis)";
    fitResults->setAxisTitle(histTitle.str(), 1);
    if ((debugMode == true) && (outputDebugFile.is_open() == true)) {
      outputDebugFile << "\n"
                      << "Runnumber " << runNumber << endl;
      outputDebugFile << "BeginTimeOfFit " << formatTime(beginTimeOfFit >> 32) << " " << (beginTimeOfFit >> 32) << endl;
      outputDebugFile << "BeginLumiRange " << beginLumiOfFit << endl;
      outputDebugFile << histTitle.str().c_str() << "\n" << endl;
    }
  } else if ((nLumiFit == 0) || (beginTimeOfFit == 0) || (runNumber == 0)) {
    histTitle << "Ongoing: no ongoing fits";
    fitResults->setAxisTitle(histTitle.str(), 1);
    if ((debugMode == true) && (outputDebugFile.is_open() == true))
      outputDebugFile << histTitle.str().c_str() << "\n" << endl;

    endLumiOfFit = lumiBlock.luminosityBlock();

    hitCounter->getTH1()->SetBinContent(endLumiOfFit, (double)totalHits);
    hitCounter->getTH1()->SetBinError(endLumiOfFit, std::sqrt((double)totalHits));

    reset("whole");
  }

  if (internalDebug == true)
    edm::LogInfo("Vx3DHLTAnalyzer") << "::\tHistogram title: " << histTitle.str();
}

void Vx3DHLTAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, Run const& iRun, EventSetup const& /* iSetup */) {
  ibooker.setCurrentFolder("BeamPixel");

  Vx_X = ibooker.book1D(
      "F - vertex x", "Primary Vertex X Distribution", int(rint(xRange / xStep)), -xRange / 2., xRange / 2.);
  Vx_Y = ibooker.book1D(
      "F - vertex y", "Primary Vertex Y Distribution", int(rint(yRange / yStep)), -yRange / 2., yRange / 2.);
  Vx_Z = ibooker.book1D(
      "F - vertex z", "Primary Vertex Z Distribution", int(rint(zRange / zStep)), -zRange / 2., zRange / 2.);
  Vx_X->setAxisTitle("Primary Vertices X [cm]", 1);
  Vx_X->setAxisTitle("Entries [#]", 2);
  Vx_Y->setAxisTitle("Primary Vertices Y [cm]", 1);
  Vx_Y->setAxisTitle("Entries [#]", 2);
  Vx_Z->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_Z->setAxisTitle("Entries [#]", 2);

  Vx_X_Fit = ibooker.book1D("G - vertex x fit",
                            "Primary Vertex X Distribution (For Fit)",
                            int(rint(xRange / xStep)),
                            -xRange / 2.,
                            xRange / 2.);
  Vx_Y_Fit = ibooker.book1D("G - vertex y fit",
                            "Primary Vertex Y Distribution (For Fit)",
                            int(rint(yRange / yStep)),
                            -yRange / 2.,
                            yRange / 2.);
  Vx_Z_Fit = ibooker.book1D("G - vertex z fit",
                            "Primary Vertex Z Distribution (For Fit)",
                            int(rint(zRange / zStep)),
                            -zRange / 2.,
                            zRange / 2.);
  Vx_X_Fit->setAxisTitle("Primary Vertices X [cm]", 1);
  Vx_X_Fit->setAxisTitle("Entries [#]", 2);
  Vx_Y_Fit->setAxisTitle("Primary Vertices Y [cm]", 1);
  Vx_Y_Fit->setAxisTitle("Entries [#]", 2);
  Vx_Z_Fit->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_Z_Fit->setAxisTitle("Entries [#]", 2);

  Vx_X_Cum = ibooker.book1D("I - vertex x cum",
                            "Primary Vertex X Distribution (Cumulative)",
                            int(rint(xRange / xStep)),
                            -xRange / 2.,
                            xRange / 2.);
  Vx_Y_Cum = ibooker.book1D("I - vertex y cum",
                            "Primary Vertex Y Distribution (Cumulative)",
                            int(rint(yRange / yStep)),
                            -yRange / 2.,
                            yRange / 2.);
  Vx_Z_Cum = ibooker.book1D("I - vertex z cum",
                            "Primary Vertex Z Distribution (Cumulative)",
                            int(rint(zRange / zStep)),
                            -zRange / 2.,
                            zRange / 2.);
  Vx_X_Cum->setAxisTitle("Primary Vertices X [cm]", 1);
  Vx_X_Cum->setAxisTitle("Entries [#]", 2);
  Vx_Y_Cum->setAxisTitle("Primary Vertices Y [cm]", 1);
  Vx_Y_Cum->setAxisTitle("Entries [#]", 2);
  Vx_Z_Cum->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_Z_Cum->setAxisTitle("Entries [#]", 2);

  mXlumi = ibooker.book1D(
      "B - muX vs lumi", "#mu_{x} vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  mYlumi = ibooker.book1D(
      "B - muY vs lumi", "#mu_{y} vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  mZlumi = ibooker.book1D(
      "B - muZ vs lumi", "#mu_{z} vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  mXlumi->setAxisTitle("Lumisection [#]", 1);
  mXlumi->setAxisTitle("#mu_{x} [cm]", 2);
  mXlumi->getTH1()->SetOption("E1");
  mYlumi->setAxisTitle("Lumisection [#]", 1);
  mYlumi->setAxisTitle("#mu_{y} [cm]", 2);
  mYlumi->getTH1()->SetOption("E1");
  mZlumi->setAxisTitle("Lumisection [#]", 1);
  mZlumi->setAxisTitle("#mu_{z} [cm]", 2);
  mZlumi->getTH1()->SetOption("E1");

  sXlumi = ibooker.book1D(
      "C - sigmaX vs lumi", "#sigma_{x} vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  sYlumi = ibooker.book1D(
      "C - sigmaY vs lumi", "#sigma_{y} vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  sZlumi = ibooker.book1D(
      "C - sigmaZ vs lumi", "#sigma_{z} vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  sXlumi->setAxisTitle("Lumisection [#]", 1);
  sXlumi->setAxisTitle("#sigma_{x} [cm]", 2);
  sXlumi->getTH1()->SetOption("E1");
  sYlumi->setAxisTitle("Lumisection [#]", 1);
  sYlumi->setAxisTitle("#sigma_{y} [cm]", 2);
  sYlumi->getTH1()->SetOption("E1");
  sZlumi->setAxisTitle("Lumisection [#]", 1);
  sZlumi->setAxisTitle("#sigma_{z} [cm]", 2);
  sZlumi->getTH1()->SetOption("E1");

  dxdzlumi = ibooker.book1D(
      "D - dxdz vs lumi", "dX/dZ vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  dydzlumi = ibooker.book1D(
      "D - dydz vs lumi", "dY/dZ vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  dxdzlumi->setAxisTitle("Lumisection [#]", 1);
  dxdzlumi->setAxisTitle("dX/dZ [rad]", 2);
  dxdzlumi->getTH1()->SetOption("E1");
  dydzlumi->setAxisTitle("Lumisection [#]", 1);
  dydzlumi->setAxisTitle("dY/dZ [rad]", 2);
  dydzlumi->getTH1()->SetOption("E1");

  Vx_ZX = ibooker.book2D("E - vertex zx",
                         "Primary Vertex ZX Distribution",
                         int(rint(zRange / zStep)),
                         -zRange / 2.,
                         zRange / 2.,
                         int(rint(xRange / xStep)),
                         -xRange / 2.,
                         xRange / 2.);
  Vx_ZY = ibooker.book2D("E - vertex zy",
                         "Primary Vertex ZY Distribution",
                         int(rint(zRange / zStep)),
                         -zRange / 2.,
                         zRange / 2.,
                         int(rint(yRange / yStep)),
                         -yRange / 2.,
                         yRange / 2.);
  Vx_XY = ibooker.book2D("E - vertex xy",
                         "Primary Vertex XY Distribution",
                         int(rint(xRange / xStep)),
                         -xRange / 2.,
                         xRange / 2.,
                         int(rint(yRange / yStep)),
                         -yRange / 2.,
                         yRange / 2.);
  Vx_ZX->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_ZX->setAxisTitle("Primary Vertices X [cm]", 2);
  Vx_ZX->setAxisTitle("Entries [#]", 3);
  Vx_ZY->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_ZY->setAxisTitle("Primary Vertices Y [cm]", 2);
  Vx_ZY->setAxisTitle("Entries [#]", 3);
  Vx_XY->setAxisTitle("Primary Vertices X [cm]", 1);
  Vx_XY->setAxisTitle("Primary Vertices Y [cm]", 2);
  Vx_XY->setAxisTitle("Entries [#]", 3);

  Vx_ZX_Cum = ibooker.book2D("H - vertex zx cum",
                             "Primary Vertex ZX Distribution (Cumulative)",
                             int(rint(zRange / zStep)),
                             -zRange / 2.,
                             zRange / 2.,
                             int(rint(xRange / xStep)),
                             -xRange / 2.,
                             xRange / 2.);
  Vx_ZY_Cum = ibooker.book2D("H - vertex zy cum",
                             "Primary Vertex ZY Distribution (Cumulative)",
                             int(rint(zRange / zStep)),
                             -zRange / 2.,
                             zRange / 2.,
                             int(rint(yRange / yStep)),
                             -yRange / 2.,
                             yRange / 2.);
  Vx_XY_Cum = ibooker.book2D("H - vertex xy cum",
                             "Primary Vertex XY Distribution (Cumulative)",
                             int(rint(xRange / xStep)),
                             -xRange / 2.,
                             xRange / 2.,
                             int(rint(yRange / yStep)),
                             -yRange / 2.,
                             yRange / 2.);
  Vx_ZX_Cum->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_ZX_Cum->setAxisTitle("Primary Vertices X [cm]", 2);
  Vx_ZX_Cum->setAxisTitle("Entries [#]", 3);
  Vx_ZY_Cum->setAxisTitle("Primary Vertices Z [cm]", 1);
  Vx_ZY_Cum->setAxisTitle("Primary Vertices Y [cm]", 2);
  Vx_ZY_Cum->setAxisTitle("Entries [#]", 3);
  Vx_XY_Cum->setAxisTitle("Primary Vertices X [cm]", 1);
  Vx_XY_Cum->setAxisTitle("Primary Vertices Y [cm]", 2);
  Vx_XY_Cum->setAxisTitle("Entries [#]", 3);

  hitCounter = ibooker.book1D(
      "J - pixelHits vs lumi", "# Pixel-Hits vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  hitCounter->setAxisTitle("Lumisection [#]", 1);
  hitCounter->setAxisTitle("Pixel-Hits [#]", 2);
  hitCounter->getTH1()->SetOption("E1");

  goodVxCounter = ibooker.book1D("K - good vertices vs lumi",
                                 "# Good vertices vs. Lumisection",
                                 nLumiXaxisRange,
                                 0.5,
                                 ((double)nLumiXaxisRange) + 0.5);
  goodVxCounter->setAxisTitle("Lumisection [#]", 1);
  goodVxCounter->setAxisTitle("Good vertices [#]", 2);
  goodVxCounter->getTH1()->SetOption("E1");

  statusCounter = ibooker.book1D(
      "L - status vs lumi", "App. Status vs. Lumisection", nLumiXaxisRange, 0.5, ((double)nLumiXaxisRange) + 0.5);
  statusCounter->setAxisTitle("Lumisection [#]", 1);
  statusCounter->getTH1()->SetOption("E1");
  statusCounter->getTH1()->GetYaxis()->Set(11, -5.5, 5.5);
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(1, "Max Lumi.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(2, "Neg. det.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(3, "Infinite err.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(4, "No vtx.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(5, "Infinite EDM");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(6, "OK");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(7, "MINUIT stat.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(8, "MINUIT stat.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(9, "MINUIT stat.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(10, "MINUIT stat.");
  statusCounter->getTH1()->GetYaxis()->SetBinLabel(11, "MINUIT stat.");

  fitResults = ibooker.book2D("A - fit results", "Results of Beam Spot Fit", 2, 0., 2., 9, 0., 9.);
  fitResults->setAxisTitle("Ongoing: bootstrapping", 1);
  fitResults->setBinLabel(9, "X[cm]", 2);
  fitResults->setBinLabel(8, "Y[cm]", 2);
  fitResults->setBinLabel(7, "Z[cm]", 2);
  fitResults->setBinLabel(6, "#sigma_{X}[cm]", 2);
  fitResults->setBinLabel(5, "#sigma_{Y}[cm]", 2);
  fitResults->setBinLabel(4, "#sigma_{Z}[cm]", 2);
  fitResults->setBinLabel(3, "#frac{dX}{dZ}[rad]", 2);
  fitResults->setBinLabel(2, "#frac{dY}{dZ}[rad]", 2);
  fitResults->setBinLabel(1, "Vtx[#]", 2);
  fitResults->setBinLabel(1, "Value", 1);
  fitResults->setBinLabel(2, "Error (stat)", 1);
  fitResults->getTH1()->SetOption("text");

  ibooker.setCurrentFolder("BeamPixel/EventInfo");

  reportSummary = ibooker.bookFloat("reportSummary");
  reportSummary->Fill(-1);
  reportSummaryMap = ibooker.book2D("reportSummaryMap", "Pixel-Vertices Beam Spot: % Good Fits", 1, 0., 1., 1, 0., 1.);
  reportSummaryMap->getTH1()->SetBinContent(1, 1, -1);

  ibooker.setCurrentFolder("BeamPixel/EventInfo/reportSummaryContents");

  // Convention for reportSummary and reportSummaryMap:
  // - -1%  at the moment of creation of the histogram (i.e. white histogram)
  // - n%  numberGoodFits / numberFits

  reset("scratch");  // Initialize histograms after creation
}

// Define this as a plug-in
DEFINE_FWK_MODULE(Vx3DHLTAnalyzer);
