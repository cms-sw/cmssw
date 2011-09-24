// -*- C++ -*-
//
// Package:    HiEvtPlaneFlattenGen
// Class:      HiEvtPlaneFlattenGen
// 
/**\class HiEvtPlaneFlatten HiEvtPlaneFlatten.cc HiEvtPlaneFlatten/HiEvtPlaneFlatten/src/HiEvtPlaneFlatten.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stephen Sanders
//         Created:  Mon Jun  7 14:40:12 EDT 2010
// $Id: HiEvtPlaneFlattenGen.h,v 1.7 2011/09/15 16:43:56 ssanders Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
//#include "HepMC/GenEvent.h"
//#include "HepMC/GenParticle.h"
//#include "HepMC/GenVertex.h"
//#include "HepMC/HeavyIon.h"
//#include "HepMC/SimpleVector.h"
//#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "Math/Vector3D.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
//#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//#include "SimDataFormats/Track/interface/SimTrack.h"
//#include "SimDataFormats/Track/interface/SimTrackContainer.h"
//#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
//#include "SimDataFormats/Vertex/interface/SimVertex.h"
//#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "TH1D.h"

#include <time.h>
#include <cstdlib>
#include <vector>

#define MAXCUT 5000

using namespace std;
//
// class declaration
//
static const double pi = 3.14159265358979312;
static const double pi2 = 1.57079632679489656;
static const int nvtxbins = 10;
static const double vtxbins[]={-25,-20,-15,-10,-5,0,5,10,15,20,25};
static const int NumCentBins = 12; 
const double  wcent[] = {0,5,10,15,20,30,40,50,60,70,80,90,100}; 

class HiEvtPlaneFlattenGen {
public:
  explicit HiEvtPlaneFlattenGen();
  void Init(int order, int ncentbins,const double *wcent, string tag, int vord);
  int GetCutIndx(double cent, double vtx, int iord);
  void Fill(double psi, double vtx, double cent);
  double GetFlatPsi(double psi, double vtx, double cent);
  ~HiEvtPlaneFlattenGen();
  int GetHBins(){return hbins;}
  double GetX(int bin){return flatX[bin];}
  double GetY(int bin){return flatY[bin];}
  double GetCnt(int bin) {return flatCnt[bin];}
  void SetXDB(int indx, double val) {flatXDB[indx]=val;}
  void SetYDB(int indx, double val) {flatYDB[indx]=val;}
  Double_t bounds(Double_t ang) {
    if(ang<-pi) ang+=2.*pi;
    if(ang>pi)  ang-=2.*pi;
    return ang;
  }
  Double_t bounds2(Double_t ang) {
    double range = TMath::Pi()/(double) vorder;
    if(ang<-range) ang+=2*range;
    if(ang>range)  ang-=2*range;
    return ang;
  }
private:
  double flatX[MAXCUT];
  double flatY[MAXCUT];
  double flatXDB[MAXCUT];
  double flatYDB[MAXCUT];
  double flatCnt[MAXCUT];
  int hOrder;    //flattening order
  int hcentbins; //# of centrality bins
  double hwcent[50]; //width of each centrality bin
  TH1D * hcent;
  TH1D * hvtx;
  int hbins;
  int vorder; //order of flattened event plane
};

