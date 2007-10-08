#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle2_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle2_H

#include <iostream>
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

// temp
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1.h"
#include "TH2.h"
class TFile;

using namespace std;

class StripCPEfromTrackAngle2 : public StripCPE 
{
 public:

  
  //  StripCPEfromTrackAngle2(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom):StripCPE(conf,mag, geom){cout<<"StripCPEfromTrackAngle2 constructor"<<endl;
  StripCPEfromTrackAngle2(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom, const SiStripLorentzAngle* LorentzAngle):StripCPE(conf,mag, geom, LorentzAngle){cout<<"StripCPEfromTrackAngle2 constructor"<<endl;

  /*  StripCPEfromTrackAngle(edm::ParameterSet & conf, 
			 const MagneticField * mag, 
			 const TrackerGeometry* geom, 
			 const SiStripLorentzAngle* LorentzAngle)
    :StripCPE(conf,mag, geom, LorentzAngle ){}
  */
  /*
    myFile = new TFile("debugStripCPEfromTrackAngle2.root","RECREATE");
    myTree = new TTree("HitTree","Tracker Validation tree");
    // GENERAL block
    myTree->Branch("BSubDet", &BSubDet, "BSubDet/I");
    myTree->Branch("Bilay", &Bilay, "Bilay/I");
    myTree->Branch("Bpitch", &Bpitch, "Bpitch/F");
    myTree->Branch("Bthickness", &Bthickness, "Bthickness/F");
    myTree->Branch("BThicknessOnPitch", &BThicknessOnPitch, "BThicknessOnPitch/F");
    myTree->Branch("BWtrack", &BWtrack, "BWtrack/F");
    myTree->Branch("Btanalpha", &Btanalpha, "Btanalpha/F");
    myTree->Branch("BtanalphaL", &BtanalphaL, "BtanalphaL/F");
    myTree->Branch("BclusterWidth", &BclusterWidth, "BclusterWidth/I");
    myTree->Branch("BWexp", &BWexp, "BWexp/I");
    myTree->Branch("Biopt", &Biopt, "Biopt/I");
    myTree->Branch("Buerr", &Buerr, "Buerr/F");
    myTree->Branch("Bdriftx", &Bdriftx, "Bdriftx/F");
    myTree->Branch("Beresultxx", &Beresultxx, "Beresultxx/F");
  */
	//        myTree->Branch("", &, "/F");


};
    
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,const GeomDetUnit& det, const LocalTrajectoryParameters & ltp)const{
    return localParameters(cl,ltp);
  }; 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl, const LocalTrajectoryParameters & ltp)const; 

  
  ~StripCPEfromTrackAngle2()
 {
   cout<<"StripCPEfromTrackAngle2 destructor"<<endl;
   /*   myFile->cd();
    myTree->Write();
    myFile->Close();*/
    //    delete myFile;

 };

  private:

  mutable TFile* myFile;
  mutable TTree* myTree;

  mutable float BWtrack;
  mutable float  Btanalpha;
  mutable float  BtanalphaL;
  mutable int BSubDet;
  mutable int Bilay;
  mutable float Bpitch;
  mutable int BclusterWidth;
  mutable int BWexp;
  mutable int Biopt;
  mutable float Buerr;
  mutable float Bthickness;
  mutable float BThicknessOnPitch;
  mutable float Bdriftx;
  mutable float Beresultxx;

};

#endif




