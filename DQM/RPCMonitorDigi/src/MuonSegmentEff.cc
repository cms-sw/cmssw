// -*- C++ -*-
//
// Package:    MuonSegmentEff
// Class:      MuonSegmentEff
// 
/**\class MuonSegmentEff MuonSegmentEff.cc dtcscrpc/MuonSegmentEff/src/MuonSegmentEff.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Carrillo (Uniandes)
//         Created:  Tue Oct  2 16:57:49 CEST 2007
// $Id: MuonSegmentEff.cc,v 1.23 2008/05/21 14:41:28 carrillo Exp $
//
//

#include "DQM/RPCMonitorDigi/interface/MuonSegmentEff.h"

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <cmath>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TString.h"

class DTStationIndex{
public: 
  DTStationIndex():_region(0),_wheel(0),_sector(0),_station(0){}
  DTStationIndex(int region, int wheel, int sector, int station) : 
    _region(region),
    _wheel(wheel),
    _sector(sector),
    _station(station){}
  ~DTStationIndex(){}
  int region() const {return _region;}
  int wheel() const {return _wheel;}
  int sector() const {return _sector;}
  int station() const {return _station;}
  bool operator<(const DTStationIndex& dtind) const{
    if(dtind.region()!=this->region())
      return dtind.region()<this->region();
    else if(dtind.wheel()!=this->wheel())
      return dtind.wheel()<this->wheel();
    else if(dtind.sector()!=this->sector())
      return dtind.sector()<this->sector();
    else if(dtind.station()!=this->station())
      return dtind.station()<this->station();
    return false;
  }
private:
  int _region;
  int _wheel;
  int _sector;
  int _station; 
};


class CSCStationIndex{
public:
  CSCStationIndex():_region(0),_station(0),_ring(0),_chamber(0){}
  CSCStationIndex(int region, int station, int ring, int chamber):
    _region(region),
    _station(station),
    _ring(ring),
    _chamber(chamber){}
  ~CSCStationIndex(){}
  int region() const {return _region;}
  int station() const {return _station;}
  int ring() const {return _ring;}
  int chamber() const {return _chamber;}
  bool operator<(const CSCStationIndex& cscind) const{
    if(cscind.region()!=this->region())
      return cscind.region()<this->region();
    else if(cscind.station()!=this->station())
      return cscind.station()<this->station();
    else if(cscind.ring()!=this->ring())
      return cscind.ring()<this->ring();
    else if(cscind.chamber()!=this->chamber())
      return cscind.chamber()<this->chamber();
    return false;
  }

private:
  int _region;
  int _station;
  int _ring;  
  int _chamber;
};


void MuonSegmentEff::beginJob(const edm::EventSetup&)
{

}




MuonSegmentEff::MuonSegmentEff(const edm::ParameterSet& iConfig)
{

  std::map<RPCDetId, int> buff;
  counter.clear();
  counter.reserve(3);
  counter.push_back(buff);
  counter.push_back(buff);
  counter.push_back(buff);
  totalcounter.clear();
  totalcounter.reserve(3);
  totalcounter[0]=0;
  totalcounter[1]=0;
  totalcounter[2]=0;

  incldt=iConfig.getUntrackedParameter<bool>("incldt",true);
  incldtMB4=iConfig.getUntrackedParameter<bool>("incldtMB4",true);
  inclcsc=iConfig.getUntrackedParameter<bool>("inclcsc",true);
  MinimalResidual= iConfig.getUntrackedParameter<double>("MinimalResidual",2.);
  MinimalResidualRB4=iConfig.getUntrackedParameter<double>("MinimalResidualRB4",4.);
  MinCosAng=iConfig.getUntrackedParameter<double>("MinCosAng",0.9999);
  MaxD=iConfig.getUntrackedParameter<double>("MaxD",20.);
  MaxDrb4=iConfig.getUntrackedParameter<double>("MaxDrb4",30.);
  MaxStripToCountInAverage=iConfig.getUntrackedParameter<double>("MaxStripToCountInAverage",5.);
  MaxStripToCountInAverageRB4=iConfig.getUntrackedParameter<double>("MaxStripToCountInAverageRB4",7.);
  muonRPCDigis=iConfig.getUntrackedParameter<std::string>("muonRPCDigis","muonRPCDigis");
  cscSegments=iConfig.getUntrackedParameter<std::string>("cscSegments","cscSegments");
  dt4DSegments=iConfig.getUntrackedParameter<std::string>("dt4DSegments","dt4DSegments");
  rejected=iConfig.getUntrackedParameter<std::string>("rejected","rejected.txt");
  rollseff=iConfig.getUntrackedParameter<std::string>("rollseff","rollseff.txt");
  GlobalRootLabel= iConfig.getUntrackedParameter<std::string>("GlobalRootFileName","GlobalEfficiencyFromTrack.root");

  std::cout<<rejected<<std::endl;
  std::cout<<rollseff<<std::endl;
  
  ofrej.open(rejected.c_str());
  ofeff.open(rollseff.c_str());
  oftwiki.open("tabletotwiki.txt");

  // Giuseppe
  nameInLog = iConfig.getUntrackedParameter<std::string>("moduleLogName", "RPC_Eff");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", true); 
  EffSaveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EffEventsInterval", 10000); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "CMSRPCEff.root"); 
  //Interface
  dbe = edm::Service<DQMStore>().operator->();
  _idList.clear(); 

  //GLOBAL
  std::cout<<"Booking the Global Histograms"<<std::endl;
  
  fOutputFile  = new TFile(GlobalRootLabel.c_str(), "RECREATE" );

  hGlobalRes = new TH1F("GlobalResiduals","All RPC Residuals",250,-10.,10.);
  hGlobalResLa1 = new TH1F("GlobalResidualsLa1","RPC Residuals Layer 1",250,-10.,10.);
  hGlobalResLa2 = new TH1F("GlobalResidualsLa2","RPC Residuals Layer 2",250,-10.,10.);
  hGlobalResLa3 = new TH1F("GlobalResidualsLa3","RPC Residuals Layer 3",250,-10.,10.);
  hGlobalResLa4 = new TH1F("GlobalResidualsLa4","RPC Residuals Layer 4",250,-10.,10.);
  hGlobalResLa5 = new TH1F("GlobalResidualsLa5","RPC Residuals Layer 5",250,-10.,10.);
  hGlobalResLa6 = new TH1F("GlobalResidualsLa6","RPC Residuals Layer 6",250,-10.,10.);

  hGlobalResClu1La1 = new TH1F("GlobalResidualsClu1La1","RPC Residuals Layer 1 Cluster Size 1",250,-10.,10.);
  hGlobalResClu1La2 = new TH1F("GlobalResidualsClu1La2","RPC Residuals Layer 2 Cluster Size 1",250,-10.,10.);
  hGlobalResClu1La3 = new TH1F("GlobalResidualsClu1La3","RPC Residuals Layer 3 Cluster Size 1",250,-10.,10.);
  hGlobalResClu1La4 = new TH1F("GlobalResidualsClu1La4","RPC Residuals Layer 4 Cluster Size 1",250,-10.,10.);
  hGlobalResClu1La5 = new TH1F("GlobalResidualsClu1La5","RPC Residuals Layer 5 Cluster Size 1",250,-10.,10.);
  hGlobalResClu1La6 = new TH1F("GlobalResidualsClu1La6","RPC Residuals Layer 6 Cluster Size 1",250,-10.,10.);

  hGlobalResClu3La1 = new TH1F("GlobalResidualsClu3La1","RPC Residuals Layer 1 Cluster Size 3",250,-10.,10.);
  hGlobalResClu3La2 = new TH1F("GlobalResidualsClu3La2","RPC Residuals Layer 2 Cluster Size 3",250,-10.,10.);
  hGlobalResClu3La3 = new TH1F("GlobalResidualsClu3La3","RPC Residuals Layer 3 Cluster Size 3",250,-10.,10.);
  hGlobalResClu3La4 = new TH1F("GlobalResidualsClu3La4","RPC Residuals Layer 4 Cluster Size 3",250,-10.,10.);
  hGlobalResClu3La5 = new TH1F("GlobalResidualsClu3La5","RPC Residuals Layer 5 Cluster Size 3",250,-10.,10.);
  hGlobalResClu3La6 = new TH1F("GlobalResidualsClu3La6","RPC Residuals Layer 6 Cluster Size 3",250,-10.,10.);

  hGlobalResClu1 = new TH1F("GlobalResidualsClu1RB1in","Global RPC Residuals 1 RB1in",250,-10.,10.);
  hGlobalResClu2 = new TH1F("GlobalResidualsClu2RB1in","Global RPC Residuals 2 RB1in",250,-10.,10.);
  hGlobalResClu3 = new TH1F("GlobalResidualsClu3RB1in","Global RPC Residuals 3 RB1in",250,-10.,10.);
  hGlobalResClu4 = new TH1F("GlobalResidualsClu4RB1in","Global RPC Residuals 4 RB1in",250,-10.,10.);


  hGlobalResY = new TH1F("GlobalResidualsY","Global RPC Residuals Y",500,-100.,100);
  
  //wheel-2
  EffGlobWm2 = new TH1F("GlobEfficiencyWheel_-2","Global Efficiency Wheel -2",240,0.5,240.5);
  EffGlobm2s1 = new TH1F("GlobEfficiencyWheel_m2_Sec1","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s2 = new TH1F("GlobEfficiencyWheel_m2_Sec2","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s3 = new TH1F("GlobEfficiencyWheel_m2_Sec3","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s4 = new TH1F("GlobEfficiencyWheel_m2_Sec4","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s5 = new TH1F("GlobEfficiencyWheel_m2_Sec5","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s6 = new TH1F("GlobEfficiencyWheel_m2_Sec6","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s7 = new TH1F("GlobEfficiencyWheel_m2_Sec7","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s8 = new TH1F("GlobEfficiencyWheel_m2_Sec8","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s9 = new TH1F("GlobEfficiencyWheel_m2_Sec9","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s10 = new TH1F("GlobEfficiencyWheel_m2_Sec10","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s11 = new TH1F("GlobEfficiencyWheel_m2_Sec11","Eff. vs. roll",20,0.5,20.5);
  EffGlobm2s12 = new TH1F("GlobEfficiencyWheel_m2_Sec12","Eff. vs. roll",20,0.5,20.5);

  //wheel-1
  EffGlobWm1 = new TH1F("GlobEfficiencyWheel_-1","Global Efficiency Wheel -1",240,0.5,240.5);
  EffGlobm1s1 = new TH1F("GlobEfficiencyWheel_m1_Sec1","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s2 = new TH1F("GlobEfficiencyWheel_m1_Sec2","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s3 = new TH1F("GlobEfficiencyWheel_m1_Sec3","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s4 = new TH1F("GlobEfficiencyWheel_m1_Sec4","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s5 = new TH1F("GlobEfficiencyWheel_m1_Sec5","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s6 = new TH1F("GlobEfficiencyWheel_m1_Sec6","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s7 = new TH1F("GlobEfficiencyWheel_m1_Sec7","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s8 = new TH1F("GlobEfficiencyWheel_m1_Sec8","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s9 = new TH1F("GlobEfficiencyWheel_m1_Sec9","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s10 = new TH1F("GlobEfficiencyWheel_m1_Sec10","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s11 = new TH1F("GlobEfficiencyWheel_m1_Sec11","Eff. vs. roll",20,0.5,20.5);
  EffGlobm1s12 = new TH1F("GlobEfficiencyWheel_m1_Sec12","Eff. vs. roll",20,0.5,20.5);
  
  //wheel0
  EffGlobW0 = new TH1F("GlobEfficiencyWheel_0","Global Efficiency Wheel 0",240,0.5,240.5);
  EffGlob1 = new TH1F("GlobEfficiencyWheel_0_Sec1","Eff. vs. roll",20,0.5,20.5);
  EffGlob2 = new TH1F("GlobEfficiencyWheel_0_Sec2","Eff. vs. roll",20,0.5,20.5);
  EffGlob3 = new TH1F("GlobEfficiencyWheel_0_Sec3","Eff. vs. roll",20,0.5,20.5);
  EffGlob4 = new TH1F("GlobEfficiencyWheel_0_Sec4","Eff. vs. roll",20,0.5,20.5);
  EffGlob5 = new TH1F("GlobEfficiencyWheel_0_Sec5","Eff. vs. roll",20,0.5,20.5);
  EffGlob6 = new TH1F("GlobEfficiencyWheel_0_Sec6","Eff. vs. roll",20,0.5,20.5);
  EffGlob7 = new TH1F("GlobEfficiencyWheel_0_Sec7","Eff. vs. roll",20,0.5,20.5);
  EffGlob8 = new TH1F("GlobEfficiencyWheel_0_Sec8","Eff. vs. roll",20,0.5,20.5);
  EffGlob9 = new TH1F("GlobEfficiencyWheel_0_Sec9","Eff. vs. roll",20,0.5,20.5);
  EffGlob10 = new TH1F("GlobEfficiencyWheel_0_Sec10","Eff. vs. roll",20,0.5,20.5);
  EffGlob11 = new TH1F("GlobEfficiencyWheel_0_Sec11","Eff. vs. roll",20,0.5,20.5);
  EffGlob12 = new TH1F("GlobEfficiencyWheel_0_Sec12","Eff. vs. roll",20,0.5,20.5);

  //wheel1
  EffGlobW1 = new TH1F("GlobEfficiencyWheel_1","Global Efficiency Wheel 1",240,0.5,240.5);
  EffGlob1s1 = new TH1F("GlobEfficiencyWheel_1_Sec1","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s2 = new TH1F("GlobEfficiencyWheel_1_Sec2","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s3 = new TH1F("GlobEfficiencyWheel_1_Sec3","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s4 = new TH1F("GlobEfficiencyWheel_1_Sec4","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s5 = new TH1F("GlobEfficiencyWheel_1_Sec5","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s6 = new TH1F("GlobEfficiencyWheel_1_Sec6","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s7 = new TH1F("GlobEfficiencyWheel_1_Sec7","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s8 = new TH1F("GlobEfficiencyWheel_1_Sec8","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s9 = new TH1F("GlobEfficiencyWheel_1_Sec9","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s10 = new TH1F("GlobEfficiencyWheel_1_Sec10","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s11 = new TH1F("GlobEfficiencyWheel_1_Sec11","Eff. vs. roll",20,0.5,20.5);
  EffGlob1s12 = new TH1F("GlobEfficiencyWheel_1_Sec12","Eff. vs. roll",20,0.5,20.5);
  
  //wheel2
  EffGlobW2 = new TH1F("GlobEfficiencyWheel_2","Global Efficiency Wheel 2",240,0.5,240.5);
  EffGlob2s1 = new TH1F("GlobEfficiencyWheel_2_Sec1","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s2 = new TH1F("GlobEfficiencyWheel_2_Sec2","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s3 = new TH1F("GlobEfficiencyWheel_2_Sec3","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s4 = new TH1F("GlobEfficiencyWheel_2_Sec4","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s5 = new TH1F("GlobEfficiencyWheel_2_Sec5","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s6 = new TH1F("GlobEfficiencyWheel_2_Sec6","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s7 = new TH1F("GlobEfficiencyWheel_2_Sec7","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s8 = new TH1F("GlobEfficiencyWheel_2_Sec8","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s9 = new TH1F("GlobEfficiencyWheel_2_Sec9","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s10 = new TH1F("GlobEfficiencyWheel_2_Sec10","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s11 = new TH1F("GlobEfficiencyWheel_2_Sec11","Eff. vs. roll",20,0.5,20.5);
  EffGlob2s12 = new TH1F("GlobEfficiencyWheel_2_Sec12","Eff. vs. roll",20,0.5,20.5);
  
}


MuonSegmentEff::~MuonSegmentEff()
{
  std::cout<<"Begin Destructor "<<std::endl;
  // effres->close();
  //delete effres;
  
  fOutputFile->WriteTObject(hGlobalRes);

  fOutputFile->WriteTObject(hGlobalResLa1);
  fOutputFile->WriteTObject(hGlobalResLa2);
  fOutputFile->WriteTObject(hGlobalResLa3);
  fOutputFile->WriteTObject(hGlobalResLa4);
  fOutputFile->WriteTObject(hGlobalResLa5);
  fOutputFile->WriteTObject(hGlobalResLa6);
  

  fOutputFile->WriteTObject(hGlobalResClu1);
  fOutputFile->WriteTObject(hGlobalResClu2);
  fOutputFile->WriteTObject(hGlobalResClu3);
  fOutputFile->WriteTObject(hGlobalResClu4);
  
  fOutputFile->WriteTObject(hGlobalResY);

  fOutputFile->WriteTObject(hGlobalResClu1La1);
  fOutputFile->WriteTObject(hGlobalResClu1La2);
  fOutputFile->WriteTObject(hGlobalResClu1La3);
  fOutputFile->WriteTObject(hGlobalResClu1La4);
  fOutputFile->WriteTObject(hGlobalResClu1La5);
  fOutputFile->WriteTObject(hGlobalResClu1La6);

  fOutputFile->WriteTObject(hGlobalResClu3La1);
  fOutputFile->WriteTObject(hGlobalResClu3La2);
  fOutputFile->WriteTObject(hGlobalResClu3La3);
  fOutputFile->WriteTObject(hGlobalResClu3La4);
  fOutputFile->WriteTObject(hGlobalResClu3La5);
  fOutputFile->WriteTObject(hGlobalResClu3La6);
  

  //wheel-2
  fOutputFile->WriteTObject(EffGlobWm2);
  fOutputFile->WriteTObject(EffGlobm2s1);
  fOutputFile->WriteTObject(EffGlobm2s2);
  fOutputFile->WriteTObject(EffGlobm2s3);
  fOutputFile->WriteTObject(EffGlobm2s4);
  fOutputFile->WriteTObject(EffGlobm2s5);
  fOutputFile->WriteTObject(EffGlobm2s6);
  fOutputFile->WriteTObject(EffGlobm2s7);
  fOutputFile->WriteTObject(EffGlobm2s8);
  fOutputFile->WriteTObject(EffGlobm2s9);
  fOutputFile->WriteTObject(EffGlobm2s10);
  fOutputFile->WriteTObject(EffGlobm2s11);
  fOutputFile->WriteTObject(EffGlobm2s12);

  //wheel-1
  fOutputFile->WriteTObject(EffGlobWm1);
  fOutputFile->WriteTObject(EffGlobm1s1);
  fOutputFile->WriteTObject(EffGlobm1s2);
  fOutputFile->WriteTObject(EffGlobm1s3);
  fOutputFile->WriteTObject(EffGlobm1s4);
  fOutputFile->WriteTObject(EffGlobm1s5);
  fOutputFile->WriteTObject(EffGlobm1s6);
  fOutputFile->WriteTObject(EffGlobm1s7);
  fOutputFile->WriteTObject(EffGlobm1s8);
  fOutputFile->WriteTObject(EffGlobm1s9);
  fOutputFile->WriteTObject(EffGlobm1s10);
  fOutputFile->WriteTObject(EffGlobm1s11);
  fOutputFile->WriteTObject(EffGlobm1s12);
  
  //wheel0
  fOutputFile->WriteTObject(EffGlobW0);
  fOutputFile->WriteTObject(EffGlob1);
  fOutputFile->WriteTObject(EffGlob2);
  fOutputFile->WriteTObject(EffGlob3);
  fOutputFile->WriteTObject(EffGlob4);
  fOutputFile->WriteTObject(EffGlob5);
  fOutputFile->WriteTObject(EffGlob6);
  fOutputFile->WriteTObject(EffGlob7);
  fOutputFile->WriteTObject(EffGlob8);
  fOutputFile->WriteTObject(EffGlob9);
  fOutputFile->WriteTObject(EffGlob10);
  fOutputFile->WriteTObject(EffGlob11);
  fOutputFile->WriteTObject(EffGlob12);

  //wheel1
  fOutputFile->WriteTObject(EffGlobW1);
  fOutputFile->WriteTObject(EffGlob1s1);
  fOutputFile->WriteTObject(EffGlob1s2);
  fOutputFile->WriteTObject(EffGlob1s3);
  fOutputFile->WriteTObject(EffGlob1s4);
  fOutputFile->WriteTObject(EffGlob1s5);
  fOutputFile->WriteTObject(EffGlob1s6);
  fOutputFile->WriteTObject(EffGlob1s7);
  fOutputFile->WriteTObject(EffGlob1s8);
  fOutputFile->WriteTObject(EffGlob1s9);
  fOutputFile->WriteTObject(EffGlob1s10);
  fOutputFile->WriteTObject(EffGlob1s11);
  fOutputFile->WriteTObject(EffGlob1s12);
  
  //wheel2
  fOutputFile->WriteTObject(EffGlobW2);
  fOutputFile->WriteTObject(EffGlob2s1);
  fOutputFile->WriteTObject(EffGlob2s2);
  fOutputFile->WriteTObject(EffGlob2s3);
  fOutputFile->WriteTObject(EffGlob2s4);
  fOutputFile->WriteTObject(EffGlob2s5);
  fOutputFile->WriteTObject(EffGlob2s6);
  fOutputFile->WriteTObject(EffGlob2s7);
  fOutputFile->WriteTObject(EffGlob2s8);
  fOutputFile->WriteTObject(EffGlob2s9);
  fOutputFile->WriteTObject(EffGlob2s10);
  fOutputFile->WriteTObject(EffGlob2s11);
  fOutputFile->WriteTObject(EffGlob2s12);

  fOutputFile->Close();
  edm::LogInfo (nameInLog) <<"Beginning DQMMonitorDigi " ;
  if(EffSaveRootFile) dbe->showDirStructure();
}



void MuonSegmentEff::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  std::map<RPCDetId, int> buff;

  char layerLabel[128];
  char meIdRPC [128];
  char meIdDT [128];
  char meRPC [128];
  char meIdCSC [128];

  std::cout<<"New Event "<<iEvent.id().event()<<std::endl;

  std::cout <<"\t Getting the RPC Geometry"<<std::endl;
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  std::cout <<"\t Getting the RPC Digis"<<std::endl;
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(muonRPCDigis, rpcDigis);
  
  std::map<DTStationIndex,std::set<RPCDetId> > rollstoreDT;
  std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;    
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	
	if(region==0&&(incldt||incldtMB4)){
	  //std::cout<<"--Filling the barrel"<<rpcId<<std::endl;
	  int wheel=rpcId.ring();
	  int sector=rpcId.sector();
	  int station=rpcId.station();
	  DTStationIndex ind(region,wheel,sector,station);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreDT.find(ind)!=rollstoreDT.end()) myrolls=rollstoreDT[ind];
	  myrolls.insert(rpcId);
	  rollstoreDT[ind]=myrolls;
	}
	else if(inclcsc){
	  //std::cout<<"--Filling the EndCaps!"<<rpcId<<std::endl;
	  int region=rpcId.region();
          int station=rpcId.station();
          int ring=rpcId.ring();
          int cscring=ring;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = rpcsrv.segment();
	  int cscchamber = rpcsegment;
          if((station==2||station==3)&&ring==3){//Adding Ring 3 of RPC to the CSC Ring 2
            cscring = 2;
          }
	  if((station==4)&&(ring==2||ring==3)){//RE4 have just ring 1
            cscstation=3;
            cscring=2;
          }
          CSCStationIndex ind(region,cscstation,cscring,cscchamber);
          std::set<RPCDetId> myrolls;
	  if (rollstoreCSC.find(ind)!=rollstoreCSC.end()){
            myrolls=rollstoreCSC[ind];
          }
          
          myrolls.insert(rpcId);
          rollstoreCSC[ind]=myrolls;
        }
      }
    }
  }
  

  if(incldt){
#include "dtpart.inl"
  }
  
  if(incldtMB4){
#include "rb4part.inl"
  }
  
  if(inclcsc){
#include "cscpart.inl"
  }
}


void MuonSegmentEff::endJob() {
  std::cout<<"Begin End Job"<<std::endl;
  int indexm2[13];
  int indexm1[13];
  int index0[13];
  int index1[13];
  int index2[13];
  std::cout<<"Starging loop 1"<<std::endl;
  for(int j=0;j<13;j++){
    indexm2[j]=1;
    indexm1[j]=1;
    index0[j]=1;
    index1[j]=1;
    index2[j]=1;
  }

  std::cout<<"Starging loop 2"<<std::endl;
  int indexWheel[5];
  for(int j=0;j<5;j++){
    indexWheel[j]=1;
  }
  
  std::cout<<"Printing twiki header "<<std::endl;
  oftwiki <<"|  RPC Name  |  Observed  |  Predicted  |  Efficiency %  |  Error %  |";
  

  std::map<RPCDetId, int> pred = counter[0];
  std::map<RPCDetId, int> obse = counter[1];
  std::map<RPCDetId, int> reje = counter[2];
  std::map<RPCDetId, int>::iterator irpc;


  
  std::cout<<"Looping on final container "<<std::endl;
  for (irpc=pred.begin(); irpc!=pred.end();irpc++){
    RPCDetId id=irpc->first;
    
    RPCGeomServ RPCname(id);
    std::string nameRoll = RPCname.name();
    std::string wheel;
    std::string rpc;
    std::string partition;
    
    int p=pred[id]; 
    int o=obse[id]; 
    int r=reje[id]; 
    assert(p==o+r);
    
    //-----------------------Fillin Global Histograms----------------------------------------

    
    //std::cout<<"Doing Global Histograms "<<std::endl;
    
    if(p!=0&&id.region()==0){
      float ef = float(o)/float(p); 
      float er = sqrt(ef*(1.-ef)/float(p));
      ef=ef*100;
      er=er*100;
      if(ef>0.){
	char cam[128];	
	sprintf(cam,"%s",nameRoll.c_str());
	TString camera = (TString)cam;
	int Ring=id.ring();
	int Sector=id.sector();
	
	//std::cout<<"p!=0 now going into ifs... "<<std::endl;

	if(Ring==-2){
	  //indexWheel[0]++;  EffGlobWm2->SetBinContent(indexWheel[0],ef);  EffGlobWm2->SetBinError(indexWheel[0],er);  EffGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera);  EffGlobWm2->GetXaxis()->LabelsOption("v");
	  if(Sector==1)indexm2[1]++;  EffGlobm2s1->SetBinContent(indexm2[1],ef);  EffGlobm2s1->SetBinError(indexm2[1],er);  EffGlobm2s1->GetXaxis()->SetBinLabel(indexm2[1],camera);  EffGlobm2s1->GetXaxis()->LabelsOption("v");
	  if(Sector==2)indexm2[2]++;  EffGlobm2s2->SetBinContent(indexm2[2],ef);  EffGlobm2s2->SetBinError(indexm2[2],er);  EffGlobm2s2->GetXaxis()->SetBinLabel(indexm2[2],camera);  EffGlobm2s2->GetXaxis()->LabelsOption("v");
	  if(Sector==3)indexm2[3]++;  EffGlobm2s3->SetBinContent(indexm2[3],ef);  EffGlobm2s3->SetBinError(indexm2[3],er);  EffGlobm2s3->GetXaxis()->SetBinLabel(indexm2[3],camera);  EffGlobm2s3->GetXaxis()->LabelsOption("v");
	  if(Sector==4)indexm2[4]++;  EffGlobm2s4->SetBinContent(indexm2[4],ef);  EffGlobm2s4->SetBinError(indexm2[4],er);  EffGlobm2s4->GetXaxis()->SetBinLabel(indexm2[4],camera);  EffGlobm2s4->GetXaxis()->LabelsOption("v");
	  if(Sector==5)indexm2[5]++;  EffGlobm2s5->SetBinContent(indexm2[5],ef);  EffGlobm2s5->SetBinError(indexm2[5],er);  EffGlobm2s5->GetXaxis()->SetBinLabel(indexm2[5],camera);  EffGlobm2s5->GetXaxis()->LabelsOption("v");
	  if(Sector==6)indexm2[6]++;  EffGlobm2s6->SetBinContent(indexm2[6],ef);  EffGlobm2s6->SetBinError(indexm2[6],er);  EffGlobm2s6->GetXaxis()->SetBinLabel(indexm2[6],camera);  EffGlobm2s6->GetXaxis()->LabelsOption("v");
	  if(Sector==7)indexm2[7]++;  EffGlobm2s7->SetBinContent(indexm2[7],ef);  EffGlobm2s7->SetBinError(indexm2[7],er);  EffGlobm2s7->GetXaxis()->SetBinLabel(indexm2[7],camera);  EffGlobm2s7->GetXaxis()->LabelsOption("v");
	  if(Sector==8)indexm2[8]++;  EffGlobm2s8->SetBinContent(indexm2[8],ef);  EffGlobm2s8->SetBinError(indexm2[8],er);  EffGlobm2s8->GetXaxis()->SetBinLabel(indexm2[8],camera);  EffGlobm2s8->GetXaxis()->LabelsOption("v");
	  if(Sector==9)indexm2[9]++;  EffGlobm2s9->SetBinContent(indexm2[9],ef);  EffGlobm2s9->SetBinError(indexm2[9],er);  EffGlobm2s9->GetXaxis()->SetBinLabel(indexm2[9],camera);  EffGlobm2s9->GetXaxis()->LabelsOption("v");
	  if(Sector==10)indexm2[10]++;  EffGlobm2s10->SetBinContent(indexm2[10],ef);  EffGlobm2s10->SetBinError(indexm2[10],er);  EffGlobm2s10->GetXaxis()->SetBinLabel(indexm2[10],camera);  EffGlobm2s10->GetXaxis()->LabelsOption("v");
	  if(Sector==11)indexm2[11]++;  EffGlobm2s11->SetBinContent(indexm2[11],ef);  EffGlobm2s11->SetBinError(indexm2[11],er);  EffGlobm2s11->GetXaxis()->SetBinLabel(indexm2[11],camera);  EffGlobm2s11->GetXaxis()->LabelsOption("v");
	  if(Sector==12)indexm2[12]++;  EffGlobm2s12->SetBinContent(indexm2[12],ef);  EffGlobm2s12->SetBinError(indexm2[12],er);  EffGlobm2s12->GetXaxis()->SetBinLabel(indexm2[12],camera);  EffGlobm2s12->GetXaxis()->LabelsOption("v");
	}

	if(Ring==-1){
	  //indexWheel[1]++;  EffGlobWm1->SetBinContent(indexWheel[1],ef);  EffGlobWm1->SetBinError(indexWheel[1],er);  EffGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera);  EffGlobWm1->GetXaxis()->LabelsOption("v");
	  if(Sector==1)indexm1[1]++;  EffGlobm1s1->SetBinContent(indexm1[1],ef);  EffGlobm1s1->SetBinError(indexm1[1],er);  EffGlobm1s1->GetXaxis()->SetBinLabel(indexm1[1],camera);  EffGlobm1s1->GetXaxis()->LabelsOption("v");
	  if(Sector==2)indexm1[2]++;  EffGlobm1s2->SetBinContent(indexm1[2],ef);  EffGlobm1s2->SetBinError(indexm1[2],er);  EffGlobm1s2->GetXaxis()->SetBinLabel(indexm1[2],camera);  EffGlobm1s2->GetXaxis()->LabelsOption("v");
	  if(Sector==3)indexm1[3]++;  EffGlobm1s3->SetBinContent(indexm1[3],ef);  EffGlobm1s3->SetBinError(indexm1[3],er);  EffGlobm1s3->GetXaxis()->SetBinLabel(indexm1[3],camera);  EffGlobm1s3->GetXaxis()->LabelsOption("v");
	  if(Sector==4)indexm1[4]++;  EffGlobm1s4->SetBinContent(indexm1[4],ef);  EffGlobm1s4->SetBinError(indexm1[4],er);  EffGlobm1s4->GetXaxis()->SetBinLabel(indexm1[4],camera);  EffGlobm1s4->GetXaxis()->LabelsOption("v");
	  if(Sector==5)indexm1[5]++;  EffGlobm1s5->SetBinContent(indexm1[5],ef);  EffGlobm1s5->SetBinError(indexm1[5],er);  EffGlobm1s5->GetXaxis()->SetBinLabel(indexm1[5],camera);  EffGlobm1s5->GetXaxis()->LabelsOption("v");
	  if(Sector==6)indexm1[6]++;  EffGlobm1s6->SetBinContent(indexm1[6],ef);  EffGlobm1s6->SetBinError(indexm1[6],er);  EffGlobm1s6->GetXaxis()->SetBinLabel(indexm1[6],camera);  EffGlobm1s6->GetXaxis()->LabelsOption("v");
	  if(Sector==7)indexm1[7]++;  EffGlobm1s7->SetBinContent(indexm1[7],ef);  EffGlobm1s7->SetBinError(indexm1[7],er);  EffGlobm1s7->GetXaxis()->SetBinLabel(indexm1[7],camera);  EffGlobm1s7->GetXaxis()->LabelsOption("v");
	  if(Sector==8)indexm1[8]++;  EffGlobm1s8->SetBinContent(indexm1[8],ef);  EffGlobm1s8->SetBinError(indexm1[8],er);  EffGlobm1s8->GetXaxis()->SetBinLabel(indexm1[8],camera);  EffGlobm1s8->GetXaxis()->LabelsOption("v");
	  if(Sector==9)indexm1[9]++;  EffGlobm1s9->SetBinContent(indexm1[9],ef);  EffGlobm1s9->SetBinError(indexm1[9],er);  EffGlobm1s9->GetXaxis()->SetBinLabel(indexm1[9],camera);  EffGlobm1s9->GetXaxis()->LabelsOption("v");
	  if(Sector==10)indexm1[10]++;  EffGlobm1s10->SetBinContent(indexm1[10],ef);  EffGlobm1s10->SetBinError(indexm1[10],er);  EffGlobm1s10->GetXaxis()->SetBinLabel(indexm1[10],camera);  EffGlobm1s10->GetXaxis()->LabelsOption("v");
	  if(Sector==11)indexm1[11]++;  EffGlobm1s11->SetBinContent(indexm1[11],ef);  EffGlobm1s11->SetBinError(indexm1[11],er);  EffGlobm1s11->GetXaxis()->SetBinLabel(indexm1[11],camera);  EffGlobm1s11->GetXaxis()->LabelsOption("v");
	  if(Sector==12)indexm1[12]++;  EffGlobm1s12->SetBinContent(indexm1[12],ef);  EffGlobm1s12->SetBinError(indexm1[12],er);  EffGlobm1s12->GetXaxis()->SetBinLabel(indexm1[12],camera);  EffGlobm1s12->GetXaxis()->LabelsOption("v");
	}

	if(Ring==0){
	  indexWheel[2]++;  EffGlobW0->SetBinContent(indexWheel[2],ef);  EffGlobW0->SetBinError(indexWheel[2],er);  EffGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera);  EffGlobW0->GetXaxis()->LabelsOption("v");
	  if(Sector==1)index0[1]++;  EffGlob1->SetBinContent(index0[1],ef);  EffGlob1->SetBinError(index0[1],er);  EffGlob1->GetXaxis()->SetBinLabel(index0[1],camera);  EffGlob1->GetXaxis()->LabelsOption("v");
	  if(Sector==2)index0[2]++;  EffGlob2->SetBinContent(index0[2],ef);  EffGlob2->SetBinError(index0[2],er);  EffGlob2->GetXaxis()->SetBinLabel(index0[2],camera);  EffGlob2->GetXaxis()->LabelsOption("v");
	  if(Sector==3)index0[3]++;  EffGlob3->SetBinContent(index0[3],ef);  EffGlob3->SetBinError(index0[3],er);  EffGlob3->GetXaxis()->SetBinLabel(index0[3],camera);  EffGlob3->GetXaxis()->LabelsOption("v");
	  if(Sector==4)index0[4]++;  EffGlob4->SetBinContent(index0[4],ef);  EffGlob4->SetBinError(index0[4],er);  EffGlob4->GetXaxis()->SetBinLabel(index0[4],camera);  EffGlob4->GetXaxis()->LabelsOption("v");
	  if(Sector==5)index0[5]++;  EffGlob5->SetBinContent(index0[5],ef);  EffGlob5->SetBinError(index0[5],er);  EffGlob5->GetXaxis()->SetBinLabel(index0[5],camera);  EffGlob5->GetXaxis()->LabelsOption("v");
	  if(Sector==6)index0[6]++;  EffGlob6->SetBinContent(index0[6],ef);  EffGlob6->SetBinError(index0[6],er);  EffGlob6->GetXaxis()->SetBinLabel(index0[6],camera);  EffGlob6->GetXaxis()->LabelsOption("v");
	  if(Sector==7)index0[7]++;  EffGlob7->SetBinContent(index0[7],ef);  EffGlob7->SetBinError(index0[7],er);  EffGlob7->GetXaxis()->SetBinLabel(index0[7],camera);  EffGlob7->GetXaxis()->LabelsOption("v");
	  if(Sector==8)index0[8]++;  EffGlob8->SetBinContent(index0[8],ef);  EffGlob8->SetBinError(index0[8],er);  EffGlob8->GetXaxis()->SetBinLabel(index0[8],camera);  EffGlob8->GetXaxis()->LabelsOption("v");
	  if(Sector==9)index0[9]++;  EffGlob9->SetBinContent(index0[9],ef);  EffGlob9->SetBinError(index0[9],er);  EffGlob9->GetXaxis()->SetBinLabel(index0[9],camera);  EffGlob9->GetXaxis()->LabelsOption("v");
	  if(Sector==10)index0[10]++;  EffGlob10->SetBinContent(index0[10],ef);  EffGlob10->SetBinError(index0[10],er);  EffGlob10->GetXaxis()->SetBinLabel(index0[10],camera);  EffGlob10->GetXaxis()->LabelsOption("v");
	  if(Sector==11)index0[11]++;  EffGlob11->SetBinContent(index0[11],ef);  EffGlob11->SetBinError(index0[11],er);  EffGlob11->GetXaxis()->SetBinLabel(index0[11],camera);  EffGlob11->GetXaxis()->LabelsOption("v");
	  if(Sector==12)index0[12]++;  EffGlob12->SetBinContent(index0[12],ef);  EffGlob12->SetBinError(index0[12],er);  EffGlob12->GetXaxis()->SetBinLabel(index0[12],camera);  EffGlob12->GetXaxis()->LabelsOption("v");
	}
	
	if(Ring==1){
	  //indexWheel[3]++;  EffGlobW1->SetBinContent(indexWheel[3],ef);  EffGlobW1->SetBinError(indexWheel[3],er);  EffGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera);  EffGlobW1->GetXaxis()->LabelsOption("v");
	  if(Sector==1)index1[1]++;  EffGlob1s1->SetBinContent(index1[1],ef);  EffGlob1s1->SetBinError(index1[1],er);  EffGlob1s1->GetXaxis()->SetBinLabel(index1[1],camera);  EffGlob1s1->GetXaxis()->LabelsOption("v");
	  if(Sector==2)index1[2]++;  EffGlob1s2->SetBinContent(index1[2],ef);  EffGlob1s2->SetBinError(index1[2],er);  EffGlob1s2->GetXaxis()->SetBinLabel(index1[2],camera);  EffGlob1s2->GetXaxis()->LabelsOption("v");
	  if(Sector==3)index1[3]++;  EffGlob1s3->SetBinContent(index1[3],ef);  EffGlob1s3->SetBinError(index1[3],er);  EffGlob1s3->GetXaxis()->SetBinLabel(index1[3],camera);  EffGlob1s3->GetXaxis()->LabelsOption("v");
	  if(Sector==4)index1[4]++;  EffGlob1s4->SetBinContent(index1[4],ef);  EffGlob1s4->SetBinError(index1[4],er);  EffGlob1s4->GetXaxis()->SetBinLabel(index1[4],camera);  EffGlob1s4->GetXaxis()->LabelsOption("v");
	  if(Sector==5)index1[5]++;  EffGlob1s5->SetBinContent(index1[5],ef);  EffGlob1s5->SetBinError(index1[5],er);  EffGlob1s5->GetXaxis()->SetBinLabel(index1[5],camera);  EffGlob1s5->GetXaxis()->LabelsOption("v");
	  if(Sector==6)index1[6]++;  EffGlob1s6->SetBinContent(index1[6],ef);  EffGlob1s6->SetBinError(index1[6],er);  EffGlob1s6->GetXaxis()->SetBinLabel(index1[6],camera);  EffGlob1s6->GetXaxis()->LabelsOption("v");
	  if(Sector==7)index1[7]++;  EffGlob1s7->SetBinContent(index1[7],ef);  EffGlob1s7->SetBinError(index1[7],er);  EffGlob1s7->GetXaxis()->SetBinLabel(index1[7],camera);  EffGlob1s7->GetXaxis()->LabelsOption("v");
	  if(Sector==8)index1[8]++;  EffGlob1s8->SetBinContent(index1[8],ef);  EffGlob1s8->SetBinError(index1[8],er);  EffGlob1s8->GetXaxis()->SetBinLabel(index1[8],camera);  EffGlob1s8->GetXaxis()->LabelsOption("v");
	  if(Sector==9)index1[9]++;  EffGlob1s9->SetBinContent(index1[9],ef);  EffGlob1s9->SetBinError(index1[9],er);  EffGlob1s9->GetXaxis()->SetBinLabel(index1[9],camera);  EffGlob1s9->GetXaxis()->LabelsOption("v");
	  if(Sector==10)index1[10]++;  EffGlob1s10->SetBinContent(index1[10],ef);  EffGlob1s10->SetBinError(index1[10],er);  EffGlob1s10->GetXaxis()->SetBinLabel(index1[10],camera);  EffGlob1s10->GetXaxis()->LabelsOption("v");
	  if(Sector==11)index1[11]++;  EffGlob1s11->SetBinContent(index1[11],ef);  EffGlob1s11->SetBinError(index1[11],er);  EffGlob1s11->GetXaxis()->SetBinLabel(index1[11],camera);  EffGlob1s11->GetXaxis()->LabelsOption("v");
	  if(Sector==12)index1[12]++;  EffGlob1s12->SetBinContent(index1[12],ef);  EffGlob1s12->SetBinError(index1[12],er);  EffGlob1s12->GetXaxis()->SetBinLabel(index1[12],camera);  EffGlob1s12->GetXaxis()->LabelsOption("v");
	}

	
	if(Ring==2){
	  //indexWheel[4]++;  EffGlobW2->SetBinContent(indexWheel[4],ef);  EffGlobW2->SetBinError(indexWheel[4],er);  EffGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera);  EffGlobW2->GetXaxis()->LabelsOption("v");
	  if(Sector==1)index2[1]++;  EffGlob2s1->SetBinContent(index2[1],ef);  EffGlob2s1->SetBinError(index2[1],er);  EffGlob2s1->GetXaxis()->SetBinLabel(index2[1],camera);  EffGlob2s1->GetXaxis()->LabelsOption("v");
	  if(Sector==2)index2[2]++;  EffGlob2s2->SetBinContent(index2[2],ef);  EffGlob2s2->SetBinError(index2[2],er);  EffGlob2s2->GetXaxis()->SetBinLabel(index2[2],camera);  EffGlob2s2->GetXaxis()->LabelsOption("v");
	  if(Sector==3)index2[3]++;  EffGlob2s3->SetBinContent(index2[3],ef);  EffGlob2s3->SetBinError(index2[3],er);  EffGlob2s3->GetXaxis()->SetBinLabel(index2[3],camera);  EffGlob2s3->GetXaxis()->LabelsOption("v");
	  if(Sector==4)index2[4]++;  EffGlob2s4->SetBinContent(index2[4],ef);  EffGlob2s4->SetBinError(index2[4],er);  EffGlob2s4->GetXaxis()->SetBinLabel(index2[4],camera);  EffGlob2s4->GetXaxis()->LabelsOption("v");
	  if(Sector==5)index2[5]++;  EffGlob2s5->SetBinContent(index2[5],ef);  EffGlob2s5->SetBinError(index2[5],er);  EffGlob2s5->GetXaxis()->SetBinLabel(index2[5],camera);  EffGlob2s5->GetXaxis()->LabelsOption("v");
	  if(Sector==6)index2[6]++;  EffGlob2s6->SetBinContent(index2[6],ef);  EffGlob2s6->SetBinError(index2[6],er);  EffGlob2s6->GetXaxis()->SetBinLabel(index2[6],camera);  EffGlob2s6->GetXaxis()->LabelsOption("v");
	  if(Sector==7)index2[7]++;  EffGlob2s7->SetBinContent(index2[7],ef);  EffGlob2s7->SetBinError(index2[7],er);  EffGlob2s7->GetXaxis()->SetBinLabel(index2[7],camera);  EffGlob2s7->GetXaxis()->LabelsOption("v");
	  if(Sector==8)index2[8]++;  EffGlob2s8->SetBinContent(index2[8],ef);  EffGlob2s8->SetBinError(index2[8],er);  EffGlob2s8->GetXaxis()->SetBinLabel(index2[8],camera);  EffGlob2s8->GetXaxis()->LabelsOption("v");
	  if(Sector==9)index2[9]++;  EffGlob2s9->SetBinContent(index2[9],ef);  EffGlob2s9->SetBinError(index2[9],er);  EffGlob2s9->GetXaxis()->SetBinLabel(index2[9],camera);  EffGlob2s9->GetXaxis()->LabelsOption("v");
	  if(Sector==10)index2[10]++;  EffGlob2s10->SetBinContent(index2[10],ef);  EffGlob2s10->SetBinError(index2[10],er);  EffGlob2s10->GetXaxis()->SetBinLabel(index2[10],camera);  EffGlob2s10->GetXaxis()->LabelsOption("v");
	  if(Sector==11)index2[11]++;  EffGlob2s11->SetBinContent(index2[11],ef);  EffGlob2s11->SetBinError(index2[11],er);  EffGlob2s11->GetXaxis()->SetBinLabel(index2[11],camera);  EffGlob2s11->GetXaxis()->LabelsOption("v");
	  if(Sector==12)index2[12]++;  EffGlob2s12->SetBinContent(index2[12],ef);  EffGlob2s12->SetBinError(index2[12],er);  EffGlob2s12->GetXaxis()->SetBinLabel(index2[12],camera);  EffGlob2s12->GetXaxis()->LabelsOption("v");
	}
      }
    }
    
    //-----------------------Done Global Histogram----------------------------------------

    if(p!=0){
      float ef = float(o)/float(p); 
      float er = sqrt(ef*(1.-ef)/float(p));
      std::cout <<"\n "<<id<<"\t Predicted "<<p<<"\t Observed "<<o<<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %";
      ofeff <<"\n "<<id<<"\t Predicted "<<p<<"\t Observed "<<o<<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %";
      RPCGeomServ RPCname(id);
      oftwiki <<"\n |  "<<RPCname.name()<<"  |  "<<o<<"  |  "<<p<<"  |  "<<ef*100.<<"  |  "<<er*100<<"  |";
      //if(ef<0.8){
      //std::cout<<"\t \t Warning!";
      //ofeff<<"\t \t Warning!";
      //} 
    }
    else{
      std::cout<<"No predictions in this file p=0"<<std::endl;
      ofeff<<"No predictions in this file p=0"<<std::endl;
    }
    std::cout<<"Done Final Loop"<<std::endl;
  }
  if(totalcounter[0]!=0){
    float tote = float(totalcounter[1])/float(totalcounter[0]);
    float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
  
    std::cout <<"\n\n \t \t TOTAL EFFICIENCY \t Predicted "<<totalcounter[0]<<"\t Observed "<<totalcounter[1]<<"\t Eff = "<<tote*100.<<"\t +/- \t"<<totr*100.<<" %"<<std::endl;
    std::cout <<totalcounter[1]<<" "<<totalcounter[0]<<" flagcode"<<std::endl;
    
    ofeff <<"\n\n \t \t TOTAL EFFICIENCY \t Predicted "<<totalcounter[0]<<"\t Observed "<<totalcounter[1]<<"\t Eff = "<<tote*100.<<"\t +/- \t"<<totr*100.<<" %"<<std::endl;
    ofeff <<totalcounter[1]<<" "<<totalcounter[0]<<" flagcode"<<std::endl;

  }
  else{
    std::cout<<"No predictions in this file = 0!!!"<<std::endl;
    ofeff <<"No predictions in this file = 0!!!"<<std::endl;
  }

  std::vector<std::string>::iterator meIt;
  int k = 0;

  if(EffSaveRootFile==true){
    for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){
      k++;
      
      char detUnitLabel[128];

      char meIdRPC [128];
      char meIdDT [128];
      char effIdRPC_DT [128];

      char meIdRPC_2D [128];
      char meIdDT_2D [128];
      char effIdRPC_DT_2D [128];

    
      sprintf(detUnitLabel ,"%s",(*meIt).c_str());
    
      std::cout<<"Creating Efficiency Root File #"<<k<<std::endl;

      sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
      sprintf(meIdRPC_2D,"RPCDataOccupancy2DFromDT_%s",detUnitLabel);

      sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
      sprintf(meIdDT_2D,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
      sprintf(effIdRPC_DT,"EfficienyFromDTExtrapolation_%s",detUnitLabel);
      sprintf(effIdRPC_DT_2D,"EfficienyFromDT2DExtrapolation_%s",detUnitLabel);

      std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

      for(unsigned int i=1;i<=100;++i){
	if(meMap[meIdDT]->getBinContent(i) != 0){
	  float eff = meMap[meIdRPC]->getBinContent(i)/meMap[meIdDT]->getBinContent(i);
	  float erreff = sqrt(eff*(1-eff)/meMap[meIdDT]->getBinContent(i));
	  meMap[effIdRPC_DT]->setBinContent(i,eff*100.);
	  meMap[effIdRPC_DT]->setBinError(i,erreff*100.);
	}
      }
      for(unsigned int i=1;i<=100;++i){
	for(unsigned int j=1;j<=200;++j){
	  if(meMap[meIdDT_2D]->getBinContent(i,j) != 0){
	    float eff = meMap[meIdRPC_2D]->getBinContent(i,j)/meMap[meIdDT_2D]->getBinContent(i,j);
	    float erreff = sqrt(eff*(1-eff)/meMap[meIdDT_2D]->getBinContent(i,j));
	    meMap[effIdRPC_DT_2D]->setBinContent(i,j,eff*100.);
	    meMap[effIdRPC_DT_2D]->setBinError(i,j,erreff*100.);
	  }
	}
      }
      ///CSC

    
      char meRPC [128];
      char meIdCSC [128];
      char effIdRPC_CSC [128];

      char meRPC_2D [128];
      char meIdCSC_2D [128];
      char effIdRPC_CSC_2D [128];

      sprintf(detUnitLabel ,"%s",(*meIt).c_str());

      sprintf(meRPC,"RPCDataOccupancyFromCSC_%s",detUnitLabel);
      sprintf(meRPC_2D,"RPCDataOccupancy2DFromCSC_%s",detUnitLabel);

      sprintf(meIdCSC,"ExpectedOccupancyFromCSC_%s",detUnitLabel);
      sprintf(meIdCSC_2D,"ExpectedOccupancy2DFromCSC_%s",detUnitLabel);

      sprintf(effIdRPC_CSC,"EfficienyFromCSCExtrapolation_%s",detUnitLabel);
      sprintf(effIdRPC_CSC_2D,"EfficienyFromCSC2DExtrapolation_%s",detUnitLabel);

      //std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

      for(unsigned int i=1;i<=100;++i){
      
	if(meMap[meIdCSC]->getBinContent(i) != 0){
	  float eff = meMap[meRPC]->getBinContent(i)/meMap[meIdCSC]->getBinContent(i);
	  float erreff = sqrt(eff*(1-eff)/meMap[meIdCSC]->getBinContent(i));
	  meMap[effIdRPC_CSC]->setBinContent(i,eff*100.);
	  meMap[effIdRPC_CSC]->setBinError(i,erreff*100.);
	}
      }
      for(unsigned int i=1;i<=100;++i){
	for(unsigned int j=1;j<=200;++j){
	  if(meMap[meIdCSC_2D]->getBinContent(i,j) != 0){
	    float eff = meMap[meRPC_2D]->getBinContent(i,j)/meMap[meIdCSC_2D]->getBinContent(i,j);
	    float erreff = sqrt(eff*(1-eff)/meMap[meIdCSC_2D]->getBinContent(i,j));
	    meMap[effIdRPC_CSC_2D]->setBinContent(i,j,eff*100.);
	    meMap[effIdRPC_CSC_2D]->setBinError(i,j,erreff*100.);
	  }
	}
      }
    }
  
    //Giuseppe
  
    dbe->save(EffRootFileName);
    std::cout<<"Saving RootFile"<<std::endl;
  }  

  TCanvas *Ca1 = new TCanvas("Ca1","GlobalPlotsForWheels",6000,1200);
  
  std::cout<<"Creating Wheel 0  Image"<<std::endl;
  EffGlobW0->Draw();
  Ca1->SaveAs("Wheel0Efficiency.png");
  Ca1->Clear();
  
  TCanvas *Ca2 = new TCanvas("Ca2","Residuals",800,600);
  std::cout<<"Creating Residuals for Different Layers"<<std::endl;
  hGlobalResLa1->Draw();
  hGlobalResLa1->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsLayer1.png");
  Ca2->Clear();

  hGlobalResLa2->Draw();
  hGlobalResLa2->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsLayer2.png");
  Ca2->Clear();

  hGlobalResLa3->Draw();
  hGlobalResLa3->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsLayer3.png");
  Ca2->Clear();

  hGlobalResLa4->Draw();
  hGlobalResLa4->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsLayer4.png");
  Ca2->Clear();
  
  hGlobalResLa5->Draw();
  hGlobalResLa5->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsLayer5.png");
  Ca2->Clear();

  hGlobalResLa6->Draw();
  hGlobalResLa6->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsLayer6.png");
  Ca2->Clear();
  
  hGlobalResClu1->Draw();
  hGlobalResClu1->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsCluster1.png");
  Ca2->Clear();
  
  hGlobalResClu2->Draw();
  hGlobalResClu2->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsCluster2.png");
  Ca2->Clear();

  hGlobalResClu3->Draw();
  hGlobalResClu3->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsCluster3.png");
  Ca2->Clear();

  hGlobalResClu4->Draw();
  hGlobalResClu4->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsCluster4.png");
  Ca2->Clear();

  hGlobalResClu1La1->Draw();
  hGlobalResClu1La1->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu1La1.png");
  Ca2->Clear();

  hGlobalResClu1La2->Draw();
  hGlobalResClu1La2->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu1La2.png");
  Ca2->Clear();

  hGlobalResClu1La3->Draw();
  hGlobalResClu1La3->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu1La3.png");
  Ca2->Clear();
  
  hGlobalResClu1La4->Draw();
  hGlobalResClu1La4->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu1La4.png");
  Ca2->Clear();

  hGlobalResClu1La5->Draw();
  hGlobalResClu1La5->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu1La5.png");
  Ca2->Clear();

  hGlobalResClu1La6->Draw();
  hGlobalResClu1La6->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu1La6.png");
  Ca2->Clear();
  
  std::ofstream layertableClu1;
  layertableClu1.open("layertableClu1.txt");
  layertableClu1<<"|  Layer  |  RMS for Cluster Size =1  |  Strip Width / sqrt(12)  |"<<std::endl;
  layertableClu1<<"|  1  |  "<<hGlobalResClu1La1->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu1<<"|  2  |  "<<hGlobalResClu1La2->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu1<<"|  3  |  "<<hGlobalResClu1La3->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu1<<"|  4  |  "<<hGlobalResClu1La4->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu1<<"|  5  |  "<<hGlobalResClu1La5->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu1<<"|  6  |  "<<hGlobalResClu1La6->GetRMS()<<"  |    | "<<std::endl; 
  
  layertableClu1.close();
  
  hGlobalResClu3La1->Draw();
  hGlobalResClu3La1->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu3La1.png");
  Ca2->Clear();

  hGlobalResClu3La2->Draw();
  hGlobalResClu3La2->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu3La2.png");
  Ca2->Clear();

  hGlobalResClu3La3->Draw();
  hGlobalResClu3La3->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu3La3.png");
  Ca2->Clear();
  
  hGlobalResClu3La4->Draw();
  hGlobalResClu3La4->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu3La4.png");
  Ca2->Clear();

  hGlobalResClu3La5->Draw();
  hGlobalResClu3La5->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu3La5.png");
  Ca2->Clear();

  hGlobalResClu3La6->Draw();
  hGlobalResClu3La6->GetXaxis()->SetTitle("Residuals (cm)");
  Ca2->SaveAs("ResidualsClu3La6.png");
  Ca2->Clear();

  std::ofstream layertableClu3;
  layertableClu3.open("layertableClu3.txt");
  layertableClu3<<"|  Layer  |  RMS for Cluster Size =1  |  Strip Width / sqrt(12)  |"<<std::endl;
  layertableClu3<<"|  1  |  "<<hGlobalResClu3La1->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu3<<"|  2  |  "<<hGlobalResClu3La2->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu3<<"|  3  |  "<<hGlobalResClu3La3->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu3<<"|  4  |  "<<hGlobalResClu3La4->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu3<<"|  5  |  "<<hGlobalResClu3La5->GetRMS()<<"  |    | "<<std::endl; 
  layertableClu3<<"|  6  |  "<<hGlobalResClu3La6->GetRMS()<<"  |    | "<<std::endl; 
  
  layertableClu3.close();








  
  //fOutputFile->Close();//??? Parece que esto borra el contenido del archivo!!!

  ofeff.close();
  oftwiki.close();
  ofrej.close();

}

