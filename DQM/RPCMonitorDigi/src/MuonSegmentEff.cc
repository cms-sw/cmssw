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
// $Id: MuonSegmentEff.cc,v 1.2 2007/10/05 08:48:59 mmaggi Exp $
//
//

#include "DQM/RPCMonitorDigi/interface/MuonSegmentEff.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>

#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>

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
  CSCStationIndex():_region(0){}
  CSCStationIndex(int region):
    _region(region){}
  ~CSCStationIndex(){}
  int region() const {return _region;}
  bool operator<(const CSCStationIndex& cscind) const{
    if(cscind.region()!=this->region())
      return cscind.region()<this->region();
    return false;
  }
private:
  int _region;
};


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
  ofrej.open("rejected.txt");

  incldt=iConfig.getParameter<bool>("incldt");
  inclcsc=iConfig.getParameter<bool>("inclcsc");
  widestrip=iConfig.getParameter<int>("widestrip");
  muonRPCDigis=iConfig.getParameter<std::string>("muonRPCDigis");
  cscSegments=iConfig.getParameter<std::string>("cscSegments");
  dt4DSegments=iConfig.getParameter<std::string>("dt4DSegments");


}


MuonSegmentEff::~MuonSegmentEff()
{

}

void MuonSegmentEff::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  std::map<RPCDetId, int> buff;

  float dx=0.,dy=0.,dz=0.,Xo=0.,Yo=0.,X=0.,Y=0.,Z=0.;
  
  std::cout<<"New Event "<<iEvent.id().event()<<std::endl;
  
  std::cout <<"\t Getting the RPC Geometry"<<std::endl;
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  std::cout <<"\t Getting the CSC Geometry"<<std::endl;
  edm::ESHandle<CSCGeometry> cscGeo;
  iSetup.get<MuonGeometryRecord>().get(cscGeo);

  std::cout <<"\t Getting the DT Geometry"<<std::endl;
  edm::ESHandle<DTGeometry> dtGeo;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  
  std::cout <<"\t Getting the RPC Digis"<<std::endl;
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(muonRPCDigis, rpcDigis);

  std::cout <<"\t Getting the CSC Segments"<<std::endl;
  edm::Handle<CSCSegmentCollection> allCSCSegments;
  iEvent.getByLabel(cscSegments, allCSCSegments);

  std::cout <<"\t Getting the DT Segments"<<std::endl;
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel(dt4DSegments, all4DSegments);
  
  std::map<DTStationIndex,std::set<RPCDetId> > rollstoreDT;
  std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	if(region==0){
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
	else{
	  //std::cout<<"--Filling the EndCaps"<<rpcId<<std::endl;
	  int region=rpcId.region();
	  int sector=rpcId.sector();
	  int subsector=rpcId.subsector();
	  int station=rpcId.station();
	  CSCStationIndex ind(region);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreCSC.find(ind)!=rollstoreCSC.end()) myrolls=rollstoreCSC[ind];
	  myrolls.insert(rpcId);
	  rollstoreCSC[ind]=myrolls;
	}
      }
    }
  }

  if(incldt){
#include "dtpart.inl"
  }
  
  if(inclcsc){
#include "cscpart.inl"
  }
  
}

void 
MuonSegmentEff::beginJob(const edm::EventSetup&)
{
}

void 
MuonSegmentEff::endJob() {
  std::map<RPCDetId, int> pred = counter[0];
  std::map<RPCDetId, int> obse = counter[1];
  std::map<RPCDetId, int> reje = counter[2];
  std::map<RPCDetId, int>::iterator irpc;
  
  for (irpc=pred.begin(); irpc!=pred.end();irpc++){
    RPCDetId id=irpc->first;
    int p=pred[id]; 
    int o=obse[id]; 
    int r=reje[id]; 
    assert(p==o+r);
    float ef = float(o)/float(p);
    float er = sqrt(ef*(1.-ef)/float(p));
    std::cout <<"\n "<<id<<"\t Predicted "<<p<<"\t Observed "<<o<<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %";
    if(ef<0.8){
      std::cout<<"\t \t Warning!";
    } 
  }
  float tote = float(totalcounter[1])/float(totalcounter[0]);
  float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
  
  std::cout <<"\n\n \t \t TOTAL EFFICIENCY \t Predicted "<<totalcounter[1]<<"\t Observed "<<totalcounter[0]<<"\t Eff = "<<tote*100.<<"\t +/- \t"<<totr*100.<<"%"<<std::endl;
  std::cout <<totalcounter[1]<<" "<<totalcounter[0]<<" flagcode"<<std::endl;
}
//define this as a plug-in
//DEFINE_FWK_MODULE(MuonSegmentEff);
