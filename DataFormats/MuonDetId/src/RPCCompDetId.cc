// -*- C++ -*-
//
// Package:     src
// Class  :     RPCCompDetId
// 
// Implementation:
//     [Notes on implementation]
//
// Author:      Marcello Maggi
// Created:     Wed Nov  2 12:09:10 CET 2011
// $Id: RPCCompDetId.cc,v 1.4 2012/02/04 23:27:47 eulisse Exp $
#include <iostream>
#include <sstream>
#include <iomanip>
#include "DataFormats/MuonDetId/interface/RPCCompDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h" 

RPCCompDetId::RPCCompDetId():DetId(DetId::Muon, MuonSubdetId::RPC),_dbname(""),_type(0){}

RPCCompDetId::RPCCompDetId(uint32_t id):DetId(id),_dbname(""),_type(0) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::RPC) {
    throw cms::Exception("InvalidDetId") << "RPCCompDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
      					 << " is not a valid RPC id";  
  }
}



RPCCompDetId::RPCCompDetId(DetId id):DetId(id),_dbname(""),_type(0) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::RPC) {
    throw cms::Exception("InvalidDetId") << "RPCCompDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid RPC id";
  }
}



RPCCompDetId::RPCCompDetId(int region,
			   int ring,
			   int station,
			   int sector,
			   int layer,
			   int subsector,
			   int type):
  DetId(DetId::Muon, MuonSubdetId::RPC),_dbname(""),_type(type)
{
  this->init(region,ring,station,sector,layer,subsector); 
}

RPCCompDetId::RPCCompDetId(const std::string& name, int type):
  DetId(DetId::Muon, MuonSubdetId::RPC),_dbname(name),_type(type)
{
  this->init();
}

bool 
RPCCompDetId::operator < (const RPCCompDetId& r) const{
  return this->dbname()<r.dbname();
}

int 
RPCCompDetId::region() const{
  return int((id_>>RegionStartBit_) & RegionMask_) + allRegionId;
}

int
RPCCompDetId::ring() const{
  return int((id_>>RingStartBit_) & RingMask_) + allRingId;
}


int
RPCCompDetId::wheel() const{
  int w=allRingId;
  if (this->region()==0)
    w=this->ring();
  return w;
}

int
RPCCompDetId::station() const{
  return int((id_>>StationStartBit_) & StationMask_) + allStationId;
}

int
RPCCompDetId::disk() const{
  int d=allStationId;
  if (this->region()!=0)
    d=this->station();
  return d;
}


int
RPCCompDetId::sector() const{
  return int((id_>>SectorStartBit_) & SectorMask_) + allSectorId;
}


int
RPCCompDetId::layer() const{
  return int((id_>>LayerStartBit_) & LayerMask_) + allLayerId;
}


int
RPCCompDetId::subsector() const{
  return int((id_>>SubSectorStartBit_) & SubSectorMask_) + allSubSectorId;
}


int
RPCCompDetId::type() const{
  return _type;
}

std::string
RPCCompDetId::dbname() const{
  std::string a=_dbname;
  if (a.size() == 0){
    if(this->type() == 0){
      a=this->gasDBname();
    }
  }
  return a;
}

void
RPCCompDetId::init(int region,
		   int ring,
		   int station,
		   int sector,
		   int layer,
		   int subsector)
{
  int maxRing=maxRingForwardId;
  if (!region)
    {
      maxRing=maxRingBarrelId;
    }
  
  if ( region     < allRegionId    || region    > maxRegionId ||
       ring       < allRingId      || ring      > maxRing ||
       station    < allStationId   || station   > maxStationId ||
       sector     < allSectorId    || sector    > maxSectorId ||
       layer      < allLayerId     || layer     > maxLayerId ||
       subsector  < allSubSectorId || subsector > maxSubSectorId ){
    
    throw cms::Exception("InvalidDetId") << "RPCDetId ctor:"
                                         << " Invalid parameters: "
                                         << " region "<<region
                                         << " ring "<<ring
                                         << " station "<<station
                                         << " sector "<<sector
                                         << " layer "<<layer
                                         << " subsector "<<subsector
                                         << std::endl;
  }
  int regionInBits    = region  -  allRegionId;
  int ringInBits      = ring     - allRingId;
  int stationInBits   = station  - allStationId;
  int sectorInBits    = sector   - allSectorId;
  int layerInBits     = layer    - allLayerId;
  int subSectorInBits = subsector- allSubSectorId;

  id_ |= ( regionInBits    & RegionMask_)    << RegionStartBit_    |
         ( ringInBits      & RingMask_)      << RingStartBit_      |
         ( stationInBits   & StationMask_)   << StationStartBit_   |
         ( sectorInBits    & SectorMask_)    << SectorStartBit_    |
         ( layerInBits     & LayerMask_)     << LayerStartBit_     |
         ( subSectorInBits & SubSectorMask_) << SubSectorStartBit_ ;
}

void 
RPCCompDetId::init()
{
  if (this->type()==0){
    this->initGas();
  }
}

void
RPCCompDetId::initGas()
{
  std::string buf(this->dbname()); 
  // check if the name contains the dcs namespace
  if (buf.find(':')!=buf.npos){
    buf = buf.substr(buf.find(':')+1,buf.npos);
  }
  _dbname=buf;
  // Check if endcap o barrel
  int region=0;
  if(buf.substr(0,1)=="W"){
    region=0;
  }else if(buf.substr(0,2)=="EP"){
    region=1;
  }else if(buf.substr(0,2)=="EM"){
    region=-1;
  }else{
    throw cms::Exception("InvalidDBName")<<" RPCCompDetId: "<<this->dbname()
					 <<" is not a valid DB Name for RPCCompDetId"
                                         << " det: " << det()
					 << " subdet: " << subdetId();
  }
  int ring=allRingId;
  int station = allStationId;
  int sector=allSectorId;
  int layer=allLayerId;
  int subsector=allSubSectorId;
    //Barrel
  if (region==0) {
    // Extract the Wheel (named ring)
    {
      std::stringstream os;
      os<<buf.substr(2,1);
      os>>ring;
      if (buf.substr(1,1)=="M"){
	ring *= -1;
      }
    }
    //Extract the station
    {
      std::stringstream os;
      os<<buf.substr(buf.find("RB")+2,1);
      os>>station;
    }
    //Extract the sector
    {
      std::stringstream os;
      os<<buf.substr(buf.find("S")+1,2);
      os>>sector;
    }
    //Extract subsector of sectors 4 and 10
    {
      if (buf.find("4L")!=buf.npos)
	subsector=1;
      if (buf.find("4R")!=buf.npos)
	subsector=2;
    }
  }else{
  // Extract the Ring 
    {
      std::stringstream os;
      os<<buf.substr(buf.find("_R0")+3,1);
      os>>ring;
    }
  //Extract the disk (named station)
    {
      std::stringstream os;
      os<<buf.substr(2,1);
      os>>station;
    }
    //Extract the sector or chamber
    {
      std::stringstream os;
      os<<buf.substr(buf.find("_C")+2,2);
      os>>sector;
    }
    //Extract layer
    {
      if (buf.find("UP")!=buf.npos)
	layer=1;
      if (buf.find("DW")!=buf.npos)
	layer=2;
    }
  }
  this->init(region,ring,station,sector,layer,subsector); 
}

std::string
RPCCompDetId::gasDBname() const{
  std::stringstream os;
  if(this->region()==0){
    // Barrel
    std::string wsign="P";
    if (this->wheel()<0)wsign= "M";
    std::string lr="";
    if (this->subsector()==1) lr="L";
    if (this->subsector()==2) lr="R";
    os<<"W"<<wsign<<abs(this->wheel())<<"_S"<<std::setw(2)<<std::setfill('0')<<this->sector()<<"_RB"<<this->station()<<lr;
  } else {
    // Endcap
    std::string esign="P";
    if (this->region()<0)
      esign="M";

    os<<"E"<<esign<<this->disk();


    if (this->disk()==1){
      os<<"_R"<<std::setw(2)<<std::setfill('0')<<this->ring()
	<<"_C"<<std::setw(2)<<std::setfill('0')<<this->sector()
	<<"_C"<<std::setw(2)<<std::setfill('0')<<this->sector()+5;
    }else{
      os<<"_R"<<std::setw(2)<<std::setfill('0')<<this->ring()
	<<"_R"<<std::setw(2)<<std::setfill('0')<<this->ring()+1
	<<"_C"<<std::setw(2)<<std::setfill('0')<<this->sector()
	<<"_C"<<std::setw(2)<<std::setfill('0')<<this->sector()+2;
     
    }
    std::string lay="";
    if(this->layer()==1)
      lay="UP";
    else if (this->layer()==2)
      lay="DW";

    os<<"_"<<lay;
      
 }
  return os.str();
}

std::ostream& operator<<( std::ostream& os, const RPCCompDetId& id ){

  os <<id.dbname();

  return os;
}






