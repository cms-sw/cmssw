/*
 *   File: DataFormats/Scalers/src/DcsStatus.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include <cstdio>
#include <ostream>

const int DcsStatus::partitionList[DcsStatus::nPartitions] = {
  EBp         ,
  EBm         ,
  EEp         ,
  EEm         ,
  HBHEa       ,
  HBHEb       ,
  HBHEc       ,
  HF          ,
  HO          ,
  RPC         ,
  DT0         ,
  DTp         ,
  DTm         ,
  CSCp        ,
  CSCm        ,
  CASTOR      ,
  ZDC         ,
  TIBTID      ,
  TOB         ,
  TECp        ,
  TECm        ,
  BPIX        ,
  FPIX        ,
  ESp         ,
  ESm };

const char * DcsStatus::partitionName[DcsStatus::nPartitions] = {
  "EBp"         ,
  "EBm"         ,
  "EEp"         ,
  "EEm"         ,
  "HBHEa"       ,
  "HBHEb"       ,
  "HBHEc"       ,
  "HF"          ,
  "HO"          ,
  "RPC"         ,
  "DT0"         ,
  "DTp"         ,
  "DTm"         ,
  "CSCp"        ,
  "CSCm"        ,
  "CASTOR"      ,
  "ZDC"         ,
  "TIBTID"      ,
  "TOB"         ,
  "TECp"        ,
  "TECm"        ,
  "BPIX"        ,
  "FPIX"        ,
  "ESp"         ,
  "ESm" };

DcsStatus::DcsStatus() : 
   trigType_(0),
   eventID_(0),
   sourceID_(0),
   bunchNumber_(0),
   version_(0),
   collectionTime_(0,0),
   ready_(0),
   magnetCurrent_((float)0.0),
   magnetTemperature_((float)0.0)
{ 
}

DcsStatus::DcsStatus(const unsigned char * rawData)
{ 
  DcsStatus();

  struct ScalersEventRecordRaw_v4 * raw 
    = (struct ScalersEventRecordRaw_v4 *)rawData;
  trigType_     = ( raw->header >> 56 ) &        0xFULL;
  eventID_      = ( raw->header >> 32 ) & 0x00FFFFFFULL;
  sourceID_     = ( raw->header >>  8 ) & 0x00000FFFULL;
  bunchNumber_  = ( raw->header >> 20 ) &      0xFFFULL;

  version_ = raw->version;
  if ( version_ >= 4 )
  {
    collectionTime_.set_tv_sec(static_cast<long>(raw->dcsStatus.collectionTime_sec));
    collectionTime_.set_tv_nsec(raw->dcsStatus.collectionTime_nsec);
    ready_             = raw->dcsStatus.ready;
    magnetCurrent_     = raw->dcsStatus.magnetCurrent;
    magnetTemperature_ = raw->dcsStatus.magnetTemperature;
  }
}

DcsStatus::~DcsStatus() { } 


/// Pretty-print operator for DcsStatus
std::ostream& operator<<(std::ostream& s, const DcsStatus& c) 
{
  char zeit[128];
  char line[128];
  struct tm * hora;

  s << "DcsStatus    Version: " << c.version() << 
    "   SourceID: "<< c.sourceID() << std::endl;

  timespec ts = c.collectionTime();
  hora = gmtime(&ts.tv_sec);
  strftime(zeit, sizeof(zeit), "%Y.%m.%d %H:%M:%S", hora);
  sprintf(line, " CollectionTime: %s.%9.9d", zeit, 
	  (int)ts.tv_nsec);
  s << line << std::endl;

  sprintf(line, " TrigType: %d   EventID: %d    BunchNumber: %d", 
	  c.trigType(), c.eventID(), c.bunchNumber());
  s << line << std::endl;

  sprintf(line," MagnetCurrent: %e    MagnetTemperature: %e", 
	  c.magnetCurrent(), c.magnetTemperature());
  s << line << std::endl;

  sprintf(line," Ready: %d  0x%8.8X", c.ready(), c.ready());
  s << line << std::endl;

  for ( int i=0; i<DcsStatus::nPartitions; i++)
  {
    if ( c.ready(DcsStatus::partitionList[i]))
    {
      sprintf(line,"  %2d %6s: READY", i, DcsStatus::partitionName[i]);
    }
    else
    {
      sprintf(line,"  %2d %6s: NOT READY", i, DcsStatus::partitionName[i]);
    }
  s << line << std::endl;
  }
  return s;
}
