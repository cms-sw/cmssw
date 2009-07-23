
/*
 *   File: DataFormats/Scalers/src/LumiScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include <cstdio>
#include <ostream>

LumiScalers::LumiScalers() : 
   trigType_(0),
   eventID_(0),
   sourceID_(0),
   bunchNumber_(0),
   version_(0),
   normalization_(0.0),
   deadTimeNormalization_(0.0),
   lumiFill_(0.0),
   lumiRun_(0.0),
   liveLumiFill_(0.0),
   liveLumiRun_(0.0),
   instantLumi_(0.0),
   instantLumiErr_(0.0),
   instantLumiQlty_(0),
   lumiETFill_(0.0),
   lumiETRun_(0.0),
   liveLumiETFill_(0.0),
   liveLumiETRun_(0.0),
   instantETLumi_(0.0),
   instantETLumiErr_(0.0),
   instantETLumiQlty_(0),
   lumiOccFill_(nOcc),
   lumiOccRun_(nOcc),
   liveLumiOccFill_(nOcc),
   liveLumiOccRun_(nOcc),
   instantOccLumi_(nOcc),
   instantOccLumiErr_(nOcc),
   instantOccLumiQlty_(nOcc),
   lumiNoise_(nOcc),
   sectionNumber_(0),
   startOrbit_(0),
   numOrbits_(0)
{ 
}

LumiScalers::LumiScalers(const unsigned char * rawData)
{ 
  LumiScalers();

  struct ScalersEventRecordRaw_v1 * raw 
    = (struct ScalersEventRecordRaw_v1 *)rawData;
  trigType_     = ( raw->header >> 56 ) &        0xFULL;
  eventID_      = ( raw->header >> 32 ) & 0x00FFFFFFULL;
  sourceID_     = ( raw->header >>  8 ) & 0x00000FFFULL;
  bunchNumber_  = ( raw->header >> 20 ) &      0xFFFULL;

  version_ = raw->version;
  if ( version_ >= 1 )
  {
    collectionTime_.set_tv_sec(static_cast<long>(raw->lumi.collectionTime_sec));
    collectionTime_.set_tv_nsec(raw->lumi.collectionTime_nsec);
    deadTimeNormalization_  = raw->lumi.DeadtimeNormalization;
    normalization_          = raw->lumi.Normalization;
    lumiFill_               = raw->lumi.LumiFill;
    lumiRun_                = raw->lumi.LumiRun;
    liveLumiFill_           = raw->lumi.LiveLumiFill;
    liveLumiRun_            = raw->lumi.LiveLumiRun;
    instantLumi_            = raw->lumi.InstantLumi;
    instantLumiErr_         = raw->lumi.InstantLumiErr;
    instantLumiQlty_        = raw->lumi.InstantLumiQlty;
    lumiETFill_             = raw->lumi.LumiETFill;
    lumiETRun_              = raw->lumi.LumiETRun;
    liveLumiETFill_         = raw->lumi.LiveLumiETFill;
    liveLumiETRun_          = raw->lumi.LiveLumiETRun;
    instantETLumi_          = raw->lumi.InstantETLumi;
    instantETLumiErr_       = raw->lumi.InstantETLumiErr;
    instantETLumiQlty_      = raw->lumi.InstantETLumiQlty;
    for ( int i=0; i<ScalersRaw::N_LUMI_OCC_v1; i++)
    {
      lumiOccFill_.push_back(raw->lumi.LumiOccFill[i]);
      lumiOccRun_.push_back(raw->lumi.LumiOccRun[i]);
      liveLumiOccFill_.push_back(raw->lumi.LiveLumiOccFill[i]);
      liveLumiOccRun_.push_back(raw->lumi.LiveLumiOccRun[i]);
      instantOccLumi_.push_back(raw->lumi.InstantOccLumi[i]);
      instantOccLumiErr_.push_back(raw->lumi.InstantOccLumiErr[i]);
      instantOccLumiQlty_.push_back(raw->lumi.InstantOccLumiQlty[i]);
      lumiNoise_.push_back(raw->lumi.lumiNoise[i]);
    }
    sectionNumber_ = raw->lumi.sectionNumber;
    startOrbit_    = raw->lumi.startOrbit;
    numOrbits_     = raw->lumi.numOrbits;
  }
}

LumiScalers::~LumiScalers() { } 


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c) 
{
  char zeit[128];
  char line[128];
  struct tm * hora;

  s << "LumiScalers    Version: " << c.version() << 
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

  sprintf(line," SectionNumber: %10d   StartOrbit: %10d  NumOrbits: %10d",
	  c.sectionNumber(), c.startOrbit(), c.numOrbits());
  s << line << std::endl;

  sprintf(line," Normalization: %e  DeadTimeNormalization: %e",
	  c.normalization(), c.deadTimeNormalization());
  s << line << std::endl;

  // Integrated Luminosity

  sprintf(line," LumiFill:            %e   LumiRun:            %e", 
	  c.lumiFill(), c.lumiRun());
  s << line << std::endl;
  sprintf(line," LiveLumiFill:        %e   LiveLumiRun:        %e", 
	  c.liveLumiFill(), c.liveLumiRun());
  s << line << std::endl;

  sprintf(line," LumiETFill:          %e   LumiETRun:          %e", 
	  c.lumiFill(), c.lumiRun());
  s << line << std::endl;

  sprintf(line," LiveLumiETFill:      %e   LiveLumETiRun:      %e", 
	  c.liveLumiETFill(), c.liveLumiETRun());
  s << line << std::endl;

  int length = c.instantOccLumi().size();
  for (int i=0; i<length; i++)
  {
    sprintf(line,
	       " LumiOccFill[%d]:      %e   LumiOccRun[%d]:      %e", 
	    i, c.lumiOccFill()[i], i, c.lumiOccRun()[i]);
    s << line << std::endl;

    sprintf(line,
	       " LiveLumiOccFill[%d]:  %e   LiveLumiOccRun[%d]:v %e", 
	    i, c.liveLumiOccFill()[i], i, c.liveLumiOccRun()[i]);
    s << line << std::endl;
  }

  // Instantaneous Luminosity

  sprintf(line," InstantLumi:       %e  Err: %e  Qlty: %d",
	  c.instantLumi(), c.instantLumiErr(), c.instantLumiQlty());
  s << line << std::endl;

  sprintf(line," InstantETLumi:     %e  Err: %e  Qlty: %d",
	  c.instantETLumi(), c.instantETLumiErr(), c.instantETLumiQlty());
  s << line << std::endl;

  for (int i=0; i<length; i++)
  {
    sprintf(line," InstantOccLumi[%d]: %e  Err: %e  Qlty: %d",
	    i, c.instantOccLumi()[i], c.instantOccLumiErr()[i], 
	    c.instantOccLumiQlty()[i]);
    s << line << std::endl;
    sprintf(line,"      LumiNoise[%d]: %e", i, c.lumiNoise()[i]);
    s << line << std::endl;
  }

  return s;
}
