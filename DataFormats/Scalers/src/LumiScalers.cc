
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
   numOrbits_(0),
   pileup_(0.0),
   pileupRMS_(0.0)
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
  version_      = raw->version;

  struct LumiScalersRaw_v1 * lumi = NULL;

  if ( version_ >= 1 )
  {
    if ( version_ <= 2 )
    {
      lumi = & (raw->lumi);
    }
    else 
    {
      struct ScalersEventRecordRaw_v3 * raw3 
	= (struct ScalersEventRecordRaw_v3 *)rawData;
      lumi = & (raw3->lumi);
    }
    collectionTime_.set_tv_sec(static_cast<long>(lumi->collectionTime_sec));
    collectionTime_.set_tv_nsec(lumi->collectionTime_nsec);
    deadTimeNormalization_  = lumi->DeadtimeNormalization;
    normalization_          = lumi->Normalization;
    lumiFill_               = lumi->LumiFill;
    lumiRun_                = lumi->LumiRun;
    liveLumiFill_           = lumi->LiveLumiFill;
    liveLumiRun_            = lumi->LiveLumiRun;
    instantLumi_            = lumi->InstantLumi;
    instantLumiErr_         = lumi->InstantLumiErr;
    instantLumiQlty_        = lumi->InstantLumiQlty;
    lumiETFill_             = lumi->LumiETFill;
    lumiETRun_              = lumi->LumiETRun;
    liveLumiETFill_         = lumi->LiveLumiETFill;
    liveLumiETRun_          = lumi->LiveLumiETRun;
    instantETLumi_          = lumi->InstantETLumi;
    instantETLumiErr_       = lumi->InstantETLumiErr;
    instantETLumiQlty_      = lumi->InstantETLumiQlty;
    for ( int i=0; i<ScalersRaw::N_LUMI_OCC_v1; i++)
    {
      lumiOccFill_.push_back(lumi->LumiOccFill[i]);
      lumiOccRun_.push_back(lumi->LumiOccRun[i]);
      liveLumiOccFill_.push_back(lumi->LiveLumiOccFill[i]);
      liveLumiOccRun_.push_back(lumi->LiveLumiOccRun[i]);
      instantOccLumi_.push_back(lumi->InstantOccLumi[i]);
      instantOccLumiErr_.push_back(lumi->InstantOccLumiErr[i]);
      instantOccLumiQlty_.push_back(lumi->InstantOccLumiQlty[i]);
      lumiNoise_.push_back(lumi->lumiNoise[i]);
    }
    sectionNumber_ = lumi->sectionNumber;
    startOrbit_    = lumi->startOrbit;
    numOrbits_     = lumi->numOrbits;
    if ( version_ >= 7 )
    {
      struct ScalersEventRecordRaw_v6 * raw6 
	= (struct ScalersEventRecordRaw_v6 *)rawData;
      float * fspare = (float *) raw6->spare;
      pileup_    = fspare[ScalersRaw::I_SPARE_PILEUP_v7];
      pileupRMS_ = fspare[ScalersRaw::I_SPARE_PILEUPRMS_v7];
      if ( version_ >= 7 )
      {
	bunchLumi_ = fspare[ScalersRaw::I_SPARE_BUNCHLUMI_v8];
      }
      else
      {
	bunchLumi_ = (float)0.0;
      }
    }
    else
    {
      pileup_    = (float)0.0;
      pileupRMS_ = (float)0.0;
      bunchLumi_ = (float)0.0;
    }
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
	       " LiveLumiOccFill[%d]:  %e   LiveLumiOccRun[%d]:  %e", 
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

  sprintf(line," Pileup: %f   PileupRMS: %f   BunchLumi: %f", 
	  c.pileup(), c.pileupRMS(), c.bunchLumi());
  s << line << std::endl;

  return s;
}
