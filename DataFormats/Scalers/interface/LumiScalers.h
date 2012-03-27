/*
 *  File: DataFormats/Scalers/interface/LumiScalers.h   (W.Badgett)
 *
 *  Various Luminosity Scalers from the HF Lumi System
 *
 */

#ifndef DATAFORMATS_SCALERS_LUMISCALERS_H
#define DATAFORMATS_SCALERS_LUMISCALERS_H

#include "DataFormats/Scalers/interface/TimeSpec.h"

#include <ctime>
#include <iosfwd>
#include <vector>
#include <string>

/*! \file LumiScalers.h
 * \Header file for HF Lumi Scalers
 * 
 * \author: William Badgett
 *
 */


/// \class LumiScalers.h
/// \brief Persistable copy of HF Lumi Scalers

class LumiScalers
{
 public:

  LumiScalers();
  LumiScalers(const unsigned char * rawData);
  virtual ~LumiScalers();

  enum
  {
    nOcc = 2
  };

  /// name method
  std::string name() const { return "LumiScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  unsigned int trigType() const            { return(trigType_);}
  unsigned int eventID() const             { return(eventID_);}
  unsigned int sourceID() const            { return(sourceID_);}
  unsigned int bunchNumber() const         { return(bunchNumber_);}

  int version() const                      { return(version_);}
  timespec collectionTime() const          { return(collectionTime_.get_timespec());}
  float normalization() const              { return(normalization_);}
  float deadTimeNormalization() const  
  { return(deadTimeNormalization_);}

  float lumiFill() const                   { return(lumiFill_);}
  float lumiRun() const                    { return(lumiRun_);}
  float liveLumiFill() const               { return(liveLumiFill_);}
  float liveLumiRun() const                { return(liveLumiRun_);}
  float instantLumi() const                { return(instantLumi_);}
  float instantLumiErr() const             { return(instantLumiErr_);}
  unsigned char instantLumiQlty() const    { return(instantLumiQlty_);}

  float lumiETFill() const                 { return(lumiETFill_);}
  float lumiETRun() const                  { return(lumiETRun_);}
  float liveLumiETFill() const             { return(liveLumiETFill_);}
  float liveLumiETRun() const              { return(liveLumiETRun_);}
  float instantETLumi() const              { return(instantETLumi_);}
  float instantETLumiErr() const           { return(instantETLumiErr_);}
  unsigned char instantETLumiQlty() const  { return(instantETLumiQlty_);}

  std::vector<float> lumiOccFill() const      
  { return(lumiOccFill_);}
  std::vector<float> lumiOccRun() const      
  { return(lumiOccRun_);}
  std::vector<float> liveLumiOccFill() const      
  { return(liveLumiOccFill_);}
  std::vector<float> liveLumiOccRun() const      
  { return(liveLumiOccRun_);}
  std::vector<float> instantOccLumi() const      
  { return(instantOccLumi_);}
  std::vector<float> instantOccLumiErr() const   
  { return(instantOccLumiErr_);}
  std::vector<unsigned char> instantOccLumiQlty() const  
  { return(instantOccLumiQlty_);}
  std::vector<float> lumiNoise() const    { return(lumiNoise_);}

  unsigned int sectionNumber() const       { return(sectionNumber_);}
  unsigned int startOrbit() const          { return(startOrbit_);}
  unsigned int numOrbits() const           { return(numOrbits_);}

  float pileup() const                     { return(pileup_);}
  float spare() const                      { return(spare_);}

  /// equality operator
  int operator==(const LumiScalers& e) const { return false; }

  /// inequality operator
  int operator!=(const LumiScalers& e) const { return false; }

protected:

  unsigned int trigType_;
  unsigned int eventID_;
  unsigned int sourceID_;
  unsigned int bunchNumber_;

  int version_;

  TimeSpec collectionTime_;

  float normalization_;
  float deadTimeNormalization_;

  float lumiFill_;
  float lumiRun_;
  float liveLumiFill_;
  float liveLumiRun_;
  float instantLumi_;
  float instantLumiErr_;
  unsigned char instantLumiQlty_;

  float lumiETFill_;
  float lumiETRun_;
  float liveLumiETFill_;
  float liveLumiETRun_;
  float instantETLumi_;
  float instantETLumiErr_;

  unsigned char instantETLumiQlty_;

  std::vector<float> lumiOccFill_;
  std::vector<float> lumiOccRun_;
  std::vector<float> liveLumiOccFill_;
  std::vector<float> liveLumiOccRun_;
  std::vector<float> instantOccLumi_;
  std::vector<float> instantOccLumiErr_;
  std::vector<unsigned char> instantOccLumiQlty_;
  std::vector<float> lumiNoise_;

  unsigned int sectionNumber_;
  unsigned int startOrbit_;
  unsigned int numOrbits_;

  float pileup_;
  float spare_;
};


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c);

typedef std::vector<LumiScalers> LumiScalersCollection;

#endif
