/*
 *  File: DataFormats/Scalers/interface/LumiScalers.h   (W.Badgett)
 *
 *  Various Luminosity Scalers from the HF Lumi System
 *
 */

#ifndef LUMISCALERS_H
#define LUMISCALERS_H

#include <ostream>


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
  LumiScalers(uint16_t rawData);
  virtual ~LumiScalers();

  enum
  {
    nOcc = 2
  };

  /// name method
  std::string name() const { return "LumiScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  double getNormalization()             { return(normalization);}
  int getVersion()                      { return(version);}
  double getNormalization()             { return(normalization);}
  double getInstantLumi()               { return(instantLumi);}
  double getInstantLumiErr()            { return(instantLumiErr);}
  double getInstantLumiQlty()           { return(instantLumiQlty);}
  double getInstantETLumi()             { return(instantETLumi);}
  double getInstantETLumiErr()          { return(instantETLumiErr);}
  double getInstantETLumiQlty()         { return(instantETLumiQlty);}
  double [nOcc] getInstantOccLumi()     { return(instantOccLumi);}
  double [nOcc] getInstantOccLumiErr()  { return(instantOccLumiErr);}
  double [nOcc] getInstantOccLumiQlty() { return(instantOccLumiQlty);}
  double [nOcc] getLumiNoise()          { return(lumiNoise);}
  unsigned int getSectionNumber()       { return(sectionNumber);}
  unsigned int getStartOrbit()          { return(startOrbit);}
  unsigned int getNumOrbits()           { return(numOrbits);}

  /// equality operator
  int operator==(const LumiScalers& e) const { return m_data==e.raw(); }

  /// inequality operator
  int operator!=(const LumiScalers& e) const { return m_data!=e.raw(); }

protected:

  uint16_t m_data;
  int version;
  double normalization;
  double instantLumi;
  double instantLumiErr;
  double instantLumiQlty;
  double instantETLumi;
  double instantETLumiErr;
  double instantETLumiQlty;
  double instantOccLumi[nOcc];
  double instantOccLumiErr[nOcc];
  double instantOccLumiQlty[nOcc];
  double lumiNoise[nOcc];
  unsigned int sectionNumber;
  unsigned int startOrbit;
  unsigned int numOrbits;
};


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c);


#endif
