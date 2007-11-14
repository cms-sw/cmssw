/*
 *  File: DataFormats/Scalers/interface/LumiScalers.h   (W.Badgett)
 *
 *  Various Luminosity Scalers from the HF Lumi System
 *
 */

#ifndef LUMISCALERS_H
#define LUMISCALERS_H

#include <ostream>
#include <vector>

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

  int version()                      { return(version_);}
  double normalization()             { return(normalization_);}
  double instantLumi()               { return(instantLumi_);}
  double instantLumiErr()            { return(instantLumiErr_);}
  double instantLumiQlty()           { return(instantLumiQlty_);}
  double instantETLumi()             { return(instantETLumi_);}
  double instantETLumiErr()          { return(instantETLumiErr_);}
  double instantETLumiQlty()         { return(instantETLumiQlty_);}
  std::vector<double> instantOccLumi()      { return(instantOccLumi_);}
  std::vector<double> instantOccLumiErr()   { return(instantOccLumiErr_);}
  std::vector<double> instantOccLumiQlty()  { return(instantOccLumiQlty_);}
  std::vector<double> lumiNoise()           { return(lumiNoise_);}
  unsigned int sectionNumber()       { return(sectionNumber_);}
  unsigned int startOrbit()          { return(startOrbit_);}
  unsigned int numOrbits()           { return(numOrbits_);}

  /// equality operator
  int operator==(const LumiScalers& e) const { return false; }

  /// inequality operator
  int operator!=(const LumiScalers& e) const { return false; }

protected:

  int version_;
  double normalization_;
  double instantLumi_;
  double instantLumiErr_;
  double instantLumiQlty_;
  double instantETLumi_;
  double instantETLumiErr_;
  double instantETLumiQlty_;
  std::vector<double> instantOccLumi_;
  std::vector<double> instantOccLumiErr_;
  std::vector<double> instantOccLumiQlty_;
  std::vector<double> lumiNoise_;
  unsigned int sectionNumber_;
  unsigned int startOrbit_;
  unsigned int numOrbits_;
};


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c);

typedef std::vector<LumiScalers> LumiScalersCollection;

#endif
