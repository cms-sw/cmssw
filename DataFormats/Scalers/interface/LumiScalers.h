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

  int version() const                      { return(version_);}
  double normalization() const             { return(normalization_);}
  double instantLumi() const               { return(instantLumi_);}
  double instantLumiErr() const            { return(instantLumiErr_);}
  double instantLumiQlty() const           { return(instantLumiQlty_);}
  double instantETLumi() const             { return(instantETLumi_);}
  double instantETLumiErr() const          { return(instantETLumiErr_);}
  double instantETLumiQlty() const         { return(instantETLumiQlty_);}
  std::vector<double> instantOccLumi() const      
  { return(instantOccLumi_);}
  std::vector<double> instantOccLumiErr() const   
  { return(instantOccLumiErr_);}
  std::vector<double> instantOccLumiQlty() const  
  { return(instantOccLumiQlty_);}
  std::vector<double> lumiNoise() const    { return(lumiNoise_);}
  unsigned int sectionNumber() const       { return(sectionNumber_);}
  unsigned int startOrbit() const          { return(startOrbit_);}
  unsigned int numOrbits() const           { return(numOrbits_);}

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
