#ifndef SiStripLorentzAngle_h
#define SiStripLorentzAngle_h

#include <vector>
#include <map>
#include <iostream>
#include <boost/cstdint.hpp>
// #include "CondFormats/SiStripObjects/interface/SiStripBaseObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

/**
 * Stores the lorentz angle value for all DetIds. <br>
 * The values are saved internally in a std::map<detid, lorentzAngle>. <br>
 * It can be filled either by the complete map (putLorentzAngles) or passing
 * a single detIds and lorentzAngles (putLorentzAngle). <br>
 * In the same way getLorentzAngles returns the complete map, while getLorentzAngle
 * the value corresponding to a given DetId. <br>
 * The printDebug method prints LorentzAngles for all detIds. <br>
 * The printSummary mehtod uses the SiStripDetSummary class to produce a summary
 * of LorentzAngle values divided by subdetector and layer/disk.
 */

// class SiStripLorentzAngle : public SiStripBaseObject
class SiStripLorentzAngle
{
public:
  SiStripLorentzAngle(){};
  ~SiStripLorentzAngle(){};

  inline void putLorentsAngles(std::map<unsigned int,float>& LA){m_LA=LA;}   
  inline const std::map<unsigned int,float>&  getLorentzAngles () const {return m_LA;}

  bool putLorentzAngle(const uint32_t&, float);
  float getLorentzAngle (const uint32_t&) const;

  /// Prints LorentzAngles for all detIds.
  void printDebug(std::stringstream& ss) const;
  /// Prints the mean value of the LorentzAngle divided by subdetector, layer and mono/stereo.
  void printSummary(std::stringstream& ss) const;

private:
  std::map<unsigned int,float> m_LA; 
};

#endif
