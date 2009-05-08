#ifndef PixelCaenAlias_H
#define PixelCaenAlias_H

/** \class PixelCaenAlias
 *
 *  Base class to get the PVSS aliases of digital LV, analog LV and biased HV channels
 *  for a given DetId in the Pixel Detector.
 *
 *  $Date: 2009/04/22 11:42:41 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include <string>

class PixelCaenAlias
{
  public:

  const std::string& digitalLV() const { return theDigitalLV; }
  const std::string& analogLV() const { return theAnalogLV; }
  const std::string& biasedHV() const { return theBiasedHV; }

  virtual ~PixelCaenAlias() {}

  protected:

  std::string theDigitalLV; // alias of digital LV channel in PVSS
  std::string theAnalogLV;  // alias of analog LV channel in PVSS
  std::string theBiasedHV;  // alias of biased HV channel in PVSS
};

#endif
