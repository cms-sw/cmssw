#ifndef PixelCaenChannels_H
#define PixelCaenChannels_H

/** \class PixelCaenChannels
 *
 *  Helper class to get digital LV, analog LV and biased HV channels from Pixel DCS object
 *  for a given DetId.
 *
 *  $Date: 2008/11/30 19:41:08 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

class CaenChannel;
class DetId;

template<class> class PixelDCSObject;

class PixelCaenChannels
{
  public:

  /// Get digital LV, analog LV, biased HV channels from DCS object for a given DetId.
  PixelCaenChannels(
                    const PixelDCSObject<CaenChannel>&,
                    const DetId&
                    );

  const CaenChannel& digitalLV() const { return *theDigitalLV; }
  const CaenChannel& analogLV() const { return *theAnalogLV; }
  const CaenChannel& biasedHV() const { return *theBiasedHV; }

  private:

  const CaenChannel* theDigitalLV; // digital LV channel in PVSS
  const CaenChannel* theAnalogLV;  // analog LV channel in PVSS
  const CaenChannel* theBiasedHV;  // biased HV channel in PVSS
};

#endif
