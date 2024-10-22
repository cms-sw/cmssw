#ifndef DIGIHCAL_HCALQIESAMPLE_H
#define DIGIHCAL_HCALQIESAMPLE_H

#include <ostream>
#include <cstdint>

static
#ifdef __CUDA_ARCH__
    __constant__
#else
    constexpr
#endif
    float nominal_adc2fc[128] = {
        -0.5f,   0.5f,    1.5f,    2.5f,    3.5f,    4.5f,    5.5f,    6.5f,    7.5f,    8.5f,    9.5f,    10.5f,
        11.5f,   12.5f,   13.5f,   15.0f,   17.0f,   19.0f,   21.0f,   23.0f,   25.0f,   27.0f,   29.5f,   32.5f,
        35.5f,   38.5f,   42.0f,   46.0f,   50.0f,   54.5f,   59.5f,   64.5f,   59.5f,   64.5f,   69.5f,   74.5f,
        79.5f,   84.5f,   89.5f,   94.5f,   99.5f,   104.5f,  109.5f,  114.5f,  119.5f,  124.5f,  129.5f,  137.0f,
        147.0f,  157.0f,  167.0f,  177.0f,  187.0f,  197.0f,  209.5f,  224.5f,  239.5f,  254.5f,  272.0f,  292.0f,
        312.0f,  334.5f,  359.5f,  384.5f,  359.5f,  384.5f,  409.5f,  434.5f,  459.5f,  484.5f,  509.5f,  534.5f,
        559.5f,  584.5f,  609.5f,  634.5f,  659.5f,  684.5f,  709.5f,  747.0f,  797.0f,  847.0f,  897.0f,  947.0f,
        997.0f,  1047.0f, 1109.5f, 1184.5f, 1259.5f, 1334.5f, 1422.0f, 1522.0f, 1622.0f, 1734.5f, 1859.5f, 1984.5f,
        1859.5f, 1984.5f, 2109.5f, 2234.5f, 2359.5f, 2484.5f, 2609.5f, 2734.5f, 2859.5f, 2984.5f, 3109.5f, 3234.5f,
        3359.5f, 3484.5f, 3609.5f, 3797.0f, 4047.0f, 4297.0f, 4547.0f, 4797.0f, 5047.0f, 5297.0f, 5609.5f, 5984.5f,
        6359.5f, 6734.5f, 7172.0f, 7672.0f, 8172.0f, 8734.5f, 9359.5f, 9984.5f};

/** \class HcalQIESample
 *  Simple container packer/unpacker for a single QIE data word
 *
 *
 *  \author J. Mans - Minnesota
 */
class HcalQIESample {
public:
  constexpr HcalQIESample() : theSample{0} {}
  constexpr HcalQIESample(uint16_t data) : theSample{data} {}
  constexpr HcalQIESample(int adc, int capid, int fiber, int fiberchan, bool dv = true, bool er = false)
      : theSample((uint16_t)((adc & 0x7f) | ((capid & 0x3) << 7) | (((fiber - 1) & 0x7) << 13) |
                             ((fiberchan & 0x3) << 11) | ((dv) ? (0x0200) : (0)) | ((er) ? (0x0400) : (0)))) {}

  /// get the raw word
  constexpr uint16_t raw() const { return theSample; }
  /// get the ADC sample
  constexpr int adc() const { return theSample & 0x7F; }
  /// get the nominal FC (no calibrations applied)
  constexpr double nominal_fC() const { return nominal_adc2fc[adc()]; }
  /// get the Capacitor id
  constexpr int capid() const { return (theSample >> 7) & 0x3; }
  /// is the Data Valid bit set?
  constexpr bool dv() const { return (theSample & 0x0200) != 0; }
  /// is the error bit set?
  constexpr bool er() const { return (theSample & 0x0400) != 0; }
  /// get the fiber number
  constexpr int fiber() const { return ((theSample >> 13) & 0x7) + 1; }
  /// get the fiber channel number
  constexpr int fiberChan() const { return (theSample >> 11) & 0x3; }
  /// get the id channel
  constexpr int fiberAndChan() const { return (theSample >> 11) & 0x1F; }

  /// for streaming
  constexpr uint16_t operator()() { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const HcalQIESample&);

#endif
