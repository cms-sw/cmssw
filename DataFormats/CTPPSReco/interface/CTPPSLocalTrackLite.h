/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSLocalTrackLite
#define DataFormats_CTPPSReco_CTPPSLocalTrackLite

/**
 *\brief Local (=single RP) track with essential information only.
 **/
class CTPPSLocalTrackLite
{
  public:
    CTPPSLocalTrackLite() : rpId(0), x(0.), x_unc(-1.), y(0.), y_unc(-1.), time(0.), time_unc(-1.)
    {
    }

    CTPPSLocalTrackLite(uint32_t pid, float px, float pxu, float py, float pyu, float pt=0., float ptu=-1.)
      : rpId(pid), x(px), x_unc(pxu), y(py), y_unc(pyu), time(pt), time_unc(ptu)
    {
    }

    /// returns the RP id
    uint32_t getRPId() const
    {
      return rpId;
    }

    /// returns the horizontal track position
    float getX() const
    {
      return x;
    }

    /// returns the horizontal track position uncertainty
    float getXUnc() const
    {
      return x_unc;
    }

    /// returns the vertical track position
    float getY() const
    {
      return y;
    }

    /// returns the vertical track position uncertainty
    float getYUnc() const
    {
      return y_unc;
    }

    /// returns the track time
    float getTime() const
    {
      return time;
    }

    /// returns the track time uncertainty
    float getTimeUnc() const
    {
      return time_unc;
    }

  protected:
    /// RP id
    uint32_t rpId;

    /// horizontal hit position and uncertainty, mm
    float x, x_unc;

    /// vertical hit position and uncertainty, mm
    float y, y_unc;

    /// time information and uncertainty
    float time, time_unc;
};

#endif
