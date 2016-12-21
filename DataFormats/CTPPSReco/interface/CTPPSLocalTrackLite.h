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
struct CTPPSLocalTrackLite
{
  public:
    CTPPSLocalTrackLite(uint32_t pid=0, float px=0., float pxu=-1., float py=0., float pyu=-1., float pt=0., float ptu=-1.)
      : rpId(pid), x(px), x_unc(pxu), y(py), y_unc(pyu), time(pt), time_unc(ptu)
    {
    }

    uint32_t getRPId() const
    {
      return rpId;
    }

    float getX() const
    {
      return x;
    }

    float getXUnc() const
    {
      return x_unc;
    }

    float getY() const
    {
      return y;
    }

    float getYUnc() const
    {
      return y_unc;
    }

    float getTime() const
    {
      return time;
    }

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
