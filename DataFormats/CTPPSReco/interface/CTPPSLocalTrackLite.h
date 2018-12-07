/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSLocalTrackLite
#define DataFormats_CTPPSReco_CTPPSLocalTrackLite

#include <cstdint>

/**
 *\brief Local (=single RP) track with essential information only.
 **/
class CTPPSLocalTrackLite
{
  public:
    CTPPSLocalTrackLite() : rpId(0), x(0.), x_unc(-1.), y(0.), y_unc(-1.), tx(999.), tx_unc(-1.), ty(999.), ty_unc(-1.), chiSquaredOverNDF(-1.), reco_info(4), numberOfPointUsedForFit(0), time(0.), time_unc(-1.)
    {
    }

    CTPPSLocalTrackLite(uint32_t pid, float px, float pxu, float py, float pyu, float ptx=999., float ptxu=-1., float pty=999., float ptyu=-1., float pchiSquaredOverNDF=-1., unsigned short preco_info=4, unsigned short pNumberOfPointUsedForFit=-1, float pt=0., float ptu=-1.)
      : rpId(pid), x(px), x_unc(pxu), y(py), y_unc(pyu), tx(ptx), tx_unc(ptxu), ty(pty), ty_unc(ptyu), chiSquaredOverNDF(pchiSquaredOverNDF), reco_info(preco_info), numberOfPointUsedForFit(pNumberOfPointUsedForFit), time(pt), time_unc(ptu)
    { 
    }

    /// returns the RP id
    inline uint32_t getRPId() const
    {
      return rpId;
    }

    /// returns the horizontal track position
    inline float getX() const
    {
      return x;
    }

    /// returns the horizontal track position uncertainty
    inline float getXUnc() const
    {
      return x_unc;
    }

    /// returns the vertical track position
    inline float getY() const
    {
      return y;
    }

    /// returns the vertical track position uncertainty
    inline float getYUnc() const
    {
      return y_unc;
    }

    /// returns the track time
    inline float getTime() const
    {
      return time;
    }

    /// returns the track time uncertainty
    inline float getTimeUnc() const
    {
      return time_unc;
    }

    /// returns the track horizontal angle
    inline float getTx() const
    {
        return tx;
    }

    /// returns the track horizontal angle uncertainty
    inline float getTxUnc() const
    {
        return tx_unc;
    }
    
    /// returns the track vertical angle
    inline float getTy() const
    {
        return ty;
    }
    
    /// returns the track vertical angle uncertainty
    inline float getTyUnc() const
    {
        return ty_unc;
    }
    
    /// returns the track fit chi Squared over NDF
    inline float getChiSquaredOverNDF() const
    {
        return chiSquaredOverNDF;
    }

    /// returns the track reconstruction info byte
    inline unsigned int getReco_info() const
    {
        return reco_info;
    }

    /// returns the number of points used for fit
    inline unsigned short getNumberOfPointsUsedForFit() const
    {
        return numberOfPointUsedForFit;
    }   

  protected:
    /// RP id
    uint32_t rpId;

    /// horizontal hit position and uncertainty, mm
    float x, x_unc;

    /// vertical hit position and uncertainty, mm
    float y, y_unc;

    /// horizontal angle and uncertainty, x = x0 + tx*(z-z0)
    float tx, tx_unc;

    /// vertical angle and uncertainty, y = y0 + ty*(z-z0)
    float ty, ty_unc;

    /// fit chi^2 over NDF
    float chiSquaredOverNDF;

    /// Track information byte for bx-shifted runs: 
    /// reco_info = 0 -> Default value for tracks reconstructed in non-bx-shifted ROCs
    /// reco_info = 1 -> Track reconstructed in a bx-shifted ROC with bx-shifted planes only
    /// reco_info = 2 -> Track reconstructed in a bx-shifted ROC with non-bx-shifted planes only
    /// reco_info = 3 -> Track reconstructed in a bx-shifted ROC both with bx-shifted and non-bx-shifted planes
    unsigned short reco_info;

   	/// number of points used for fit
   	unsigned short numberOfPointUsedForFit; 
   	
    /// time information and uncertainty
    float time, time_unc;

};

#endif
