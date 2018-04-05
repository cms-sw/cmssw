/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_TotemRPRecHit
#define DataFormats_CTPPSReco_TotemRPRecHit

/**
 *\brief Reconstructed hit in TOTEM RP.
 *
 * Basically a cluster (TotemRPCluster), the position of which has been converted into actual geometry (in mm).
 **/
class TotemRPRecHit
{
  public:
    TotemRPRecHit(double position=0, double sigma=0) : position_(position), sigma_(sigma)
    {
    }

    inline double getPosition() const { return position_; }
    inline void setPosition(double position) { position_=position; }

    inline double getSigma() const { return sigma_; }
    inline void setSigma(double sigma) { sigma_=sigma; }

  private:
    /// position of the hit in mm, wrt detector center (see RPTopology::GetHitPositionInReadoutDirection)
    double position_;   

    /// position uncertainty, in mm
    double sigma_;      
};

//----------------------------------------------------------------------------------------------------

inline bool operator< (const TotemRPRecHit &l, const TotemRPRecHit &r)
{
  if (l.getPosition() < r.getPosition())
    return true;
  if (l.getPosition() > r.getPosition())
    return false;

  if (l.getSigma() < r.getSigma())
    return true;

  return false;
}

#endif
