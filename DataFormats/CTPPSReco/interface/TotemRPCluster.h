/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_TotemRPCluster
#define DataFormats_CTPPSReco_TotemRPCluster

/**
 *\brief Cluster of TOTEM RP strip hits.
 **/
class TotemRPCluster
{
  public:
    TotemRPCluster(unsigned short str_beg=0, unsigned short str_end=0) : str_beg_(str_beg), str_end_(str_end)
    {
    }

    inline uint16_t getStripBegin() const { return str_beg_; }
    inline void setStripBegin(unsigned short str_beg) { str_beg_ = str_beg; }
 
    inline uint16_t getStripEnd() const { return str_end_; }
    inline void setStripEnd(unsigned short str_end) { str_end_ = str_end; }
 
    inline int getNumberOfStrips() const { return str_end_ - str_beg_ + 1; }
  
    inline double getCenterStripPosition() const { return (str_beg_ + str_end_)/2.; }
  
  private:
    uint16_t str_beg_;
    uint16_t str_end_;
};

//----------------------------------------------------------------------------------------------------

inline bool operator<( const TotemRPCluster& l, const TotemRPCluster& r)
{
  if (l.getStripBegin() < r.getStripBegin())
    return true;
  if (l.getStripBegin() > r.getStripBegin())
    return false;

  if (l.getStripEnd() < r.getStripEnd())
    return true;
  
  return false;
}

#endif
