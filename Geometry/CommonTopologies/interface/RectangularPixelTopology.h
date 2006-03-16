#ifndef Geometry_CommonTopologies_RectangularPixelTopology_H
#define Geometry_CommonTopologies_RectangularPixelTopology_H

/** Specialised strip topology for rectangular barrel detectors.
 *  The strips are parallel to the local Y axis, so X is the precisely
 *  measured coordinate.
 */

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"

  /**
   * Topology for rectangular pixel detector. The only available.
   */
class RectangularPixelTopology : public PixelTopology {
public:
  RectangularPixelTopology( int nrows, int ncols, float pitchx, float pitchy) :
    m_nrows(nrows), m_ncols(ncols), m_pitchx(pitchx), m_pitchy(pitchy) {
    m_xoffset = -nrows/2. * pitchx;
    m_yoffset = -ncols/2. * pitchy;
  }

  // Topology interface

  virtual LocalPoint localPosition( const MeasurementPoint& mp) const {
    return LocalPoint( mp.x()*m_pitchx + m_xoffset, 
		       mp.y()*m_pitchy + m_yoffset);
  }
    
  virtual LocalError localError( const MeasurementPoint&,
				 const MeasurementError& ) const;
  

  virtual MeasurementPoint measurementPosition( const LocalPoint& lp) const {
    std::pair<float,float> p = pixel(lp);
    return MeasurementPoint( p.first, p.second);
  }

  virtual MeasurementError 
  measurementError( const LocalPoint&, const LocalError& ) const;

  virtual int channel( const LocalPoint& lp) const {
    std::pair<float,float> p = pixel(lp);
    return PixelChannelIdentifier::pixelToChannel( int(p.first), int(p.second));
  }

  // PixelTopology interface

  virtual std::pair<float,float> pixel( const LocalPoint& p) const {
    return std::pair<float,float>( (p.x() - m_xoffset) / m_pitchx,
                              (p.y() - m_yoffset) / m_pitchy);
  }
      
  virtual std::pair<float,float> pitch() const {
    return std::pair<float,float>( m_pitchx, m_pitchy);
  }

  virtual int nrows() const {
    return m_nrows;
  }

  virtual int ncolumns() const {
    return m_ncols;
  }
  

private:
  int m_nrows;
  int m_ncols;
  float m_pitchx;
  float m_pitchy;
  float m_xoffset;
  float m_yoffset;
};

#endif


