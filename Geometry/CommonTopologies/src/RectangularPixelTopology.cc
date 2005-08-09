#include "Geometry/CommonTopologies/interface/RectangularPixelTopology.h"

LocalError 
RectangularPixelTopology::localError( const MeasurementPoint& mp,
				      const MeasurementError& me) const {
  return LocalError( me.uu()*(m_pitchx*m_pitchx), 0,
		     me.vv()*(m_pitchy*m_pitchy));
}

MeasurementError 
RectangularPixelTopology::measurementError( const LocalPoint& lp,
					    const LocalError& le) const {
  return MeasurementError( le.xx()/(m_pitchx*m_pitchx), 0,
			   le.yy()/(m_pitchy*m_pitchy));
}
