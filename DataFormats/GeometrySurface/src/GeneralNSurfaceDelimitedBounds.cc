

#include "DataFormats/GeometrySurface/interface/GeneralNSurfaceDelimitedBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

bool GeneralNSurfaceDelimitedBounds::inside( const Local3DPoint& lp, 
					     const LocalError& le, float scale) const
{
    // derive on-surface tolerance from LocalError in a very approximate way
    float tolerance = scale * sqrt( le.xx()*le.xx() + le.yy()*le.yy());

    return myInside( lp, tolerance);
}

bool GeneralNSurfaceDelimitedBounds::myInside( const Local3DPoint& lp, float tolerance) const
{
    // cout << "GeneralNSurfaceDelimitedBounds::myInside called with local point " << lp << endl;

    Surface::GlobalPoint gp = theSurface->toGlobal(lp);

    // cout << "corresponding Global point " << gp << endl;

    for (SurfaceContainer::const_iterator i=theLimits.begin(); i!=theLimits.end(); i++) {

// 	cout << "Local pos in boundary surface " <<  i->first->toLocal(gp) 
// 	     << " side " << i->first->side(gp, tolerance) << " should be " 
// 	     << i->second << endl;

	SurfaceOrientation::Side side = i->first->side(gp, tolerance);
	if (side != i->second && side != SurfaceOrientation::onSurface) return false;
    }
    return true;
}
