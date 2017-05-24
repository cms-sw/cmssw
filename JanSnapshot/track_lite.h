#ifndef _track_lite_h_
#define _track_lite_h_

#include <map>

//----------------------------------------------------------------------------------------------------

struct TrackData
{
	bool valid = false;

	// track position, in m
	double x = 0.;
	double y = 0.;

	// track position uncertainty, in m
	double x_unc = 0.;
	double y_unc = 0.;
};

//----------------------------------------------------------------------------------------------------

struct TrackDataCollection : public std::map<unsigned int, TrackData>
{
};

#endif
