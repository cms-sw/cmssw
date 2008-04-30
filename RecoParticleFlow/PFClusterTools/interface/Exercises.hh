#ifndef EXERCISES_HH_
#define EXERCISES_HH_

#include <TFile.h>

namespace pftools {
/**
 * \class Exercises
 * \brief Simple test harness for the PFClusterTools package. 
 * 
 * Instantiate one of these classes, and call methods to exercise the objects in the PFClusterTools library.
 * \author Jamie Balin
 * \date April 2008
 */
class Exercises {
public:
	Exercises();
	virtual ~Exercises();

	void testTreeUtility(TFile& f);

	void testCalibrators();
	
	void testCalibrationFromTree(TFile& f);
};
}

#endif /*EXERCISES_HH_*/
