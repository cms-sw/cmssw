#ifndef AlphaT_h
#define AlphaT_h

#include <cmath>
#include <numeric>
#include <vector>


/*  AlphaT
 *
 *  Calculates the AlphaT event variable for a given input jet collection
 *
 *
 *  08 Sept 2014 - Ported from ICF code (Mark Baber)
 */


struct AlphaT {

  static double getAlphaT( const std::vector<double>& et,
		    const std::vector<double>& px,
		    const std::vector<double>& py );
   
};

#endif // AlphaT_h
