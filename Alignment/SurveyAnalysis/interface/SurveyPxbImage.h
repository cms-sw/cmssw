#ifndef GUARD_surveypxbimage_h
#define GUARD_surveypxbimage_h

#include <sstream>
#include <vector>
#include <utility>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

//! Class to hold one picture of the BPix survey
class SurveyPxbImage
{
    public:
	typedef unsigned int count_t;
	typedef unsigned int id_t;
	typedef LocalPoint coord_t;
	typedef double value_t;

	//! \p enum to help access specific locations on a picture
	enum location{
	    lr, // lower right mark  (1)
	    ur, // upper right mark  (2)
	    ul, // upper left mark   (3)
	    ll  // lower left mark   (4)
	};

	// Constructors
	SurveyPxbImage();
	/*! Constructor from a string stream.\n
	  Observe the ordering:
	  A line needs to be of the form <tt>rawID1 v_1_1 u_1_1 v_2_1 u_2_1 rawId2 v_1_2 u_1_2 v_2_2 u_2_2 sigma_v sigma_u</tt>\n
	  <tt>x_i_1</tt> denoting the left, <tt>x_i_2</tt> the right module. The data is then mapped to
	  \verbatim
	  -------------++--------------
	          (3) +||+ (2)  
		       ||
	   left module || right module
		       ||
		       ||
	          (4) +||+ (1)     
	  -------------++-------------- \endverbatim
	  where <tt>(i)</tt> refers to the entry in the std::vector measurements\n
	  Therefore the mapping is as follows:
	  - <tt>v_1_1, u_1_1 -> (3)</tt>
	  - <tt>v_2_1, u_2_1 -> (4)</tt>
	  - <tt>v_1_2, u_1_2 -> (2)</tt>
	  - <tt>v_2_2, u_2_2 -> (1)</tt>
	  The sigmas denote the Gaussian error of the measurement in the u and v coordinate
	  */
	SurveyPxbImage(std::istringstream &iss);

	//! Get \p Id of first module
	const id_t getIdFirst() { return idPair.first; };
	//! Get \p Id of second module
	const id_t getIdSecond() { return idPair.second; };

	/*! Get coordinate of a measurement
	  \param m number of mark
	 */
    const coord_t getCoord(count_t m);	

	//! Get Gaussian error in u direction
	const value_t getSigmaU() { return sigma_u; }

	//! Get Gaussian error in u direction
	const value_t getSigmaV() { return sigma_v; }

	//! returns validity flag
	bool isValid() { return isValidFlag; };
	

    private:
	//! Vector to hold four measurements
	std::vector<coord_t> measurementVec;

	//! Gaussian errors
	value_t sigma_u, sigma_v;

	//! Validity Flag
	bool isValidFlag;
	
	//! Pair to hold the Id's of the involved modules
	//! first: module with lower Id
	std::pair<id_t,id_t> idPair;
	
};

#endif

