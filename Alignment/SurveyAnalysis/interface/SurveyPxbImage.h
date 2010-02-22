#ifndef GUARD_surveypxbimage_h
#define GUARD_surveypxbimage_h

#include <sstream>
#include <vector>
#include <utility>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"

//! Class to hold one picture of the BPix survey
class SurveyPxbImage
{
    public:
	typedef unsigned int count_t;
	typedef unsigned int id_t;
	typedef double value_t;
	typedef Point3DBase<value_t, LocalTag> coord_t;

	//! \p enum to help access specific locations on a picture
	enum location{
	    lr, // lower right mark  (0)
	    ur, // upper right mark  (1)
	    ul, // upper left mark   (2)
	    ll  // lower left mark   (3)
	};

	// Constructors
	SurveyPxbImage();
	/*! Constructor from a string stream.\n
	  Observe the ordering:
	  A line needs to be of the form <tt>rawID1 v_1_1 u_1_1 v_2_1 u_2_1 rawId2 v_1_2 u_1_2 v_2_2 u_2_2 sigma_v sigma_u</tt>\n
	  <tt>x_i_1</tt> denoting the left, <tt>x_i_2</tt> the right module. The data is then mapped to
	  \verbatim
	  -------------++--------------
	          (2) +||+ (1)  
		       ||
	   left module || right module
		       ||
		       ||
	          (3) +||+ (0)     
	  -------------++-------------- \endverbatim
	  where <tt>(i)</tt> refers to the entry in the std::vector measurements\n
	  Therefore the mapping is as follows:
	  - <tt>v_1_1, u_1_1 -> (2)</tt>
	  - <tt>v_2_1, u_2_1 -> (3)</tt>
	  - <tt>v_1_2, u_1_2 -> (1)</tt>
	  - <tt>v_2_2, u_2_2 -> (0)</tt>
	  The sigmas denote the Gaussian error of the measurement in the u and v coordinate
	  */
	SurveyPxbImage(std::istringstream &iss);

	void fill(std::istringstream &iss);

	//! Get \p Id of first module
	const id_t getIdFirst() { return idPair_.first; };
	//! Get \p Id of second module
	const id_t getIdSecond() { return idPair_.second; };

	/*! Get coordinate of a measurement
	  \param m number of mark
	 */
    const coord_t getCoord(count_t m);	

	//! Get Gaussian error in u direction
	const value_t getSigmaU() { return sigma_u_; }

	//! Get Gaussian error in u direction
	const value_t getSigmaV() { return sigma_v_; }

	//! returns validity flag
	bool isValid() { return isValidFlag_; };
	

    protected:
	//! Vector to hold four measurements
	std::vector<coord_t> measurementVec_;

	//! Gaussian errors
	value_t sigma_u_, sigma_v_;

	//! Validity Flag
	bool isValidFlag_;
	
	//! Pair to hold the Id's of the involved modules
	//! first: module with lower Id
	std::pair<id_t,id_t> idPair_;
	
};

#endif

