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
	typedef std::pair<id_t,id_t> idPair_t;

	//! \p enum to help access specific locations on a picture
	enum location{
	    ll, // lower left mark   (0)
	    ul, // upper left mark   (1)
	    lr, // lower right mark  (2)
	    ur  // upper right mark  (3)
	};

	// Constructors
	SurveyPxbImage();
	/*! Constructor from a string stream.\n
	  Observe the ordering:
	  A line needs to be of the form <tt>rawID1 y_1_1 x_1_1 y_2_1 x_2_1 rawId2 y_1_2 x_1_2 y_2_2 x_2_2 sigma_y sigma_x</tt>\n
	  <tt>x_i_1</tt> denoting the left, <tt>x_i_2</tt> the right module. The data is then mapped to
	  \verbatim
	  -------------++--------------
	          (1) +||+ (3)  
		       ||
	   left module || right module
		       ||
		       ||
	          (0) +||+ (2)     
	  -------------++-------------- \endverbatim
	  where <tt>(i)</tt> refers to the entry in the std::vector measurements\n
	  Therefore the mapping is as follows:
	  - <tt>y_1_1, x_1_1 -> (0)</tt>
	  - <tt>y_2_1, x_2_1 -> (1)</tt>
	  - <tt>y_1_2, x_1_2 -> (2)</tt>
	  - <tt>y_2_2, x_2_2 -> (3)</tt>
	  The sigmas denote the Gaussian error of the measurement in the u and v coordinate
	  */
	SurveyPxbImage(std::istringstream &iss) : isValidFlag_(false)
	{
		fill(iss);
	};

	void fill(std::istringstream &iss);

	//! Get \p Id of first module
	id_t getIdFirst() { return idPair_.first; };
	//! Get \p Id of second module
	id_t getIdSecond() { return idPair_.second; };
	//! Get \p Id pair
	const idPair_t getIdPair() { return idPair_; };

	/*! Get coordinate of a measurement
	  \param m number of mark
	 */
        const coord_t getCoord(count_t m);	

	//! Get Gaussian error in u direction
        value_t getSigmaX() { return sigma_x_; }

	//! Get Gaussian error in u direction
	value_t getSigmaY() { return sigma_y_; }

	//! returns validity flag
	bool isValid() { return isValidFlag_; };
	

    protected:
	//! Vector to hold four measurements
	std::vector<coord_t> measurementVec_;

	//! Gaussian errors
	value_t sigma_x_, sigma_y_;

	//! Flag if the image was rotated or not
	bool isRotated_;

	//! Validity Flag
	bool isValidFlag_;
	
	//! Pair to hold the Id's of the involved modules
	//! first: module with lower Id
	idPair_t idPair_;
	
};

#endif

