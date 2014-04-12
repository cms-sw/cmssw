//
//  SimplePixel.h (v1.00)
//
 
#ifndef SimplePixel_h
#define SimplePixel_h 1


class SimplePixel { //!< struck pixel class for use in simpletemplate2D
	
public:
	
	SimplePixel() {s = 0.; x = 0.; y = 0.; i = 0; j = 0; btype = 0;} //!< Default constructor
	
	bool operator < (const SimplePixel& rhs) const {return (this->s < rhs.s);}  //!< define < operator so that std::list can sort these by path length
	
	float s;     //!< distance from track entry
	float x;     //!< x coordinate of boundary intersection
	float y;     //!< y coordinate of boundary intersection
	int i;       //!< x index of traversed pixel
	int j;       //!< y index of traversed pixel
	int btype;   //!< type of boundary (0=end, 1 = x-boundary, 2 = y-boundary)
	
} ;


#endif
