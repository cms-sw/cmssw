#ifndef CaloDirection_h
#define CaloDirection_h
//The local directions
/**
   \enum CaloDirection

   \brief Codes the local directions in the cell lattice.
*/
enum CaloDirection{NONE,SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,
		   NORTHEAST,NORTHWEST,NORTH,DOWN,
		   DOWNSOUTH,DOWNSOUTHEAST,DOWNSOUTHWEST,DOWNEAST,DOWNWEST,
		   DOWNNORTHEAST,DOWNNORTHWEST,DOWNNORTH,UP,
		   UPSOUTH,UPSOUTHEAST,UPSOUTHWEST,UPEAST,UPWEST,
		   UPNORTHEAST,UPNORTHWEST,UPNORTH};

//class ostream;
#include <iosfwd>
std::ostream & operator<<(std::ostream&,const CaloDirection&);

#endif
