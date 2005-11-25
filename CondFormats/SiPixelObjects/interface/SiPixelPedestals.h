#ifndef CondFormats_TrackerObjects_SiPixelPedestals_h
#define CondFormats_TrackerObjects_SiPixelPedestals_h

//----------------------------------------------------------------------------
//! \class SiPixelPedestals
//! \brief Event Setup object which holds DB information for all pixels.
//!
//! \description Event Setup object which holds DB information for all pixels. 
//! DB info for a single pixel is held in SiPixelDbItem, which is a bit-packed
//! 32-bit word.
//-----------------------------------------------------------------------------

#include "CondFormats/SiPixelObjects/interface/SiPixelDbItem.h"

#include <vector>
#include <map>

class SiPixelPedestals {
 public:

  //! Constructor, destructor
  SiPixelPedestals();
  ~SiPixelPedestals();

  typedef std::vector<SiPixelDbItem> SiPixelPedestalsVector;
  typedef std::vector<SiPixelDbItem>::const_iterator SiPixelPedestalsVectorIterator;
  //  SiPixelPedestalsVector  v_pedestals;

  typedef std::map<unsigned int, SiPixelPedestalsVector>                 SiPixelPedestalsMap;
  typedef std::map<unsigned int, SiPixelPedestalsVector>::const_iterator SiPixelPedestalsMapIterator;

  // TO DO: shouldn't the map be private???

  std::map<int, SiPixelPedestalsVector> m_pedestals;
};
#endif
