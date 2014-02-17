#ifndef GctBlockHeader_h_
#define GctBlockHeader_h_


/*!
* \class GctBlockHeader
* \brief Simple class for holding the basic attributes of an 32-bit block header.
* * 
* \author Robert Frazier
* $Revision: 1.18 $
* $Date: 2009/12/27 05:29:21 $
*/

// C++ headers
#include <ostream>
#include <stdint.h>

class GctBlockHeader
{
public:
 
   /* PUBLIC METHODS */
 
  /// Constructor. Don't use directly - use the generateBlockHeader() method in GctFormatTranslateBase-derived classes.
  /*! \param valid Flag if this is a known and valid header .*/
  GctBlockHeader(uint32_t blockId,
                 uint32_t blockLength,
                 uint32_t nSamples,
                 uint32_t bxId,
                 uint32_t eventId,
                 bool valid);
  
  /// Destructor.
  ~GctBlockHeader() {};
  
  /// Get the block ID
  uint32_t blockId() const { return m_blockId; }

  /// Get the fundamental block length (for 1 time sample)
  uint32_t blockLength() const { return m_blockLength; }

  /// Get the number of time samples
  uint32_t nSamples() const { return m_nSamples; }
  
  /// Get the bunch crossing ID
  uint32_t bxId() const { return m_bxId; }

  /// Get the event ID
  uint32_t eventId() const { return m_eventId; }

  /// Returns true if it's valid block header - i.e. if the header is known and can be unpacked.
  bool valid() const { return m_valid; }


private:

  /* PRIVATE METHODS */  



  /* PRIVATE MEMBER DATA */
  
  uint32_t m_blockId;  ///< The Block ID
  
  uint32_t m_blockLength;  ///< The fundamental block length (for 1 time sample)

  uint32_t m_nSamples;  ///< The number of time-samples 
  
  uint32_t m_bxId;  ///< The bunch-crossing ID
  
  uint32_t m_eventId;  ///< The event ID
  
  bool m_valid; ///< Is this a valid block header

};

#include <vector>
typedef std::vector<GctBlockHeader> GctBlockHeaderCollection;

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h);

#endif /* GctBlockHeader_h_ */
