#ifndef L1GCTINTERNHTMISS_H
#define L1GCTINTERNHTMISS_H


/*! 
 * \class L1GctInternHtMiss
 * \brief L1 GCT internal Ht Miss component(s) Ht_x and/or Ht_y 
 * 
 * \author Robert Frazier
 * \date March 2009
 */

// C++ headers
#include <ostream>
#include <cstdint>

class L1GctInternHtMiss
{

 public:

  /// Enum for the variants of Internal HtMiss.
  enum L1GctInternHtMissType{ nulltype,
                              miss_htx,
                              miss_hty,
                              miss_htx_and_hty,
			      jf_miss_htx_and_hty };

  enum numberOfBits {
    // TODO - tidy up all these enums.
    // These numbers of bits are needed to specify data sizes in the emulator
    // Other masks are below in the private section.
    kJetMissHtNBits   = 12,
    kMissHxAndHyNBits = 14,
    kMissHxOrHyNBits  = 16
  };

  /// default constructor (for vector initialisation etc.)
  L1GctInternHtMiss();

  /// destructor
  ~L1GctInternHtMiss();

 
  /* Named Constructors */

  /// Named ctor for making missing Ht x-component object from unpacker raw data.
  static L1GctInternHtMiss unpackerMissHtx(const uint16_t capBlock,
                                           const uint16_t capIndex,
                                           const int16_t bx,
                                           const uint32_t data);

  /// Named ctor for making missing Ht y-component object from unpacker raw data.
  static L1GctInternHtMiss unpackerMissHty(const uint16_t capBlock,
                                           const uint16_t capIndex,
                                           const int16_t bx,
                                           const uint32_t data);

  /// Named ctor for making missing Ht x & y components object from unpacker raw data (wheel input).  
  static L1GctInternHtMiss unpackerMissHtxHty(const uint16_t capBlock,
                                              const uint16_t capIndex,
                                              const int16_t bx,
                                              const uint32_t data);


  /// Named ctor for making missing Ht x & y components object from emulator (jetFinder output).  
  static L1GctInternHtMiss emulatorJetMissHt(const int htx,
					     const int hty,
					     const bool overFlow,
					     const int16_t bx);

  /// Named ctor for making missing Ht x & y components object from emulator (wheel input).  
  static L1GctInternHtMiss emulatorMissHtxHty(const int htx,
					      const int hty,
					      const bool overFlow,
					      const int16_t bx);

  /// Named ctor for making missing Ht x component object from emulator  
  static L1GctInternHtMiss emulatorMissHtx(const int htx,
					   const bool overFlow,
					   const int16_t bx);

  /// Named ctor for making missing Ht y component object from emulator  
  static L1GctInternHtMiss emulatorMissHty(const int hty,
					   const bool overFlow,
					   const int16_t bx);


  /* Metadata */

  /// 'type' of object?
  L1GctInternHtMiss::L1GctInternHtMissType type() const { return type_; }

  /// Get capture block
  uint16_t capBlock() const { return capBlock_; }

  /// Get index within capture block
  uint16_t capIndex() const { return capIndex_; }

  /// Get BX number
  int16_t bx() const { return bx_; }

  /// Is there a valid Ht x-component stored?
  bool isThereHtx() const { return (type() == miss_htx || type() == miss_htx_and_hty || type() == jf_miss_htx_and_hty); }

  /// Is there a valid Ht y-component stored?
  bool isThereHty() const { return (type() == miss_hty || type() == miss_htx_and_hty || type() == jf_miss_htx_and_hty); }


  /* Access to the actual data */

  /// Get the raw data
  uint32_t raw() const { return data_; }

  /// Get Ht x-component value
  int16_t htx() const;

  /// Get Ht y-component
  int16_t hty() const;

  /// Get overflow
  bool overflow() const;


  /* Operators */

  /// Equality operator
  bool operator==(const L1GctInternHtMiss& rhs) const { return (type() == rhs.type() && raw() == rhs.raw()); }
  
  /// Inequality operator
  bool operator!=(const L1GctInternHtMiss& rhs) const { return !(*this == rhs); }  


 private:

  /// Useful bit masks and bit shifts
  enum ShiftsAndMasks { kDoubleComponentHtyShift  = 16,        // Bit shift for Hty in miss_htx_and_hty
                        kSingleComponentOflowMask = (1 << 30), // Overflow bit mask in miss_htx or miss_hty
                        kDoubleComponentOflowMask = (1 << 15), // Overflow bit mask in miss_htx_and_hty
                        kSingleComponentHtMask    = 0xffff,    // Ht component mask in miss_htx or miss_hty
                        kDoubleComponentHtMask    = 0x3fff,    // Ht component mask in miss_htx_and_hty
                        kJetFinderComponentHtMask = 0x0fff,    // Ht component mask in jf_miss_htx_and_hty
                        kSingleComponentRawMask   = kSingleComponentOflowMask | kSingleComponentHtMask,    // To mask off all the non-data bits in raw data (e.g. BC0, etc)
                        kDoubleComponentRawMask   = (kDoubleComponentHtMask << kDoubleComponentHtyShift) |
                                                     kDoubleComponentOflowMask | kDoubleComponentHtMask };  

  /* Private ctors and methods */

  /// Private constructor that the named ctors use.
  L1GctInternHtMiss(const L1GctInternHtMissType type,
                    const uint16_t capBlock,
                    const uint16_t capIndex,
                    const int16_t bx,
                    const uint32_t data);

  /// Converts 14-bit two's complement numbers to 16-bit two's complement (i.e. an int16_t)
  /*! The input is the raw bits of the 14-bit two's complement in a 16-bit unsigned number. */
  int16_t convert14BitTwosCompTo16Bit(const uint16_t data) const;


  /* Private data */

  /// 'Type' of the data
  L1GctInternHtMissType type_;

  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  /// The captured raw data
  uint32_t data_;  
};

// Pretty-print operator for L1GctInternHtMiss
std::ostream& operator<<(std::ostream& os, const L1GctInternHtMiss& rhs);

#endif
