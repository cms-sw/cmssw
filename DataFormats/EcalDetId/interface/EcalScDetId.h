// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EcalScDetId.h,v 1.5 2010/02/04 17:09:27 heltsley Exp $
//
// \author Philippe Gras (CEA/Saclay). Code adapted from EEDetId.
//
#ifndef EcalDetId_EcalScDetId_h
#define EcalDetId_EcalScDetId_h

#include <ostream>
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EcalScDetId
 *  Supercrystal identifier class for the ECAL endcap.
 *  <P>Note: internal representation of ScDetId:
 *  \verbatim
 *  31              .               15              .              0
 *  |-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-|-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-| 
 *  |  det  |sudet|         0       |1|z|     ix      |     iy      |
 *  +-------+-----+-----------------+-+-+-------------+-------------+
 *  \endverbatim
 */

class EcalScDetId : public DetId {
 public:

  /** Constructor of a null id
   */
  EcalScDetId();
  
  /** Constructor from a raw value
   * @param rawid det ID number of the supecrystal, as defined in this class
   * description.
   */
  EcalScDetId(uint32_t rawid);

  /** Constructor from supercrystal ix,iy,iz (iz=+1/-1)
   * ix x-index runs from 1 to 20 along x-axis of standard CMS coordinates
   * iy y-index runs from 1 to 20 along y-axis of standard CMS coordinates
   * iz z-index (also called "z-side") is -1 for EE- and +1 for EE+
   * @param ix x-index
   * @param iy y-index
   * @param iz z-side /z-index: -1 for EE-, +1 for EE+
   */
  EcalScDetId(int ix, int iy, int iz);
  
  /** Constructor from a raw value
   * @param id det ID number
   */
  EcalScDetId(const DetId& id);

  /** Assignment operator
   * @param id source det id
   */ 
  EcalScDetId& operator=(const DetId& id);

  /** Gets the subdetector
   * @return subdetectot ID, that is EcalEndcap
   */
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
  
  /** Gets the z-side of the crystal (1/-1)
   * @return -1 for EE-, +1 for EE+
   */
  int zside() const { return (id_&0x4000)?(1):(-1); }
  
  /** Gets the crystal x-index.
   * @see EcalDetId(int, int, int) for x-index definition
   * @return x-index
   */
  int ix() const { return (id_>>7)&0x7F; }
  
  /** Get the crystal y-index
   * @see EcalDetId(int, int, int) for y-index definition.
   * @return y-index
   */
  int iy() const { return id_&0x7F; }

  
  /** Gets the SuperCrystal number within the endcap.
   * @see EEDetId::isc() for the index description
   * @return the supercrystal index, a number from 1 to 316. Numbers 
   * 70,149,228,307 are not used.
   */
  int isc() const ;
  
  /** Gets the quadrant of the DetId
   * @see EEDetId::iquadrant()
   * @return quadrant number, from 1 to 4.
   */
  int iquadrant() const ;
  
  /** Gets a compact index for arrays. This index runs from 0 to kSizeForDenseIndexing-1
   *
   * BEWARE: this index contains 8 holes. Numbers 69, 148, 227, 306, 385, 464, 543, 622
   * are not used
   * @return the hased index.
   */
  int hashedIndex() const { return isc() + ( zside() + 1 )*ISC_MAX/2 - 1 ; } // from 0-631

  /** Same as hashed index.
   *
   * BEWARE: despite the method name is suggesting the returned index is not dense! See hashedIndex()
   */
  uint32_t denseIndex() const { return hashedIndex() ; }

  /** Validates a hashed index.
   *
   * BEWARE: this method is broken. It will returns true for the 8 invalid numbers called "holes"
   * in hashIndex() documentation.
   * @param din hashed index to validate
   * @return true if the index is valid, false if it is invalid.
   */
  static bool validDenseIndex( uint32_t din ) { return ( MIN_HASH<=(int)din && MAX_HASH>=(int)din ) ; }

  
  /** Validates a hashed index.
   *
   * BEWARE: this method is broken. It will returns true for the 8 invalid numbers called "holes"
   * in hashIndex() documentation.
   * @param hi hashed index to validate
   * @return true if the index is valid, false if it is invalid.
   */
  static bool validHashIndex( int hi ) { return validDenseIndex( hi ) ; }

  /** Lower bound of EE supercrystal x-index
   */
  static const int IX_MIN=1;

  /** Lower bound of EE supercrystal y-index
   */
  static const int IY_MIN=1;

  /** Upper bound of EE crystal y-index
   */
  static const int IX_MAX=20;

  /** Upper bound of EE crystal y-index
   */
  static const int IY_MAX=20;

  /** Lower bound of supercystal index defined in isc()
   */
  static const int ISC_MIN=EEDetId::ISC_MIN ;
  
  /** Upper bound of supercystal index defined in isc()
   * Beware it differs from the number of supercrystals in one endcap,
   * which is itseld 312, because the numbering is not dense.
   */
  static const int ISC_MAX=EEDetId::ISC_MAX ;

  /** Lower bound of hashed index returned by hashedIndex() and denseIndex() methods
   */
  static const int MIN_HASH=0;

  /** Upper bound of hashed index returned by hashedIndex() and denseIndex() methods
   */
  static const int MAX_HASH=2*ISC_MAX - 1;

  /** Checks validity of a crystal (x,y.z) index triplet.
   * @param ix supercrystal x-index
   * @param iy supercrystal y-index
   * @param iz supercrystal z-index (aka z-side)
   * @see EEDetId(int, int, int) for index definition
   * @return true if valid, false otherwise
   */
  static bool validDetId(int ix, int iy, int iz) ;

  enum {
    /** Number of dense supercrystal indices. This number differs
     * from the total number of supercystals. See hashedIndex() method.
     */
    kSizeForDenseIndexing = 2*ISC_MAX
  };

// private:
};


std::ostream& operator<<(std::ostream& s,const EcalScDetId& id);


#endif //EcalDetId_EcalScDetId_h not defined
