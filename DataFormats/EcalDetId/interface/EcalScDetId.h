// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EcalScDetId.h,v 1.8 2012/11/02 13:07:52 innocent Exp $
//
// \author Philippe Gras (CEA/Saclay). Code adapted from EEDetId.
//
#ifndef EcalDetId_EcalScDetId_h
#define EcalDetId_EcalScDetId_h

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
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
  
  /** Gets the quadrant of the DetId.
   *
   * Quadrant number definition for EE+, x and y in std CMS coordinates:
   *
   * \verbatim
   *                 A y
   *                 |
   *          Q2     |    Q1
   *                 |
   *       ----------o---------> x
   *                 |
   *          Q3     |    Q4
   *                 |
   * \endverbatim
   * This method will return the same quadrant number independently of
   * z: that is two supercrystals which are face to face will be considered
   * will have the same quadrant number. It is not clear it is the correct
   * or usual definition.
   * @see EEDetId::iquadrant()
   * @return quadrant number, from 1 to 4.
   * @deprecated This method might be withdraw in a future release
   */
  int iquadrant() const ;
  
  
  /** Gets a compact index for arrays. Index runs from 0 to 623.
   * They are ordered by increasing z (EE- then EE+), then for
   * same z by increasing y. then for same z and y by increasing x
   */
  int hashedIndex() const{
    checkHashedIndexMap();
    if(!validDetId(ix(),iy(),zside())) return -1;
    return xyz2HashedIndex[ix()-IX_MIN][iy()-IY_MIN][zside()>0?1:0];
  }

  /** Gets EcalScDetId from hasedIndex as defined by hashedIndex method
   * @param hi hashed index
   * @return the EcalScDetId. If hi is invalid return a null EcalScDetId.
   */
  static EcalScDetId unhashIndex(int hi){
    checkHashedIndexMap();
    if(hi < 0 || hi >= kSizeForDenseIndexing) return EcalScDetId();
    return hashedIndex2DetId[hi];
  }
  
  /** Same as hashed index.
   * @return the dense/hashed index
   */
  uint32_t denseIndex() const { return hashedIndex() ; }

  /** Validates a hashed index.
   * @param din hashed index to validate
   * @return true if the index is valid, false if it is invalid.
   */
  static bool validDenseIndex(uint32_t din) { return din < kSizeForDenseIndexing; }

  
  /** Validates a hashed index.
   * @param hi hashed index to validate
   * @return true if the index is valid, false if it is invalid.
   */
  static bool validHashIndex(int hi) { return validDenseIndex(hi) ; }

  /** Number of supercrystals per endcap
   */
  static const int SC_PER_EE_CNT = 312;
  
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

  /** Lower bound for hashed/dense index
   */
  static const int IHASHED_MIN = 0;

  /** Upper bound for hashed/dense index
   */
  static const int IHASHED_MAX = SC_PER_EE_CNT*2 - 1;
  
  /** Checks validity of a crystal (x,y.z) index triplet.
   * @param ix supercrystal x-index
   * @param iy supercrystal y-index
   * @param iz supercrystal z-index (aka z-side)
   * @see EEDetId(int, int, int) for index definition
   * @return true if valid, false otherwise
   */
  static bool validDetId(int ix, int iy, int iz) ;

private:
  /** Initializes x,y,z <-> hashed index map if not yet done.
   */
  static void checkHashedIndexMap();
  

  //fields
public:
  enum {
    /** Number of dense supercrystal indices.
     */
    kSizeForDenseIndexing = SC_PER_EE_CNT * 2
  };

private:
  static const int nEndcaps = 2;
  
  /** Map of z,x,y index to hashed index. See hashedIndex/
   */
  static short xyz2HashedIndex[IX_MAX][IY_MAX][nEndcaps];
  
  /** Map of hased index to x,y,z. See hashedIndex/
   */
  static EcalScDetId hashedIndex2DetId[kSizeForDenseIndexing];
};


std::ostream& operator<<(std::ostream& s,const EcalScDetId& id);


#endif //EcalDetId_EcalScDetId_h not defined
