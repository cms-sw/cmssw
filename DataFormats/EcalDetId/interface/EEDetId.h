#ifndef ECALDETID_EEDETID_H
#define ECALDETID_EEDETID_H

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EEDetId
 *  Crystal/cell identifier class for the ECAL endcap
 *
 *
 *  $Id: EEDetId.h,v 1.28 2012/11/05 17:36:08 innocent Exp $
 */
class EEDetId : public DetId {
public:
  enum {
    /** Sudetector type. Here it is ECAL endcap.
     */
    Subdet=EcalEndcap
  };
  
  /** Constructor of a null id
   */
  EEDetId() {}
  
  /** Constructor from a raw value
   * @param rawid det ID number
   */
  EEDetId(uint32_t rawid) : DetId(rawid) {}
  
  /** Constructor from crystal ix,iy,iz (iz=+1/-1) (mode = XYMODE)
   * or from sc,cr,iz (mode = SCCRYSTALMODE).
   * <p>ix runs from 1 to 100 along x-axis of standard CMS coordinates<br>
   * iy runs from 1 to 100 along y-axis of standard CMS coordinates<br>
   * iz is -1 for EE- and +1 for EE+<br>
   * <p>For isc see isc(), for ic see ic()
   * @see isc(), ic()
   * @param i ix or isc index
   * @param j iy or isc index
   * @param iz iz/zside index: -1 for EE-, +1 for EE+
   * @param mode pass XYMODE if i j refer to ix, iy, SCCRYSTALMODE if thery refer to isc, ic
   */
  // fast  
  EEDetId(int crystal_ix, int crystal_iy, int iz) : 
    DetId( Ecal, EcalEndcap ) {
    id_|=(crystal_iy&0x7f)|((crystal_ix&0x7f)<<7)|((iz>0)?(0x4000):(0));
  }
  // slow
  EEDetId(int i, int j, int iz, int mode);
  
  /** Constructor from a generic cell id
   * @param id source detid
   */
  EEDetId(const DetId& id) : DetId(id){}
  
  /** Assignment operator
   * @param id source det id
   */ 
  EEDetId& operator=(const DetId& id) {id_ = id.rawId(); return *this;}
  
  /** Gets the subdetector
   * @return subdetectot ID, that is EcalEndcap
   */
  static EcalSubdetector subdet() { return EcalEndcap;}
  
  /** Gets the z-side of the crystal (1/-1)
   * @return -1 for EE-, +1 for EE+
   */
  int zside() const { return (id_&0x4000)?(1):(-1); }
  
  /** Gets the crystal x-index.
   * @see EEDetId(int, int, int, int) for x-index definition
   * @return x-index
   */
  int ix() const { return (id_>>7)&0x7F; }
  
  /** Get the crystal y-index
   * @see EEDetId(int, int, int, int) for y-index definition.
   * @return y-index
   */
  int iy() const { return id_&0x7F; }
  
  /** Gets the DetId of the supercrystal the crystal belong to.
   * @return the supercrystal det id
   * @throw cms::Exception if the crystal det id is invalid 
   */ 
  EcalScDetId sc() const {
    const int scEdge = 5;
    return EcalScDetId(1+(ix()-1)/scEdge, 1+(iy()-1)/scEdge, zside());
  }
  
  /** Gets the SuperCrystal number within the endcap. This number runs from 1 to 316,
   * numbers 70 149 228 307 are not used.
   *
   * BEWARE: This number is not consistent with indices used in constructor:  see details below.
   *
   * Numbering in quadrant 1 of EE+ is the following
   * \verbatim 
   *  08 17 27        
   *  07 16 26 36 45 54     
   *  06 15 25 35 44 53 62    
   *  05 14 24 34 43 52 61 69   
   *  04 13 23 33 42 51 60 68 76  
   *  03 12 22 32 41 50 59 67 75  
   *  02 11 21 31 40 49 58 66 74  
   *  01 10 20 30 39 48 57 65 73 79 
   *     09 19 29 38 47 56 64 72 78 
   *        18 28 37 46 55 63 71 77
   *  
   *        == THERE IS NO INDEX 70! ==
   * \endverbatim
   *
   * Quadrant 2 indices are deduced by a symetry about y-axis and by adding an offset
   * of 79.<br>
   * Quadrant 3 and 4 indices are deduced from quadrant 1 and 2 by a symetry
   * about x-axis and adding an offset. Quadrant N starts with index 1 + (N-1)*79.
   *
   * <p>EE- indices are deduced from EE+ by a symetry about (x,y)-plane (mirrored view). <b>It is
   * inconsistent with indices used in constructor EEDetId(int, int,int) in
   * SCCRYSTALMODE</b>. Indices of constructor uses a symetry along y-axis: in principal it
   * considers the isc as a local index. The discrepancy is most probably due to a bug in the
   * implementation of this isc() method.
   */
  int isc() const ;
  
  /** Gets crystal number inside SuperCrystal.
   * Crystal numbering withing a supercrystal in each quadrant:
   * \verbatim
   *                       A y
   *  (Q2)                 |                    (Q1)
   *       25 20 15 10 5   |     5 10 15 20 25
   *       24 19 14  9 4   |     4  9 14 19 24
   *       23 18 13  8 3   |     3  8 13 18 23
   *       22 17 12  7 2   |     2  7 12 17 22
   *       21 16 11  6 1   |     1  6 11 16 21
   *                       |
   * ----------------------o---------------------------> x
   *                       |
   *       21 16 11  6 1   |     1  6 11 16 21
   *       22 17 12  7 2   |     2  7 12 17 22
   *       23 18 13  8 3   |     3  8 13 18 23
   *       24 19 14  9 4   |     4  9 14 19 24
   *       25 20 15 10 5   |     5 10 15 20 25
   *  (Q3)                                       (Q4)
   * \endverbatim
   *
   * @return crystal number from 1 to 25
   */
  int ic() const;
  
  /** Gets the quadrant of the DetId.
   * Quadrant number definition, x and y in std CMS coordinates, for EE+:
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
   *
   * @return quadrant number
   */
  int iquadrant() const ;
  
  /** Checks if crystal is in EE+
   * @return true for EE+, false for EE-
   */
  bool positiveZ() const { return id_&0x4000;}
  
  int iPhiOuterRing() const ; // 1-360 else==0 if not on outer ring!
  
  static EEDetId idOuterRing( int iPhi , int zEnd ) ;
  
  /** Gets a compact index for arrays
   * @return compact index from 0 to kSizeForDenseIndexing-1
   */
  int hashedIndex() const 
  {
    const uint32_t jx ( ix() ) ;
    const uint32_t jd ( 2*( iy() - 1 ) + ( jx - 1 )/50 ) ;
    return (  ( positiveZ() ? kEEhalf : 0) + kdi[jd] + jx - kxf[jd] ) ;
  }
  
  /** Same as hashedIndex()
   * @return compact index from 0 to kSizeForDenseIndexing-1
   */
  uint32_t denseIndex() const { return hashedIndex() ; }
  
  /** returns a new EEDetId offset by nrStepsX and nrStepsY (can be negative),
   * returns EEDetId(0) if invalid */
  EEDetId offsetBy( int nrStepsX, int nrStepsY ) const;
  
      /** returns a new EEDetId swapped (same iX, iY) to the other endcap, 
       * returns EEDetId(0) if invalid (shouldnt happen) */
  EEDetId switchZSide() const;
  
  /** following are static member functions of the above two functions
   *  which take and return a DetId, returns DetId(0) if invalid 
   */
  static DetId offsetBy( const DetId startId, int nrStepsX, int nrStepsY );
  static DetId switchZSide( const DetId startId );
  
  /** Checks validity of a dense/hashed index
   * @param din dense/hashed index as returned by hashedIndex() or denseIndex()
   * method
   * @return true if index is valid, false otherwise
   */
  static bool validDenseIndex( uint32_t din ) { return validHashIndex( din ) ; }
  
  /** Converts a hashed/dense index as defined in hashedIndex() and denseIndex()
   * methods to a det id.
   * @param din hashed/dense index
   * @return det id
   */
  static EEDetId detIdFromDenseIndex( uint32_t din ) { return unhashIndex( din ) ; }
  
  static bool isNextToBoundary(     EEDetId id ) ;
  
  static bool isNextToDBoundary(    EEDetId id ) ;
  
  static bool isNextToRingBoundary( EEDetId id ) ;
  
  /** Gets a DetId from a compact index for arrays. Converse of hashedIndex() method.
   * @param hi dense/hashed index
   * @return det id
   */
  static EEDetId unhashIndex( int hi ) ;
  
  /** Checks if a hashed/dense index is valid
   * @see hashedIndex(), denseIndex()
   * @param i hashed/dense index
   * @return true if the index is valid, false otherwise
   */
  static bool validHashIndex( int i ) { return ( i < kSizeForDenseIndexing ) ; }
  
  /** Checks validity of a crystal (x,y.z) index triplet.
   * @param crystal_ix crystal x-index
   * @param crystal_iy crystal y-index
   * @param iz crystal z-index
   * @see EEDetId(int, int, int, int) for index definition
   * @return true if valid, false otherwise
   */
  static bool validDetId(int crystal_ix, int crystal_iy, int iz) {
    return 
      crystal_ix >= IX_MIN && crystal_ix <= IX_MAX &&
      crystal_iy >= IY_MIN && crystal_iy <= IY_MAX &&  
      std::abs(iz)==1 && 
      ( fastValidDetId(crystal_ix,crystal_iy) ||
	slowValidDetId(crystal_ix,crystal_iy) );
  }
  static bool slowValidDetId(int crystal_ix, int crystal_iy);

  /**  check if ix and iy is in a "ring" inscribed in EE
   *   if is inside is valid for sure
   *   if not the slow version shall be called
   */
  static bool fastValidDetId(int crystal_ix, int crystal_iy) {
    float x =  crystal_ix; float y =  crystal_iy;
    float r = (x - 50.5f) * (x - 50.5f) + (y - 50.5f) * (y - 50.5f);
    return r > 12.f * 12.f && r < 48.f * 48.f;
  }

  /** Returns the distance along x-axis in crystal units between two EEDetId
   * @param a det id of first crystal
   * @param b det id of second crystal
   * @return distance
   */
  static int distanceX(const EEDetId& a,const EEDetId& b);
  
  /** Returns the distance along y-axis in crystal units between two EEDetId
   * @param a det id of first crystal
   * @param b det id of second crystal
   * @return distance
   */
  static int distanceY(const EEDetId& a,const EEDetId& b); 
  
  
  /** Gives supercrystal index from endcap *supercrystal* x and y indexes.
   * @see isc() for the index definition
   * @param iscCol supercrystal column number: supecrystal x-index for EE+
   * @param iscRow: supecrystal y-index
   * @return supercystal index
   */
  static int isc( int iscCol,   // output is 1-316
		  int iscRow ) ; // 
  
  /** Lower bound of EE crystal x-index
   */
  static const int IX_MIN =1;
  
  /** Lower bound of EE crystal y-index
   */
  static const int IY_MIN =1;
  
  /** Upper bound of EE crystal y-index
   */
  static const int IX_MAX =100;
  
  /** Upper bound of EE crystal y-index
   */
  static const int IY_MAX =100;
  
  /** Lower bound of supercystal index as defined in isc()
   */
  static const int ISC_MIN=1;
  
  /** Lower bound of crystal index within a supercrystal
   */
  static const int ICR_MIN=1;
  
  /** Upper bound of supercystal index defined in isc()
   * <p>Beware it differs from the number of supercrystals in one endcap,
   * which is 312, because the numbering is not dense.
   */
  static const int ISC_MAX=316;
  
  /** Upper bound of crystal index within a supercrystal
   */
  static const int ICR_MAX=25;
  
  enum {
    /** Number of crystals per Dee
     */
    kEEhalf = 7324 ,
    /** Number of dense crystal indices, that is number of
     * crystals per endcap.
     */
    kSizeForDenseIndexing = 2*kEEhalf
  };
  
  /*@{*/
  /** function modes for EEDetId(int, int, int, int) constructor
   */
  static const int XYMODE        = 0;
  static const int SCCRYSTALMODE = 1;
  /*@}*/
  
private:
  
  bool        isOuterRing() const ;
  
  static bool isOuterRingXY( int ax, int ay ) ;
  
  //Functions from B. Kennedy to retrieve ix and iy from SC and Crystal number
  
  static const int nCols = 10;
  static const int nCrys = 5; /* Number of crystals per row in SC */
  static const int QuadColLimits[nCols+1];
  static const int iYoffset[nCols+1];
  
  static const unsigned short kxf[2*IY_MAX] ;
  static const unsigned short kdi[2*IY_MAX] ;
  
  int ix( int iSC, int iCrys ) const;
  int iy( int iSC, int iCrys ) const;
  int ixQuadrantOne() const;
  int iyQuadrantOne() const;
};


std::ostream& operator<<(std::ostream& s,const EEDetId& id);

#endif
