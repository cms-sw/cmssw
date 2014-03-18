#ifndef ECALDETID_EKDETID_H
#define ECALDETID_EKDETID_H

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EKDetId
 *  Supermodule/module identifier class for the ECAL shashlik
 *
 *
 *  $Id: EKDetId.h,v 1.28 2014/02/28 17:36:08 sunanda Exp $
 */
class EKDetId : public DetId {
public:
  enum { /** Sudetector type. Here it is ECAL Shashlik  */
    Subdet=EcalShashlik
  };

  enum Configuration { /** hardcoded geometries. Here it is ECAL Shashlik  */
    BlackBox = 0,  // solid block 42x42 supermodules
    NoTaperEcalEta4 = 1,
    NoTaperEcalEta3 = 2,
    TaperEcalEta4 = 3,
    TaperEcalEta3 = 4,
    LAST
  };
  
  EKDetId() {}
  
  EKDetId(uint32_t rawid) : DetId(rawid) {}
  
  /** Constructor from crystal ix,iy,fib,ro,iz (iz=+1/-1) (mode = XYMODE)
   * or from sm,mod,fib,ro,iz (mode = SCMODULEMODE).
   * <p>ix runs from 1 to 180 along x-axis of standard CMS coordinates<br>
   * iy runs from 1 to 180 along y-axis of standard CMS coordinates<br>
   * fib runs from 0 to 5 for fiber type (0 is combined)<br>
   * ro runs from 0 to 2 for read out type (0 is combined)<br>
   * iz is -1 for EK- and +1 for EK+<br>
   * <p>For ism see ism(), for imod see imod()
   * @see ism(), imod()
   * @param i ix or ism index
   * @param j iy or ism index
   * @param iz iz/zside index: -1 for EK-, +1 for EK+
   * @param mode pass XYMODE if i j refer to ix, iy, SCMODULEMODE if thery refer to ism, imod
   */
  EKDetId(int module_ix, int module_iy, int fiber, int ro, int iz); 

  EKDetId(int i, int j, int fiber, int ro, int iz, int mode);
  
  EKDetId(const DetId& id) : DetId(id){}
  
  EKDetId& operator=(const DetId& id) {id_ = id.rawId(); return *this;}
  
  void setFiber(int fib, int ro);
  
  static EcalSubdetector subdet() { return EcalShashlik;}
  
  int zside() const { return (id_&0x200000)?(1):(-1); }
  
  int ix() const { return (id_>>8)&0xFF; }
  
  int iy() const { return id_&0xFF; }
  
  int fiber() const { return (id_>>16)&0x7; }
  
  int readout() const { return (id_>>19)&0x3; }


  static int ism(int ix, int iy);

  static int imod(int ix, int iy);
  


  /** Gets the SuperModule number
   */
  int ism() const;

  /** Gets supermodule index from supermodule ismCol:ismRow location
   * ismCol, ismRow = 0, 1, 2, ... for quadrant 1 of EK+
   * ismCol = ..., -2, -1  ismRow = 0, 1, 2, ... for quadrant 2 of EK+
   * and so on
   * assumes |ismCol|, |ismRow| < MAX_SM_SIZE (see .cc)
   */
  static int smIndex (int ismCol, int ismRow); 

  /** Gets module number inside SuperModule.
   * Module numbering withing a supermodule in each quadrant:
   * \verbatim
   *                       A y
   *  (Q2)                 |                    (Q1)
   *        5 10 15 20 25  |     5 10 15 20 25
   *        4  9 14 19 24  |     4  9 14 19 24
   *        3  8 13 18 23  |     3  8 13 18 23
   *        2  7 12 17 22  |     2  7 12 17 22
   *        1  6 11 16 21  |     1  6 11 16 21
   *                       |
   * ----------------------o---------------------------> x
   *                       |
   *        5 10 15 20 25  |     5 10 15 20 25
   *        4  9 14 19 24  |     4  9 14 19 24
   *        3  8 13 18 23  |     3  8 13 18 23
   *        2  7 12 17 22  |     2  9 12 17 22
   *        1  6 11 16 21  |     1  6 11 16 21
   *  (Q3)                                       (Q4)
   * \endverbatim
   *
   * @return module number from 1 to 25
   */
  int imod() const;
  
  /** Gets the quadrant of the DetId.
   * Quadrant number definition, x and y in std CMS coordinates, for EK+:
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
  
  /** Checks if module is in EK+
   * @return true for EK+, false for EK-
   */
  bool positiveZ() const {return (zside()>0);}
  
  /** Gets a compact index for arrays
   * @return compact index from 0 to kSizeForDenseIndexing-1
   */
  int hashedIndex() const;
  
  /** Same as hashedIndex()
   * @return compact index from 0 to kSizeForDenseIndexing-1
   */
  uint32_t denseIndex() const { return hashedIndex(); }
  
  /** Checks validity of a dense/hashed index
   * @param din dense/hashed index as returned by hashedIndex() or denseIndex()
   * method
   * @return true if index is valid, false otherwise
   */
  static bool validDenseIndex(uint32_t din) { return validHashIndex(din); }
  
  /** Converts a hashed/dense index as defined in hashedIndex() and denseIndex()
   * methods to a det id.
   * @param din hashed/dense index
   * @return det id
   */
  static EKDetId detIdFromDenseIndex(uint32_t din) { return unhashIndex(din); }

  /** Gets a DetId from a compact index for arrays. Converse of hashedIndex() method.
   * @param hi dense/hashed index
   * @return det id
   */
  static EKDetId unhashIndex( int hi ) ;
  
  /** Checks if a hashed/dense index is valid
   * @see hashedIndex(), denseIndex()
   * @param i hashed/dense index
   * @return true if the index is valid, false otherwise
   */
  static bool validHashIndex( int i );
  
  /** Checks validity of a module (SM,Module,fiber,RO,z) index.
   * @param SM supermodule index
   * @param Module module index
   * @param fiber fiber index
   * @param RO readout type
   * @param iz module z-index
   * @see EKDetId(int, int, int, int, int, int) for index definition
   * @return true if valid, false otherwise
   */
  
  /** returns a new EKDetId offset by nrStepsX and nrStepsY (can be negative),
   * returns EKDetId(0) if invalid */
  EKDetId offsetBy(int nrStepsX, int nrStepsY, Configuration conf = BlackBox) const;
  
  /** returns a new EKDetId swapped (same iX, iY) to the other endcap, 
   * returns EKDetId(0) if invalid (shouldnt happen) */
  EKDetId switchZSide() const;
  
  /** following are static member functions of the above two functions
   *  which take and return a DetId, returns DetId(0) if invalid 
   */
  static DetId offsetBy( const DetId startId, int nrStepsX, int nrStepsY, Configuration conf = BlackBox);
  static DetId switchZSide( const DetId startId );

  /** Returns the distance along x-axis in module units between two EKDetId
   * @param a det id of first module
   * @param b det id of second module
   * @return distance
   */
  static int distanceX(const EKDetId& a,const EKDetId& b);
  
  /** Returns the distance along y-axis in module units between two EKDetId
   * @param a det id of first module
   * @param b det id of second module
   * @return distance
   */
  static int distanceY(const EKDetId& a,const EKDetId& b); 
  
  
  /*@{*/
  /** function modes for EKDetId(int, int, int, int) constructor
   */
  static const int XYMODE       = 0;
  static const int SCMODULEMODE = 1;
  /*@}*/
  
  static bool validSM (int ix, int iy, Configuration conf = BlackBox);
  
  static bool validDetId(int iSM, int iMD, int fib, int ro, int iz, Configuration conf = BlackBox);
  
  static bool slowValidDetId(int module_ix, int module_iy, int fib, int ro, int iz, Configuration conf = BlackBox);
    
  static bool isNextToBoundary(EKDetId id, Configuration conf = BlackBox);
  
  static bool isNextToDBoundary(EKDetId id);
  
  static bool isNextToRingBoundary(EKDetId id, Configuration conf = BlackBox);

  static int smXLocation(int iSM);
  static int smYLocation(int iSM);
  int ix(int iSM, int iMod) const;
  int iy(int iSM, int iMod) const;
private:




  

};


std::ostream& operator<<(std::ostream& s,const EKDetId& id);

#endif
