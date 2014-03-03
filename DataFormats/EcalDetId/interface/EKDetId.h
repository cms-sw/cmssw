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
  
  /** Constructor of a null id
   */
  EKDetId() {}
  
  /** Constructor from a raw value
      @param rawid det ID number 
  */
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
  // fast  
  EKDetId(int module_ix, int module_iy, int fiber, int ro, int iz); 
  // slow
  EKDetId(int i, int j, int fiber, int ro, int iz, int mode);
  
  /** Constructor from a generic cell id
   * @param id source detid
   */
  EKDetId(const DetId& id) : DetId(id){}
  
  /** Assignment operator
   * @param id source det id
   */ 
  EKDetId& operator=(const DetId& id) {id_ = id.rawId(); return *this;}
  
  /** Set fiber number and RO type
   * @param fib number
   * @param ro  readout type
   */
  void setFiber(int fib, int ro);
  
  /** Gets the subdetector
   * @return subdetectot ID, that is EcalEndcap
   */
  static EcalSubdetector subdet() { return EcalShashlik;}
  
  /** Gets the z-side of the module (1/-1)
   * @return -1 for EK-, +1 for EK+
   */
  int zside() const { return (id_&0x200000)?(1):(-1); }
  
  /** Gets the module x-index.
   * @see EKDetId(int, int, int, int, int) for x-index definition
   * @return x-index
   */
  int ix() const { return (id_>>8)&0xFF; }
  
  /** Get the module y-index
   * @see EKDetId(int, int, int, int, int) for y-index definition.
   * @return y-index
   */
  int iy() const { return id_&0xFF; }
  
  /** Get the fiber-index
   * @see EKDetId(int, int, int, int, int) for fiber-index definition.
   * @return fiber-index
   */
  int fiber() const { return (id_>>16)&0x7; }
  
  /** Get the readout-index
   * @see EKDetId(int, int, int, int, int) for readout-index definition.
   * @return readout-index
   */
  int readout() const { return (id_>>19)&0x3; }
  
  /** Gets the SuperModule number within the endcap. This number runs from 1 to 936.
   *
   * BEWARE: This number is not consistent with indices used in constructor:  see details below.
   *
   * Numbering in quadrant 1 of EK+ is the following
   * \verbatim 
   *  13 27
   *  12 26 40 54 70 87 104
   *  11 25 39 53 69 86 103 120 136
   *  10 24 38 52 68 85 102 119 135 151 166
   *  09 23 37 51 67 84 101 118 134 150 165 180
   *  08 22 36 50 66 83 100 117 133 149 164 179 193
   *  07 21 35 49 65 82  99 116 132 148 163 178 192 205
   *  06 20 34 48 64 81  98 115 131 147 162 177 191 204
   *  05 19 33 47 63 80  97 114 130 146 161 176 190 203 215
   *  04 18 32 46 62 79  96 113 129 145 160 175 189 202 214 225
   *  03 17 31 45 61 78  95 112 128 144 159 174 188 201 213 224
   *  02 16 30 44 60 77  94 111 127 143 158 173 187 200 212 223 232
   *  01 15 29 43 59 76  93 110 126 142 157 172 186 199 211 222 231
   *     14 28 42 58 75  92 109 125 141 156 171 185 198 210 221 230
   *           41 57 74  91 108 124 140 155 170 184 197 209 220 229
   *              56 73  90 107 123 139 154 169 183 196 208 219 228
   *              55 72  89 106 122 138 153 168 182 195 207 218 227 234
   *                 71  88 105 121 137 152 167 181 194 206 217 226 233
   *  
   * \endverbatim
   *
   * Quadrant 2 indices are deduced by a symmetry about y-axis and by adding an offset
   * of 234.<br>
   * Quadrant 3 and 4 indices are deduced from quadrant 1 and 2 by a symmetry
   * about x-axis and adding an offset. Quadrant N starts with index 1 + (N-1)*234.
   *
   * <p>EK- indices are deduced from EK+ by a symmetry about (x,y)-plane (mirrored view). <b>It is
   * inconsistent with indices used in constructor EKDetId(int, int,int) in
   * SCMODULEMODE</b>. Indices of constructor uses a symmetry along y-axis: in principal it
   * considers the ism as a local index. The discrepancy is most probably due to a bug in the
   * implementation of this ism() method.
   */
  int ism() const;
  
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
  static bool validHashIndex( int i ) { return ( i < kSizeForDenseIndexing ) ; }
  
  /** Checks validity of a module (SM,Module,fiber,RO,z) index.
   * @param SM supermodule index
   * @param Module module index
   * @param fiber fiber index
   * @param RO readout type
   * @param iz module z-index
   * @see EKDetId(int, int, int, int, int, int) for index definition
   * @return true if valid, false otherwise
   */
  static bool validDetId(int iSM, int iMD, int fib, int ro, int iz);
  
  /** Checks validity of a module (module_ix,module_iy,fiber,RO,z) index.
   * @param module_ix module x-index
   * @param module_iy module y-index
   * @param fiber fiber index
   * @param RO readout type
   * @param iz module z-index
   * @see EKDetId(int, int, int, int, int, int) for index definition
   * @return true if valid, false otherwise
   */
  bool slowValidDetId(int module_ix, int module_iy, int fib, int ro, int iz) const;
    
  static bool isNextToBoundary(EKDetId id);
  
  static bool isNextToDBoundary(EKDetId id);
  
  static bool isNextToRingBoundary(EKDetId id);
  
  /** returns a new EKDetId offset by nrStepsX and nrStepsY (can be negative),
   * returns EKDetId(0) if invalid */
  EKDetId offsetBy(int nrStepsX, int nrStepsY) const;
  
  /** returns a new EKDetId swapped (same iX, iY) to the other endcap, 
   * returns EKDetId(0) if invalid (shouldnt happen) */
  EKDetId switchZSide() const;
  
  /** following are static member functions of the above two functions
   *  which take and return a DetId, returns DetId(0) if invalid 
   */
  static DetId offsetBy( const DetId startId, int nrStepsX, int nrStepsY );
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
  
  /** Lower bound of supermodule index as defined in ism()
   */
  static const int ISM_MIN=1;

  /** Upper bound of supermodule index defined in ism()
   */
  static const int ISM_MAX=936;
  
  /** Lower bound of module index within a supermodule
   */
  static const int IMOD_MIN=1;
  
  /** Upper bound of module index within a supermodule
   */
  static const int IMOD_MAX=25;
  
  /** Lower bound of EK module x-index
   */
  static const int IX_MIN=1;
  
  /** Upper bound of EK module y-index
   */
  static const int IX_MAX=180;
  
  /** Lower bound of EK module y-index
   */
  static const int IY_MIN=1;
  
  /** Upper bound of EK module y-index
   */
  static const int IY_MAX=180;
  
  /** Maximum possibility of Fiber number (0:FIB_MAX-1)
   */
  static const int FIB_MAX=6;
  
  /** Maximum possibility of Read-Out type (0:RO_MAX-1)
   */
  static const int RO_MAX=3;
  
  enum {
    /** Number of modules per Dee
     */
    kEKhalf = 421200 ,
    /** Number of dense module indices, that is number of
     * modules per endcap.
     */
    kSizeForDenseIndexing = 2*kEKhalf
  };
  
  /*@{*/
  /** function modes for EKDetId(int, int, int, int) constructor
   */
  static const int XYMODE       = 0;
  static const int SCMODULEMODE = 1;
  /*@}*/
  
  /** Gives supermodule index from endcap *supermodule* x and y indexes.
   * @see ism() for the index definition
   * @param ismCol supermodule column number: supemodule x-index for EK+
   * @param ismRow: supemodule y-index
   * @return supercystal index
   */
  static int ism(int ismCol, int ismRow);   // output is 1-936
  static int imod(int jx, int jy);
  
private:
  
  static const int nCols = 18;
  static const int nMods = 5; /* Number of modules per row in SM */
  static const int nRows = (nCols*nMods);
  static const int QuadColLimits[nCols+1];
  static const int iYoffset[nCols+1];
  
  int ix(int iSM, int iMod) const;
  int iy(int iSM, int iMod) const;
};


std::ostream& operator<<(std::ostream& s,const EKDetId& id);

#endif
