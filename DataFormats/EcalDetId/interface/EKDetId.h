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
  
  /** Constructor from module ix,iy,fib,ro,iz (iz=+1/-1)
   * <p>ix runs from 1 to N along x-axis of standard CMS coordinates<br>
   * iy runs from 1 to N along y-axis of standard CMS coordinates<br>
   * N depends on the configuration == see ShashlikDDDConstants<br>
   * fib runs from 0 to 5 for fiber type (0 is combined)<br>
   * ro runs from 0 to 2 for read out type (0 is combined)<br>
   * iz is -1 for EK- and +1 for EK+<br>
   */
  EKDetId(int module_ix, int module_iy, int fiber, int ro, int iz); 
  
  /** Constructor from a generic cell id
   * @param id source detid
   */
  EKDetId(const DetId& id) : DetId(id) {}
  
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
};

std::ostream& operator<<(std::ostream& s,const EKDetId& id);

#endif
