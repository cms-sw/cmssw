#ifndef Geometry_ForwardGeometry_ZdcTopology_H
#define Geometry_ForwardGeometry_ZdcTopology_H 1

#include <vector>
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

/** \class ZDCTopology

   $Date: 2007/08/28 18:10:10 $
   $Revision: 1.1 $
   \author E. Garcia - UIC
*/

class ZdcTopology : public CaloSubdetectorTopology {
public:

  ZdcTopology();
  /** Exlucde a cell*/
  void exclude(const HcalZDCDetId& id);
 /** Exclude a side*/
  void exclude(int zside); 
 /** Exclude a section, in either side (+1 positive, -1 negative)*/
  void exclude(int zside, HcalZDCDetId::Section section);
   /** Exclude a range of channels (deph) for a given subdetector*/
  int exclude(int zside, HcalZDCDetId::Section section, int ich1, int ich2);

  /** Is this a valid cell id? */
  virtual bool valid(const HcalZDCDetId& id) const;

  /** Get the transverse (X) neighbors of the given cell*/
  virtual std::vector<DetId> transverse(const DetId& id) const;
  
  /** Get the longitudinal neighbors (Z) of the given cell*/
  virtual std::vector<DetId> longitudinal(const DetId& id) const;

  //** I have to put this here since they inherit from CaloSubdetectorTopology
  virtual std::vector<DetId> east(const DetId& id) const;
  virtual std::vector<DetId> west(const DetId& id) const; 
  virtual std::vector<DetId> north(const DetId& id) const;
  virtual std::vector<DetId> south(const DetId& id) const;
  virtual std::vector<DetId> up(const DetId& id) const;
  virtual std::vector<DetId> down(const DetId& id) const;
  
  
  // how many channels (deph) for a given section
  int ncells(HcalZDCDetId::Section section) const;

  //return first and last cell of each section
  int firstCell(HcalZDCDetId::Section section)const;  
  int lastCell(HcalZDCDetId::Section section)const;
 
 private:
  
  bool validRaw(const HcalZDCDetId& id) const;
  
  std::vector<HcalZDCDetId> exclusionList_;
  
  bool excludeEM_, excludeHAD_, excludeLUM_, excludeZP_, excludeZN_;
  
  int firstEMModule_, lastEMModule_, firstHADModule_, lastHADModule_, 
    firstLUMModule_, lastLUMModule_;
   
  bool isExcluded(const HcalZDCDetId& id) const;
  
  int firstEMModule() const {return firstEMModule_;}
  int firstHADModule() const {return firstHADModule_;}  
  int firstLUMModule() const {return firstLUMModule_;}
  int lastEMModule()  const {return lastEMModule_;}
  int lastHADModule() const {return lastHADModule_;}  
  int lastLUMModule() const {return lastLUMModule_;}

};


#endif
