#ifndef GEOMETRY_CALOTOPOLOGY_CASTORTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_CASTORTOPOLOGY_H 1

#include <vector>
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

/** \class CastorTopology

   \author P. Katsas - UoA
*/

class CastorTopology : public CaloSubdetectorTopology {
public:

  CastorTopology();
  /** Exlucde a cell*/
  void exclude(const HcalCastorDetId& id);
 /** Exclude a side*/
  void exclude(int zside); 
 /** Exclude a section, in either side (+1 positive, -1 negative)*/
  void exclude(int zside, HcalCastorDetId::Section section);
   /** Exclude a range of channels (deph) for a given subdetector*/
  int exclude(int zside, HcalCastorDetId::Section section1, int isec1, int imod1, HcalCastorDetId::Section section2, int isec2, int imod2);

  /** Is this a valid cell id? */
  using CaloSubdetectorTopology::valid;
  virtual bool valid(const HcalCastorDetId& id) const;

  /** Is this a valid cell id? */
  virtual bool validRaw(const HcalCastorDetId& id) const;

  /** Get the neighbors of the given cell with higher #sector */
  virtual std::vector<DetId> incSector(const DetId& id) const;
  
  /** Get the neigbors of the given cell with higher #module*/
  virtual std::vector<DetId> incModule(const DetId& id) const;

  //** I have to put this here since they inherit from CaloSubdetectorTopology
  std::vector<DetId> east(const DetId& id) const override;
  std::vector<DetId> west(const DetId& id) const override; 
  std::vector<DetId> north(const DetId& id) const override;
  std::vector<DetId> south(const DetId& id) const override;
  std::vector<DetId> up(const DetId& id) const override;
  std::vector<DetId> down(const DetId& id) const override;
  
  // how many channels (deph) for a given section
  using CaloSubdetectorTopology::ncells;
  int ncells(HcalCastorDetId::Section section) const;

  //return first and last cell of each section
  int firstCell(HcalCastorDetId::Section section)const;  
  int lastCell(HcalCastorDetId::Section section)const;
 
 private:
  
  std::vector<HcalCastorDetId> exclusionList_;
  
  bool excludeEM_, excludeHAD_, excludeZP_, excludeZN_;
  
  int firstEMModule_, lastEMModule_, firstHADModule_, lastHADModule_;
   
  bool isExcluded(const HcalCastorDetId& id) const;
  
  int firstEMModule() const {return firstEMModule_;}
  int firstHADModule() const {return firstHADModule_;}  
  int lastEMModule()  const {return lastEMModule_;}
  int lastHADModule() const {return lastHADModule_;}  

};


#endif

