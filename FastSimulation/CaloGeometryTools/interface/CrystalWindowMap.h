#ifndef FastSimulation_CaloGeometryTools_CrystalWindowMap_h
#define FastSimulation_CaloGeometryTools_CrystalWindowMap_h

/** 
 *  This class is used by EcalHitMaker to determine quickly in which Pad each
 *  spots falls. 
 *  Each crystal is labelled by its number. Given a crystal number, several methods
 *  allow to get the ordered (by distance) list of the neighbouring crystals
 *  
 * \author Florian Beaudette
 * \date: 08-Jun-2004
 * \date: 05-Oct-2006
 */

// FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/Crystal.h"

//C++ headers 
#include <vector>

class CaloGeometryHelper;

class CrystalWindowMap
{
 public:
  /// Constructor from vector of Crystal
   CrystalWindowMap(const CaloGeometryHelper *, const std::vector<Crystal> & cw);
  ~CrystalWindowMap(){;};

  /// get the ordered list of the crystals around the crystal given as a first argument
  bool getCrystalWindow(unsigned , std::vector<unsigned>&  ) const ;
  /// same thing but with a different interface
  bool getCrystalWindow(unsigned iq,const std::vector<unsigned>*  cw) const;
  /// same thing but with a different interface
  const std::vector<unsigned>& getCrystalWindow(unsigned, bool& status) const;
  inline unsigned size() const {return size_;}

 private:
  const CaloGeometryHelper * myCalorimeter_;
  
  unsigned size_;
  const std::vector<Crystal>& originalVector_;

  std::vector< std::vector<unsigned> > myNeighbours_; 
};

#endif
