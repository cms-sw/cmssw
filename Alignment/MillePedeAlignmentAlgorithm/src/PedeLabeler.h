#ifndef PEDELABELER_H
#define PEDELABELER_H

/**
 * \class PedeLabeler
 *
 * provides labels for AlignmentParameters for pede
 *
 * \author    : Gero Flucke
 * date       : September 2007
 * $Date: 2007/08/31 17:24:28 $
 * $Revision: 1.1 $
 * (last update by $Author: flucke $)
 */

#include <vector>
#include <map> 


class Alignable;

/***************************************
****************************************/
class PedeLabeler
{
 public:
  /// constructor from array of Alignables 
  PedeLabeler(const std::vector<Alignable*> &alis);
  /// constructor from two Alignables (null pointers allowed )
  PedeLabeler(Alignable *ali1, Alignable *ali2);
  /** non-virtual destructor: do not inherit from this class **/
  ~PedeLabeler();
    
  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  unsigned int alignableLabel(Alignable *alignable) const;
  unsigned int lasBeamLabel(unsigned int lasBeamId) const;
  unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const;
  
  /// parameter number, 0 <= .. < theMaxNumParam, belonging to unique parameter label
  unsigned int paramNumFromLabel(unsigned int paramLabel) const;
  /// alignable label from parameter label (works also for alignable label...)
  unsigned int alignableLabelFromLabel(unsigned int label) const;
  /// Alignable from alignable or parameter label,
  /// null if no alignable (but error only if not las beam, either!)
  Alignable* alignableFromLabel(unsigned int label) const;
  /// las beam id from las beam or parameter label
  /// zero and error if not a valid las beam label
  unsigned int lasBeamIdFromLabel(unsigned int label) const;

 private:
  typedef std::map <Alignable*, unsigned int> AlignableToIdMap;
  typedef AlignableToIdMap::value_type AlignableToIdPair;
  typedef std::map <unsigned int, Alignable*> IdToAlignableMap;
  typedef std::map <unsigned int, unsigned int> UintUintMap;


  /// returns size of map
  unsigned int buildMap(const std::vector<Alignable*> &alis);
  /// returns size of map
  unsigned int buildReverseMap();

  // data members
  AlignableToIdMap  myAlignableToIdMap; /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap  myIdToAlignableMap; /// reverse map
  UintUintMap       myLasBeamToLabelMap;  /// labels for las beams
  UintUintMap       myLabelToLasBeamMap; /// reverse of the above

  static const unsigned int theMaxNumParam;
  static const unsigned int theMinLabel;
};

#endif
