#ifndef Alignment_CommonAlignment_AlignableBuilder_H
#define Alignment_CommonAlignment_AlignableBuilder_H

/** \class AlignableBuilder
 *
 *  A class to build alignable composites.
 *
 *  $Date: 2007/10/18 09:41:07 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/Counters.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"

class TrackerTopology;

class AlignableBuilder
{
  public:

  /// Init the module type.
  AlignableBuilder( align::StructureType moduleType, Counters& counter, const TrackerTopology* tTopo );

  /// Add info required to build a level to the list.
  void addLevelInfo( align::StructureType,
                     bool flat,
                     unsigned int maxComponent );

  /// Build the components for all levels given by user.
  void buildAll( AlignSetup<align::Alignables>& setup ) const;

  private:

  struct LevelInfo
  {
    align::StructureType type_; // level type
    bool flat_;                 // true if type is a flat surface (rod, string, ladder,...)
    unsigned int maxComponent_; // max no. of components in this level
    
    LevelInfo( align::StructureType,
               bool flat,
               unsigned int maxComponent );
  };

  /// Find max number of components for a given level in the hierarchy.
  unsigned int maxComponent( unsigned int level ) const;

  /// A way to index a component of a certain level in the hierarchy.
  /// Unique for each component of a given structure type.
  /// It starts from 0 and doesn't have to be continuous.
  unsigned int index( unsigned int level,
                      align::ID, const TrackerTopology* tTopo ) const;

  /// Build the components for a given level in the hierarchy.
  void build( unsigned int level, align::StructureType dauType, 
              AlignSetup<align::Alignables>& setup ) const;

  align::StructureType theModuleType; // starting level to build composites

  std::vector<LevelInfo> theLevelInfos;

  Counters theCounters;

  const TrackerTopology* theTopology;
};

#endif
