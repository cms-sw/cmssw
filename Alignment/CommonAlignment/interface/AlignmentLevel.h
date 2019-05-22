#ifndef ALIGNMENT_COMMONALIGNMENT_INTERFACE_ALIGNMENTLEVEL_H_
#define ALIGNMENT_COMMONALIGNMENT_INTERFACE_ALIGNMENTLEVEL_H_

// Original Author:  Max Stark
//         Created:  Wed, 10 Feb 2016 13:35:23 CET

#include "Alignment/CommonAlignment/interface/StructureType.h"

class AlignmentLevel {
  //========================== PUBLIC METHODS =================================
public:  //===================================================================
  AlignmentLevel(align::StructureType levelType, unsigned int maxNumComponents, bool isFlat)
      : levelType(levelType), maxNumComponents(maxNumComponents), isFlat(isFlat){};
  // copy construction + assignment
  AlignmentLevel(const AlignmentLevel&) = default;
  AlignmentLevel& operator=(const AlignmentLevel&) = default;

  // move construction + assignment
  AlignmentLevel(AlignmentLevel&&) = default;
  AlignmentLevel& operator=(AlignmentLevel&&) = default;

  virtual ~AlignmentLevel() = default;

  //=========================== PUBLIC DATA ===================================
  //===========================================================================

  /// the structure-type for this level,
  /// e.g. TPBModule for RunI-tracker-PXB
  align::StructureType levelType;

  /// the maximum number of components of the structure-type,
  /// e.g. 768 TPBModules in RunI tracker-PXB
  unsigned int maxNumComponents;

  /// true if structure-type is a flat surface (rod, string, ladder etc.)
  bool isFlat;
};

#endif /* ALIGNMENT_COMMONALIGNMENT_INTERFACE_ALIGNMENTLEVEL_H_ */
