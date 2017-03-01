#ifndef ALIGNMENT_COMMONALIGNMENT_INTERFACE_ALIGNABLECOMPOSITEBUILDER_H_
#define ALIGNMENT_COMMONALIGNMENT_INTERFACE_ALIGNABLECOMPOSITEBUILDER_H_

// Original Author:  Max Stark
//         Created:  Wed, 10 Feb 2016 14:02:51 CET

// alignment
#include "Alignment/CommonAlignment/interface/AlignableMap.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableIndexer.h"
#include "Alignment/CommonAlignment/interface/AlignmentLevel.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"



class AlignableCompositeBuilder {

  //========================== PUBLIC METHODS =================================
  public: //===================================================================

    AlignableCompositeBuilder(const TrackerTopology*, const TrackerGeometry*,
                              const AlignableIndexer&);
    virtual ~AlignableCompositeBuilder() = default;

    /// Add all desired AlignmentLevels for a sub-detector to the builder before
    /// calling buildAll(), the order matters!
    /// Example for PixelBarrel-RunI geometry:
    /// -> PXBModule, PXBLadder, TPBLayer, TPBHalfBarrel, TPBBarrel
    void addAlignmentLevel(std::unique_ptr<AlignmentLevel> level);

    /// Resets the alignment-levels.
    void clearAlignmentLevels();

    /// Builds all composite Alignables according to the levels added before via
    /// addAlignmentLevel(). The Alignables were built from bottom- to the top-
    /// hierarchy, e.g. for PixelBarrel-RunI geometry:
    /// - PXBLadder     (with PXBModule as children)
    /// - TPBLayer      (with PXBLadder as children)
    /// - TPBHalfBarrel (with TPBLayer as children)
    /// - TPBBarrel     (with TPBHalfBarrel as children)
    /// Returns the number of composite Alignables which were built.
    unsigned int buildAll(AlignableMap&);

    /// Return tracker alignable object ID provider derived from the tracker's geometry
    const AlignableObjectId& objectIdProvider() const { return alignableObjectId_; }

  //========================= PRIVATE METHODS =================================
  private: //==================================================================

    /// Builds the components for a given level in the hierarchy.
    unsigned int buildLevel(unsigned int parentLevel, AlignableMap&,
                            std::ostringstream&);

    /// Calculates the theoretical max. number of components for a given level
    /// in the hierarchy.
    unsigned int maxNumComponents(unsigned int startLevel) const;

    /// Calculates the index of an Alignable within the hierarchy; unique for
    /// each component of a given structure-type.
    unsigned int getIndexOfStructure(align::ID, unsigned int level) const;

  //========================== PRIVATE DATA ===================================
  //===========================================================================

    // TODO: The AlignableCompositeBuilder is not 'common' as the package
    //       suggests, because it uses the TrackerTopology. If this class shall
    //       ever be used to build other kinds of alignables than tracker-
    //       alignables one has to add/implement something more general than
    //       the TrackerTopology
    const TrackerTopology* trackerTopology_;
    const AlignableObjectId alignableObjectId_;

    AlignableIndexer alignableIndexer_;

    align::AlignmentLevels alignmentLevels_;

};

#endif /* ALIGNMENT_COMMONALIGNMENT_INTERFACE_ALIGNABLECOMPOSITEBUILDER_H_ */
