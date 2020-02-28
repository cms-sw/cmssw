#include "Alignment/CommonAlignment/interface/AlignableCompositeBuilder.h"

// Original Author:  Max Stark
//         Created:  Thu, 13 Jan 2016 10:22:57 CET

// core framework functionality
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// alignment
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

//_____________________________________________________________________________
AlignableCompositeBuilder ::AlignableCompositeBuilder(const TrackerTopology* trackerTopology,
                                                      const TrackerGeometry* trackerGeometry,
                                                      const AlignableIndexer& alignableIndexer)
    : trackerTopology_(trackerTopology),
      alignableObjectId_(trackerGeometry, nullptr, nullptr),
      alignableIndexer_(alignableIndexer) {}

//_____________________________________________________________________________
void AlignableCompositeBuilder ::addAlignmentLevel(std::unique_ptr<AlignmentLevel> level) {
  alignmentLevels_.push_back(std::move(level));
}

//_____________________________________________________________________________
void AlignableCompositeBuilder ::clearAlignmentLevels() { alignmentLevels_.clear(); }

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder ::buildAll(AlignableMap& alignableMap, bool update) {
  auto highestLevel = alignmentLevels_.back()->levelType;

  std::ostringstream ss;
  ss << "building CompositeAlignables for " << alignableObjectId_.idToString(highestLevel) << "\n";

  unsigned int numCompositeAlignables = 0;
  for (unsigned int level = 1; level < alignmentLevels_.size(); ++level) {
    numCompositeAlignables += buildLevel(level, alignableMap, ss, update);
  }

  ss << "built " << numCompositeAlignables << " CompositeAlignables for "
     << alignableObjectId_.idToString(highestLevel);
  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableCompositeBuilder::buildAll" << ss.str();

  return numCompositeAlignables;
}

//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder ::buildLevel(unsigned int parentLevel,
                                                    AlignableMap& alignableMap,
                                                    std::ostringstream& ss,
                                                    bool update) {
  unsigned int childLevel = parentLevel - 1;
  unsigned int maxNumParents = maxNumComponents(parentLevel);

  auto childType = alignmentLevels_[childLevel]->levelType;
  auto parentType = alignmentLevels_[parentLevel]->levelType;

  auto& children = alignableMap.find(alignableObjectId_.idToString(childType));
  auto& parents = alignableMap.get(alignableObjectId_.idToString(parentType));
  if (!update)
    parents.reserve(maxNumParents);

  // This vector is used indicate if a parent already exists. It is initialized
  // with 'naked' Alignables-pointers; if the pointer is not naked (!= nullptr)
  // for one of the child-IDs, its parent was already built before.
  align::Alignables tmpParents(maxNumParents, nullptr);

  for (auto* child : children) {
    // get the number of the child-Alignable ...
    const auto index = getIndexOfStructure(child->id(), parentLevel);
    // ... and use it as index to get the parent of this child
    auto& parent = tmpParents[index];

    // if parent was not built yet ...
    if (!parent && !update) {
      if (update) {
        throw cms::Exception("LogicError") << "@SUB=AlignableCompositeBuilder::buildLevel\n"
                                           << "trying to update a non-existing AlignableComposite";
      }
      // ... build new composite Alignable with ID of child (obviously its the
      // first child of the Alignable)
      if (alignmentLevels_[parentLevel]->isFlat) {
        parent = new AlignableComposite(child->id(), parentType, child->globalRotation());
      } else {
        parent = new AlignableComposite(child->id(), parentType, align::RotationType());
      }
      parents.push_back(parent);
    } else if (update) {
      if (alignmentLevels_[parentLevel]->isFlat) {
        // needed to update rotation of flat composites
        auto mother = dynamic_cast<AlignableComposite*>(child->mother());
        if (!mother) {
          throw cms::Exception("LogicError") << "@SUB=AlignableCompositeBuilder::buildLevel\n"
                                             << "trying to update a flat composite that is not of type "
                                             << "AlignableComposite";
        }
        if (mother->id() == child->id()) {
          mother->update(child->id(), parentType, child->globalRotation());
        }
      }
    }

    // in all cases (except updates) add the child to the parent Alignable
    if (!update)
      parent->addComponent(child);
  }

  ss << "   built " << parents.size() << " " << alignableObjectId_.idToString(alignmentLevels_[parentLevel]->levelType)
     << "(s) (theoretical maximum: " << maxNumParents << ") consisting of " << children.size() << " "
     << alignableObjectId_.idToString(alignmentLevels_[childLevel]->levelType) << "(s)\n";

  return parents.size();
}

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder ::maxNumComponents(unsigned int startLevel) const {
  unsigned int components = 1;

  for (unsigned int level = startLevel; level < alignmentLevels_.size(); ++level) {
    components *= alignmentLevels_[level]->maxNumComponents;
  }

  return components;
}

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder ::getIndexOfStructure(align::ID id, unsigned int level) const {
  // indexer returns a function pointer for the structure-type
  auto indexOf = alignableIndexer_.get(alignmentLevels_[level]->levelType, alignableObjectId_);

  if (alignmentLevels_.size() - 1 > level) {
    return getIndexOfStructure(id, level + 1) * alignmentLevels_[level]->maxNumComponents + indexOf(id) - 1;
  }

  return indexOf(id) - 1;
}
