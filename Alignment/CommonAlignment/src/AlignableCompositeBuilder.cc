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
AlignableCompositeBuilder
::AlignableCompositeBuilder(const TrackerTopology* trackerTopology,
                            AlignableIndexer& alignableIndexer) :
  trackerTopology(trackerTopology),
  alignableIndexer(alignableIndexer)
{
}

//_____________________________________________________________________________
void AlignableCompositeBuilder
::addAlignmentLevel(AlignmentLevel* level) {
  alignmentLevels.push_back(level);
}

//_____________________________________________________________________________
void AlignableCompositeBuilder
::clearAlignmentLevels() {
  alignmentLevels.clear();
}

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder
::buildAll(AlignableMap& alignableMap)
{
  auto highestLevel = alignmentLevels.back()->levelType;

  std::ostringstream ss;
  ss << "building CompositeAlignables for "
     << AlignableObjectId::idToString(highestLevel) << "\n";

  unsigned int numCompositeAlignables = 0;
  for (unsigned int level = 1; level < alignmentLevels.size(); ++level) {
    numCompositeAlignables += buildLevel(level, alignableMap, ss);
  }

  ss << "built " << numCompositeAlignables << " CompositeAlignables for "
     << AlignableObjectId::idToString(highestLevel);
  edm::LogInfo("AlignableBuildProcess")
    << "@SUB=AlignableCompositeBuilder::buildAll" << ss.str();

  return numCompositeAlignables;
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder
::buildLevel(unsigned int parentLevel,
             AlignableMap& alignableMap,
             std::ostringstream& ss)
{
  unsigned int childLevel    = parentLevel - 1;
  unsigned int maxNumParents = maxNumComponents(parentLevel);

  auto childType  = alignmentLevels[childLevel] ->levelType;
  auto parentType = alignmentLevels[parentLevel]->levelType;

  auto& children = alignableMap.find(AlignableObjectId::idToString(childType));
  auto& parents  = alignableMap.get (AlignableObjectId::idToString(parentType));
  parents.reserve(maxNumParents);

  // This vector is used indicate if a parent already exists. It is initialized
  // with 'naked' Alignables-pointers; if the pointer is not naked (!= 0) for
  // one of the child-IDs, its parent was already built before.
  Alignables tmpParents(maxNumParents, 0);

  for (auto* child : children) {
    // get the number of the child-Alignable ...
    unsigned int index = getIndexOfStructure(child->id(), parentLevel);
    // ... and use it as index to get the parent of this child
    Alignable*& parent = tmpParents[index];

    // if parent was not built yet ...
    if (parent == 0) {
      // ... built new composite Alignable with ID of child (obviously its the
      // first child of the Alignable)
      if (alignmentLevels[parentLevel]->isFlat) {
        parent = new AlignableComposite(child->id(), parentType,
                                        child->globalRotation());
      } else {
        parent = new AlignableComposite(child->id(), parentType,
                                        align::RotationType());
      }
      parents.push_back(parent);
    }

    // in all cases add the child to the parent Alignable
    parent->addComponent(child);
  }

  ss << "   built " << parents.size() << " "
     << AlignableObjectId::idToString(alignmentLevels[parentLevel]->levelType)
     << "(s) (theoretical maximum: " << maxNumParents
     << ") consisting of " << children.size() << " "
     << AlignableObjectId::idToString(alignmentLevels[childLevel]->levelType)
     << "(s)\n";

  return parents.size();
}

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder
::maxNumComponents(unsigned int startLevel) const
{
  unsigned int components = 1;

  for (unsigned int level = startLevel;
       level < alignmentLevels.size();
       ++level) {
    components *= alignmentLevels[level]->maxNumComponents;
  }

  return components;
}

//_____________________________________________________________________________
unsigned int AlignableCompositeBuilder
::getIndexOfStructure(align::ID id, unsigned int level) const
{
  // indexer returns a function pointer for the structure-type
  auto indexOf = alignableIndexer.get(alignmentLevels[level]->levelType);

  if (alignmentLevels.size() - 1 > level) {
    return getIndexOfStructure(id, level + 1)
             * alignmentLevels[level]->maxNumComponents
             + indexOf(id, trackerTopology) - 1;
  }

  return indexOf(id, trackerTopology) - 1;
}
