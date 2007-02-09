//                              -*- Mode: C++ -*- 
// LASBeamProfileFitCollection.cc --- Collection of LASBeamProfileFits
// Author          : Maarten Thomas
// Created On      : Wed Apr  5 10:12:52 2006
// Last Modified By: Maarten Thomas
// Last Modified On: Wed Jun 14 13:02:55 2006
// Update Count    : 10
// Status          : Unknown, Use with caution!
// 

#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"

// put in LASBeamProfileFit of detID
void LASBeamProfileFitCollection::put(Range input, unsigned int detID)
{
  // store size of vector before put
  IndexRange inputRange;

  // put in LASBeamProfileFit from input
  bool first = true;

  // iterators over input
  LASBeamProfileFitCollection::ContainerIterator begin = input.first;
  LASBeamProfileFitCollection::ContainerIterator end = input.second;

  for ( ; begin != end; ++begin)
    {
      container_.push_back(*begin);
      if (first)
	{
	  inputRange.first = container_.size() - 1;
	  first = false;
	}
    }
  inputRange.second = container_.size() - 1;

  // fill map
  map_[detID] = inputRange;
}

// get LASBeamProfileFit of detID
const LASBeamProfileFitCollection::Range LASBeamProfileFitCollection::get(unsigned int detID) const
{
  LASBeamProfileFitCollection::IndexRange returnIndexRange = map_[detID];

  LASBeamProfileFitCollection::Range returnRange;
  returnRange.first = container_.begin() + returnIndexRange.first;
  returnRange.second = container_.begin() + returnIndexRange.second;

  return returnRange;
}

// returns vector of detIDs in the map
const std::vector<unsigned int> LASBeamProfileFitCollection::detIDs() const
{
  LASBeamProfileFitCollection::RegistryIterator begin = map_.begin();
  LASBeamProfileFitCollection::RegistryIterator end = map_.end();

  std::vector<unsigned int> output;

  for ( ; begin != end; ++begin)
    {
      output.push_back(begin->first);
    }

  return output;
}

// appends LASBeamProfileFits to the vector of the given detID
void LASBeamProfileFitCollection::add(unsigned int& det_id, std::vector<LASBeamProfileFit>& beamProfileFit)
{
  fitMap_[det_id].reserve( fitMap_[det_id].size() + beamProfileFit.size() );

  if ( fitMap_[det_id].empty() )
    {
      fitMap_[det_id] = beamProfileFit;
    }
  else
    {
      copy( beamProfileFit.begin(), beamProfileFit.end(), back_inserter(fitMap_[det_id]) );
    }
}

// returns (by reference) the LASBeamProfileFit for a given DetId
void LASBeamProfileFitCollection::beamProfileFit(unsigned int& det_id, std::vector<LASBeamProfileFit>& beamProfileFit) const
{
  if ( fitMap_.find(det_id) != fitMap_.end() )
    {
      beamProfileFit = fitMap_[det_id];
    }
  else
    {
      beamProfileFit = std::vector<LASBeamProfileFit>();
    }
}

// returns (by reference) vector of DetIds with a LASBeamProfileFit
void LASBeamProfileFitCollection::detIDs(std::vector<unsigned int>& det_ids) const
{
  det_ids.clear();
  det_ids.reserve(static_cast<unsigned int>(fitMap_.size()));

  LASBeamProfileFitContainer::const_iterator iter;
  for (iter = fitMap_.begin(); iter != fitMap_.end(); iter++)
    {
      det_ids.push_back( iter->first );
    }
}

// returns the size of the collection
int LASBeamProfileFitCollection::size() const
{
  return map_.size();
}
