/** \file LASAlignmentParameterCollection.cc
 *  
 *
 *  $Date: 2007/03/20 16:53:24 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "DataFormats/LaserAlignment/interface/LASAlignmentParameterCollection.h"

// put in LASAlignmentParameter of detector label name_
void LASAlignmentParameterCollection::put(Range input, std::string name_)
{
  // store size of vector before put
  IndexRange inputRange;

  // put in LASAlignmentParameter from input
  bool first = true;

  // iterators over input
  LASAlignmentParameterCollection::ContainerIterator begin = input.first;
  LASAlignmentParameterCollection::ContainerIterator end = input.second;

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
  map_[name_] = inputRange;
}

// get LASAlignmentParameter of detector label
const LASAlignmentParameterCollection::Range LASAlignmentParameterCollection::get(std::string name_) const
{
  LASAlignmentParameterCollection::IndexRange returnIndexRange = map_[name_];

  LASAlignmentParameterCollection::Range returnRange;
  returnRange.first = container_.begin() + returnIndexRange.first;
  returnRange.second = container_.begin() + returnIndexRange.second;

  return returnRange;
}

// returns vector of detector labels in the map
const std::vector<std::string> LASAlignmentParameterCollection::names() const
{
  LASAlignmentParameterCollection::RegistryIterator begin = map_.begin();
  LASAlignmentParameterCollection::RegistryIterator end = map_.end();

  std::vector<std::string> output;

  for ( ; begin != end; ++begin)
    {
      output.push_back(begin->first);
    }

  return output;
}

// appends LASAlignmentParameters to the vector of the given detector label
void LASAlignmentParameterCollection::add(std::string& name_, std::vector<LASAlignmentParameter>& alignmentParameters)
{
  parameterMap_[name_].reserve( parameterMap_[name_].size() + alignmentParameters.size() );

  if ( parameterMap_[name_].empty() )
    {
      parameterMap_[name_] = alignmentParameters;
    }
  else
    {
      copy( alignmentParameters.begin(), alignmentParameters.end(), back_inserter(parameterMap_[name_]) );
    }
}

// returns (by reference) the LASAlignmentParameter for a given detector label
void LASAlignmentParameterCollection::alignmentParameter(std::string& name_, std::vector<LASAlignmentParameter>& alignmentParameters) const
{
  if ( parameterMap_.find(name_) != parameterMap_.end() )
    {
      alignmentParameters = parameterMap_[name_];
    }
  else
    {
      alignmentParameters = std::vector<LASAlignmentParameter>();
    }
}

// returns (by reference) vector of detector labels with a LASAlignmentParameter
void LASAlignmentParameterCollection::names(std::vector<std::string>& names_) const
{
  names_.clear();
  names_.reserve(static_cast<unsigned int>(parameterMap_.size()));

  LASAlignmentParameterContainer::const_iterator iter;
  for (iter = parameterMap_.begin(); iter != parameterMap_.end(); iter++)
    {
      names_.push_back( iter->first );
    }
}

// returns the size of the collection
int LASAlignmentParameterCollection::size() const
{
  return map_.size();
}
