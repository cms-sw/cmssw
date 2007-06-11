#ifndef DataFormats_LaserAlignment_LASAlignmentParameterCollection_h
#define DataFormats_LaserAlignment_LASAlignmentParameterCollection_h

/** \class LASAlignmentParameterCollection
 *  Collection of LASAlignmentParameter sets
 *
 *  $Date: 2007/04/05 13:19:24 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "DataFormats/LaserAlignment/interface/LASAlignmentParameter.h"
#include <vector>
#include <map>

class LASAlignmentParameterCollection
{
 public:
  typedef std::vector<LASAlignmentParameter>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<std::string, IndexRange> Registry;
  typedef std::map<std::string, IndexRange>::const_iterator RegistryIterator;

  // typedef for map of Detector labels to their associated LASAlignmentParameter
  typedef std::map< std::string, std::vector<LASAlignmentParameter> > LASAlignmentParameterContainer;

	/// constructor
  LASAlignmentParameterCollection() {}
	/// put new LASAlignmentParameter into the collection
  void put(Range input, std::string name_);
	/// get Range
  const Range get(std::string name_) const;
	/// get vector with names
  const std::vector<std::string> names() const;

  /// appends LASAlignmentParameter to the vector of the given detector label
  void add(std::string & name_, std::vector<LASAlignmentParameter>& alignmentParameters);

  /// returns (by reference) the LASAlignmentParameter for a given detector label
  void alignmentParameter(std::string & name_, std::vector<LASAlignmentParameter>& alignmentParameters) const;

  /// returns (by reference) vector of detector labels with a LASAlignmentParameter
  void names(std::vector<std::string>& names_) const;

  /// return the number of entries in the collection
  int size() const;

 private:
  mutable std::vector<LASAlignmentParameter> container_;
  mutable Registry map_;

  // map of detector labels to their associated LASAlignmentParameter
  mutable LASAlignmentParameterContainer parameterMap_;

};
#endif
