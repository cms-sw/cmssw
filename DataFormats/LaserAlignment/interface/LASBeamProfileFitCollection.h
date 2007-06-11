#ifndef DataFormats_LaserAlignment_LASBeamProfileFitCollection_h
#define DataFormats_LaserAlignment_LASBeamProfileFitCollection_h

/** \class LASBeamProfileFitCollection
 *  Collection of LASBeamProfileFits
 *
 *  $Date: 2007/03/20 16:53:23 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "DataFormats/LaserAlignment/interface/LASBeamProfileFit.h"
#include <vector>
#include <map>

class LASBeamProfileFitCollection
{
 public:
  typedef std::vector<LASBeamProfileFit>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

  // typedef for map of DetIds to their associated LASBeamProfileFit
  typedef std::map< unsigned int, std::vector<LASBeamProfileFit> > LASBeamProfileFitContainer;

	/// constructor
  LASBeamProfileFitCollection() {}
	/// put new LASBeamProfileFit into the collection
  void put(Range input, unsigned int detID);
	/// get Range
  const Range get(unsigned int detID) const;
	/// get vector with detIDs
  const std::vector<unsigned int> detIDs() const;

  /// appends LASBeamProfileFit to the vector of the given DetId
  void add(unsigned int& det_id, std::vector<LASBeamProfileFit>& beamProfileFit);

  /// returns (by reference) the LASBeamProfileFit for a given DetId
  void beamProfileFit(unsigned int& det_id, std::vector<LASBeamProfileFit>& beamProfileFit) const;

  /// returns (by reference) vector of DetIds with a LASBeamProfileFit
  void detIDs(std::vector<unsigned int>& det_ids) const;

  /// return the number of entries in the collection
  int size() const;

 private:
  mutable std::vector<LASBeamProfileFit> container_;
  mutable Registry map_;

  // map of DetIds to their associated LASBeamProfileFit
  mutable LASBeamProfileFitContainer fitMap_;

};

#endif
