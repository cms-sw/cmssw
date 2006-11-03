#ifndef ALIGNABLESELECTOR_H
#define ALIGNABLESELECTOR_H

/** \class AlignableSelector
 *  \author Gero Flucke (selection by strings taken from AlignableParameterBuilder)
 *
 *  Selecting Alignable's of the tracker by predefined strings with additional constraints on 
 *  eta, phi, r or z
 *
 *  $Date: 2005/07/26 10:13:49 $
 *  $Revision: 1.1 $
 *  (last update by $Author$)
 */

#include <vector>

class Alignable;
class AlignableTracker;
namespace edm {
  class ParameterSet;
}

class AlignableSelector {
 public:
  /// Constructor
  explicit AlignableSelector(AlignableTracker *aliTracker);

  /// Destructor
  virtual ~AlignableSelector();

  /// vector of alignables selected so far
  const std::vector<Alignable*>& selectedAlignables() const;
  /// remove all selected Alignables and geometrical restrictions
  void clear();
  /// set geometrical restrictions to be applied on all following selections
  /// (slices defined by vdouble 'etaRanges', 'phiRanges', 'zRanges' and 'rRanges',
  /// empty array means no restriction)
  void setGeometryCuts(const edm::ParameterSet &pSet);
  /// add Alignables corresponding to predefined name, taking into account geometrical restrictions,
  /// returns number of added alignables
  unsigned int addSelection(const std::string &name);
  /// as addSelection with one argument, but overwriting geometrical restrictions
  unsigned int addSelection(const std::string &name, const edm::ParameterSet &pSet);

  /// true if geometrical restrictions in eta, phi, r, z not satisfied
  bool outsideRanges(const Alignable *alignable) const;
  /// true if ranges.size() is even and ranges[i] <= value < ranges[i+1] for any even i
  /// ( => false if ranges.empty() == true), if(isPhi==true) takes into account phi periodicity
  bool insideRanges(double value, const std::vector<double> &ranges, bool isPhi = false) const;

 protected:
  /// adding alignables which fulfil geometrical restrictions and special switches 
  unsigned int add(const std::vector<Alignable*> &alignables);
  /// some helper methods
  unsigned int addAllDets();
  unsigned int addAllRods();
  unsigned int addAllLayers();
  unsigned int addAllAlignables();

 private:
  AlignableTracker        *theTracker;
  std::vector<Alignable*>  theSelectedAlignables;

  /// geometrical restrictions in eta, phi, r, z to be applied for next addSelection
  std::vector<double> theRangesEta;
  std::vector<double> theRangesPhi;
  std::vector<double> theRangesR;
  std::vector<double> theRangesZ;

  // further switches used in add(...)
  bool theOnlyDS;
  bool theOnlySS;
  bool theSelLayers;
  int  theMinLayer;
  int  theMaxLayer;

};
#endif
