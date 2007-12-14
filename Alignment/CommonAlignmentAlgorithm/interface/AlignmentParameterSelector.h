#ifndef ALIGNMENTPARAMETERSELECTOR_H
#define ALIGNMENTPARAMETERSELECTOR_H

/** \class AlignmentParameterSelector
 *  \author Gero Flucke (selection by strings taken from AlignableParameterBuilder)
 *
 *  Selecting Alignable's of the tracker by predefined strings,
 *  additional constraints on eta, phi, r, x, y or z are possible.
 *  Furthermore stores the 'selection' of selected AlignmentParameters.
 *
 *  $Date: 2007/01/23 16:07:08 $
 *  $Revision: 1.4 $
 *  (last update by $Author: fronga $)
 */

#include <vector>
#include <string>

class Alignable;
class AlignableTracker;
class AlignableMuon;
namespace edm {
  class ParameterSet;
}

class AlignmentParameterSelector {
 public:
  /// Constructor from tracker only
  explicit AlignmentParameterSelector(AlignableTracker *aliTracker);

  /// Constructor from tracker and muon
  AlignmentParameterSelector( AlignableTracker *aliTracker, AlignableMuon *aliMuon );

  /// Destructor
  virtual ~AlignmentParameterSelector();

  /// vector of alignables selected so far
  const std::vector<Alignable*>& selectedAlignables() const;
  /// vector of selection 'strings' for alignables, parallel to selectedAlignables()
  const std::vector<std::vector<char> >& selectedParameters() const;
  /// remove all selected Alignables and geometrical restrictions
  void clear();
  /// remove all geometrical restrictions
  void clearGeometryCuts();

  /// Add several selections defined by the PSet which must contain a vstring like e.g.
  /// vstring alignParams = { "PixelHalfBarrelLadders,111000,pixelSelection",
  ///                         "BarrelDSRods,111ff0",
  ///                         "BarrelSSRods,101ff0"}
  /// The e.g. '111ff0' is decoded into vector<char> and stored.
  /// Returns number of added selections or -1 if problems (then also an error is logged)
  /// If a string contains a third, comma separated part (e.g. ',pixelSelection'),
  /// a further PSet of that name is expected to select eta/phi/r/x/y/z-ranges.
  unsigned int addSelections(const edm::ParameterSet &pSet);
  /// set geometrical restrictions to be applied on all following selections
  /// (slices defined by vdouble 'etaRanges', 'phiRanges', 'xRanges', 'yRanges', 'zRanges'
  /// and 'rRanges', empty array means no restriction)
  void setGeometryCuts(const edm::ParameterSet &pSet);
  /// add Alignables corresponding to predefined name, taking into account geometrical restrictions
  /// as defined in setSpecials, returns number of added alignables
  unsigned int addSelection(const std::string &name, const std::vector<char> &paramSel);
  /// as addSelection with one argument, but overwriting geometrical restrictions
  unsigned int addSelection(const std::string &name, const std::vector<char> &paramSel, 
			    const edm::ParameterSet &pSet);

  /// true if geometrical restrictions in eta, phi, r, x, y, z not satisfied
  bool outsideRanges(const Alignable *alignable) const;
  /// true if ranges.size() is even and ranges[i] <= value < ranges[i+1] for any even i
  /// ( => false if ranges.empty() == true), if(isPhi==true) takes into account phi periodicity
  bool insideRanges(double value, const std::vector<double> &ranges, bool isPhi = false) const;
  /// Decomposing input string 's' into parts separated by 'delimiter'
  std::vector<std::string> decompose(const std::string &s, std::string::value_type delimiter) const;
  /// Converting std::string into std::vector<char>
  std::vector<char> convertParamSel(const std::string &selString) const;

 protected:
  /// adding alignables which fulfil geometrical restrictions and special switches 
  unsigned int add(const std::vector<Alignable*> &alignables, const std::vector<char> &paramSel);
  /// some helper methods
  unsigned int addAllDets(const std::vector<char> &paramSel);
  unsigned int addAllRods(const std::vector<char> &paramSel);
  unsigned int addAllLayers(const std::vector<char> &paramSel);
  unsigned int addAllAlignables(const std::vector<char> &paramSel);

 private:
  AlignableTracker*        theTracker;
  AlignableMuon*           theMuon;
  std::vector<Alignable*>  theSelectedAlignables;
  std::vector<std::vector<char> > theSelectedParameters;

  /// geometrical restrictions in eta, phi, r, x, y, z to be applied for next addSelection
  std::vector<double> theRangesEta;
  std::vector<double> theRangesPhi;
  std::vector<double> theRangesR;
  std::vector<double> theRangesX;
  std::vector<double> theRangesY;
  std::vector<double> theRangesZ;

  // further switches used in add(...)
  bool theOnlyDS;
  bool theOnlySS;
  bool theSelLayers;
  int  theMinLayer;
  int  theMaxLayer;
  /// Setting the special switches and returning input string, but after removing the 'special
  /// indicators' from it. Known specials are:
  /// "SS" anywhere in name: in TIB/TOB restrict to single sided Dets/Rods/Layers
  /// "DS" anywhere in name: in TIB/TOB restrict to double sided Dets/Rods/Layers
  /// "Layers14" at end of name: in TIB/TOB restrict to layers 1 to 4, similar for other digits
  std::string setSpecials(const std::string &name);

};
#endif
