#ifndef ALIGNMENTPARAMETERSELECTOR_H
#define ALIGNMENTPARAMETERSELECTOR_H

/** \class AlignmentParameterSelector
 *  \author Gero Flucke (selection by strings taken from AlignableParameterBuilder)
 *
 *  Selecting Alignable's of the tracker by predefined strings,
 *  additional constraints on eta, phi, r, x, y or z are possible.
 *  Furthermore stores the 'selection' of selected AlignmentParameters.
 *
 *  $Date: 2013/01/07 20:56:25 $
 *  $Revision: 1.11 $
 *  (last update by $Author: wmtan $)
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"

class AlignableExtras;
class AlignableTracker;
class AlignableMuon;

namespace edm {
  class ParameterSet;
}

class AlignmentParameterSelector {
 public:
  /// Constructor from tracker only or from tracker and muon
  explicit AlignmentParameterSelector(AlignableTracker *aliTracker, AlignableMuon *aliMuon = 0,
				      AlignableExtras *aliExtras = 0);

  /// Destructor
  virtual ~AlignmentParameterSelector() {}

  /// vector of alignables selected so far
  const align::Alignables& selectedAlignables() const { return theSelectedAlignables; }
  /// vector of selection 'strings' for alignables, parallel to selectedAlignables()
  const std::vector<std::vector<char> >& selectedParameters() const { return theSelectedParameters; }
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
  /// true if layer is deselected via "Layers<N><M>" or "DS/SS"
  bool layerDeselected(const Alignable *alignable) const;
  /// true if alignable is DetUnit deselected by Unit<Rphi/Stereo> selection 
  bool detUnitDeselected(const Alignable *alignable) const;
  /// true if geometrical restrictions in eta, phi, r, x, y, z not satisfied
  bool outsideGeometricalRanges(const Alignable *alignable) const;
  /// true if DetId restrictions are not satisfied
  bool outsideDetIdRanges(const Alignable *alignable) const;
  /// true if ranges.size() is even and ranges[i] <= value < ranges[i+1] for any even i
  /// ( => false if ranges.empty() == true), if(isPhi==true) takes into account phi periodicity
  /// for the integer specialized method, true is returned for ranges[i] <= value <= ranges[i+1]
  /// and isPhi is ignored
  template<typename T> bool insideRanges(T value, const std::vector<T> &ranges,
					 bool isPhi = false) const;
  /// true if value is member of vector of values
  bool isMemberOfVector(int value, const std::vector<int> &values) const;
  /// Decomposing input string 's' into parts separated by 'delimiter'
  std::vector<std::string> decompose(const std::string &s, std::string::value_type delimiter) const;
  /// Converting std::string into std::vector<char>
  std::vector<char> convertParamSel(const std::string &selString) const;

 protected:
  /// adding alignables which fulfil geometrical restrictions and special switches 
  unsigned int add(const align::Alignables &alignables, const std::vector<char> &paramSel);
  /// some helper methods
  unsigned int addAllDets(const std::vector<char> &paramSel);
  unsigned int addAllRods(const std::vector<char> &paramSel);
  unsigned int addAllLayers(const std::vector<char> &paramSel);
  unsigned int addAllAlignables(const std::vector<char> &paramSel);

  void setPXBDetIdCuts(const edm::ParameterSet &pSet);
  void setPXFDetIdCuts(const edm::ParameterSet &pSet);
  void setTIBDetIdCuts(const edm::ParameterSet &pSet);
  void setTIDDetIdCuts(const edm::ParameterSet &pSet);
  void setTOBDetIdCuts(const edm::ParameterSet &pSet);
  void setTECDetIdCuts(const edm::ParameterSet &pSet);

  const AlignableTracker* alignableTracker() const;
  
 private:
  AlignableTracker* theTracker;
  AlignableMuon*    theMuon;
  AlignableExtras*  theExtras;
  align::Alignables theSelectedAlignables;
  std::vector<std::vector<char> > theSelectedParameters;

  /// geometrical restrictions in eta, phi, r, x, y, z to be applied for next addSelection
  std::vector<double> theRangesEta;
  std::vector<double> theRangesPhi;
  std::vector<double> theRangesR;
  std::vector<double> theRangesX;
  std::vector<double> theRangesY;
  std::vector<double> theRangesZ;

  /// DetId restrictions in eta, phi, r, x, y, z to be applied for next addSelection
  std::vector<int>    theDetIds;
  std::vector<int>    theDetIdRanges;
  std::vector<int>    theExcludedDetIds;
  std::vector<int>    theExcludedDetIdRanges;
  struct PXBDetIdRanges {
    std::vector<int>    theLadderRanges;
    std::vector<int>    theLayerRanges;
    std::vector<int>    theModuleRanges;
    void clear() {
      theLadderRanges.clear(); theLayerRanges.clear();
      theModuleRanges.clear();
    }
  }                   thePXBDetIdRanges;
  struct PXFDetIdRanges {
    std::vector<int>    theBladeRanges;
    std::vector<int>    theDiskRanges;
    std::vector<int>    theModuleRanges;
    std::vector<int>    thePanelRanges;
    std::vector<int>    theSideRanges;
    void clear() { 
      theBladeRanges.clear(); theDiskRanges.clear();
      theModuleRanges.clear(); thePanelRanges.clear();
      theSideRanges.clear();
    }
  }                   thePXFDetIdRanges;
  struct TIBDetIdRanges {
    std::vector<int>    theLayerRanges;
    std::vector<int>    theModuleRanges;
    std::vector<int>    theStringRanges;
    std::vector<int>    theSideRanges;
    void clear() { 
      theLayerRanges.clear(); theModuleRanges.clear();
      theSideRanges.clear(); theStringRanges.clear();
    }
  }                   theTIBDetIdRanges;
  struct TIDDetIdRanges {
    std::vector<int>    theDiskRanges; 
    std::vector<int>    theModuleRanges;
    std::vector<int>    theRingRanges;
    std::vector<int>    theSideRanges;  
    void clear() { 
      theDiskRanges.clear(); theModuleRanges.clear();
      theRingRanges.clear(); theSideRanges.clear();
    }
  }                   theTIDDetIdRanges;
  struct TOBDetIdRanges {
    std::vector<int>    theLayerRanges; 
    std::vector<int>    theModuleRanges;
    std::vector<int>    theRodRanges;  
    std::vector<int>    theSideRanges;
    void clear() { 
      theLayerRanges.clear(); theModuleRanges.clear();
      theRodRanges.clear(); theSideRanges.clear();
    }
  }                   theTOBDetIdRanges;  
  struct TECDetIdRanges {
    std::vector<int>    theWheelRanges; 
    std::vector<int>    thePetalRanges; 
    std::vector<int>    theModuleRanges;
    std::vector<int>    theRingRanges;
    std::vector<int>    theSideRanges;  
    void clear() { 
      theWheelRanges.clear(); thePetalRanges.clear();
      theModuleRanges.clear(); theRingRanges.clear();
      theSideRanges.clear();
    }
  }                   theTECDetIdRanges;
    
  // further switches used in add(...)
  bool theOnlyDS;
  bool theOnlySS;
  bool theSelLayers;
  int  theMinLayer;
  int  theMaxLayer;
  enum RphiOrStereoDetUnit { Stereo, Both, Rphi};
  RphiOrStereoDetUnit theRphiOrStereoDetUnit;
  /// Setting the special switches and returning input string, but after removing the 'special
  /// indicators' from it. Known specials are:
  /// "SS" anywhere in name: in TIB/TOB restrict to single sided Dets/Rods/Layers
  /// "DS" anywhere in name: in TIB/TOB restrict to double sided Dets/Rods/Layers
  /// "Layers14" at end of name: in tracker restrict to layers/disks 1 to 4, similar for other digits
  /// "UnitStereo" and "UnitRphi" anywhere in name:
  ///      for a DetUnit in strip select only if stereo or rphi module (keep 'Unit' in name!)
  std::string setSpecials(const std::string &name);

};

template<> bool
AlignmentParameterSelector::insideRanges<int>(int value, const std::vector<int> &ranges,
					      bool isPhi) const;

#endif
