
// A set of binfiders adapted from CommonReco/DetLayers.
// FIXME: This file should eventually disappear and binfinders in 
// CommonReco/DetLayers  modified to be general enough!!!

namespace MagBinFinders {
  template <class T> class GeneralBinFinderInR;
  template <class T> class GeneralBinFinderInZ;
}


//----------------------------------------------------------------------
#ifndef GeneralBinFinderInR_H
#define GeneralBinFinderInR_H
/** \class MagBinFinders::GeneralBinFinderInR
 *  A R binfinder for a non-periodic group of detectors.
 *
 *  $Date: 2008/11/14 10:57:30 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - INFN Torino
 */

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
//#include "CommonReco/DetLayers/interface/RBorderFinder.h"
#include <cmath>
#include <vector>

template <class T>
class MagBinFinders::GeneralBinFinderInR : public BaseBinFinder<T>{
public:
  
  GeneralBinFinderInR() : theNbins(0) {}

  GeneralBinFinderInR(std::vector<T>& borders) :
    theNbins(borders.size()), 
    theBorders(borders) {
    // FIXME: compute bin positions.
//     for (vector<T>::const_iterator i=theBorders.begin();
// 	 i<theBorders.end(); ++i) {
//      theBorders.push_back(((*i) + (*(i+1))) / 2.);

//     cout << "GeneralBinFinderInR_ " << theNbins << " " << theBorders.size() << " " << (int) this << endl;
}
  

//   /// Construct from an already initialized RBorderFinder
//   GeneralBinFinderInR(const RBorderFinder& bf) {
//     theBorders=bf.RBorders();
//     theBins=bf.RBins();
//     theNbins=theBins.size();
//   }

//  /// Construct from the list of Det*
//   GeneralBinFinderInR(vector<Det*>::const_iterator first,
// 		      vector<Det*>::const_iterator last)
//     : theNbins( last-first)
//   {
//     vector<Det*> dets(first,last);
//     RBorderFinder bf(dets);
//     theBorders=bf.phiBorders();
//     theBins=bf.phiBins();
//     theNbins=theBins.size();
//   }

  
  /// Returns an index in the valid range for the bin that contains
  /// AND is closest to R
  virtual int binIndex( T R) const {
    int i;
    for (i = 0; i<theNbins; ++i) {
      if (R < theBorders[i]){ // FIXME: one can be skipped?
	 break;
      }
    }
    return binIndex(i-1);
  }

  /// Returns an index in the valid range
  virtual int binIndex( int i) const {
    return std::min( std::max( i, 0), theNbins-1);
  }
   
  /// The middle of the bin
  virtual T binPosition( int ind) const {
    return theBins[binIndex(ind)];
  }


private:
  int theNbins;
  std::vector<T> theBorders;
  std::vector<T> theBins;

};
#endif
//----------------------------------------------------------------------



//----------------------------------------------------------------------
#ifndef GeneralBinFinderInZ_H
#define GeneralBinFinderInZ_H

/** \class MagBinFinders::GeneralBinFinderInZ
 * A Z bin finder for a non-periodic group of detectors.
 * Search is done starting from an initial equal-size-bin guess. Therefore
 * it becomes less efficient for binnings with very different bin size. 
 * It is not particularily suited for very few bins...
 */

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include <cmath>

template <class T>
class MagBinFinders::GeneralBinFinderInZ : public BaseBinFinder<T> {
public:

  GeneralBinFinderInZ() : theNbins(0), theZStep(0), theZOffset(0) {}

  GeneralBinFinderInZ(std::vector<T>& borders) :
    theNbins(borders.size()), 
    theBorders(++(borders.begin()),borders.end()) { // Skip border of 1. bin
    // FIXME:  crashes for borders.size()==1 (trivial case)
    // FIXME set theBins!!!
    theZOffset = theBorders.front(); 
    if (theNbins>2) {
      theZStep = (theBorders.back() - theBorders.front()) / (theNbins-2);
    } else {
      theZStep = 1.; // Not relevant in this case...
    }
    
//     cout << "GeneralBinFinderInZ " << theBorders.size()
// 	 << " " << theZStep << endl;
  }
  
  // FIXME: ??? theNbins e theBorders hanno size theNbins -1.
  // theBorders[1] e' l'inizio del secondo bin !!!

//   GeneralBinFinderInZ(vector<Det*>::const_iterator first,
// 		      vector<Det*>::const_iterator last) :
//     theNbins( last-first)
//   {
//     theBins.reserve(theNbins);
//     for (vector<Det*>::const_iterator i=first; i<last-1; ++i) {
//       theBins.push_back((**i).position().z());
//       theBorders.push_back(((**i).position().z() + 
// 			    (**(i+1)).position().z()) / 2.);
//     }

//     theZOffset = theBorders.front(); 
//     theZStep = (theBorders.back() - theBorders.front()) / (theNbins-2);
//   }

  /// returns an index in the valid range for the bin closest to Z
  virtual int binIndex(T z) const {
    int bin = binIndex(int((z-theZOffset)/theZStep)+1);
    
    // check left border
    if (bin > 0) {
      if (z < theBorders[bin-1]) {
	// z is to the left of the left border, the correct bin is left
	for (int i=bin-1; ; --i) {
	  if (i <= 0) return 0;  
	  if ( z > theBorders[i-1]) return i;
	}
      }
    } 
    else return 0;

    // check right border
    if (bin < theNbins-1) {
      if ( z > theBorders[bin]) {
	// z is to the right of the right border, the correct bin is right
	for (int i=bin+1; ; ++i) {
	  if (i >= theNbins-1) return theNbins-1;  
	  if ( z < theBorders[i]) return i;
	}
      }
    }
    else return theNbins-1;

    // if we arrive here it means that the bin is ok 
    return bin;
  }

  /// returns an index in the valid range
  virtual int binIndex( int i) const {
    return std::min( std::max( i, 0), theNbins-1);
  }
   
  /// the middle of the bin.
  virtual T binPosition( int ind) const {
    return theBins[binIndex(ind)];
  }

  static double pi() { return 3.141592653589793238;}
  static double twoPi() { return 2.*pi();}

private:

  int theNbins;
  T theZStep;
  T theZOffset;
  std::vector<T> theBorders;
  std::vector<T> theBins;
};
#endif
//----------------------------------------------------------------------
