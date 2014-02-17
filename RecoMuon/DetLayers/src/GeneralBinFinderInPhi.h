#ifndef GeneralBinFinderInPhi_H
#define GeneralBinFinderInPhi_H

/** \class GeneralBinFinderInPhi
 * A phi bin finder for a non-periodic group of detectors.
 *
 *  $Date: 2012/07/16 07:32:42 $
 *  $Revision: 1.6 $
 *  \author N. Amapane - INFN Torino
 */

#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "PhiBorderFinder.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

template <class T>
class GeneralBinFinderInPhi : public BaseBinFinder<T> {
public:

  typedef PhiBorderFinder::Det Det; //FIXME!!!

  GeneralBinFinderInPhi() : theNbins(0) {}

  /// Construct from an already initialized PhiBorderFinder
  GeneralBinFinderInPhi(const PhiBorderFinder& bf) {
    theBorders=bf.phiBorders();
    theBins=bf.phiBins();
    theNbins=theBins.size();
  }

  /// Construct from the list of Det*
  GeneralBinFinderInPhi(std::vector<Det*>::const_iterator first,
			std::vector<Det*>::const_iterator last)
    : theNbins( last-first)
  {
    std::vector<const Det*> dets(first,last);
    PhiBorderFinder bf(dets);
    theBorders=bf.phiBorders();
    theBins=bf.phiBins();
    theNbins=theBins.size();
  }

  virtual ~GeneralBinFinderInPhi(){};

  /// Returns an index in the valid range for the bin that contains 
  /// AND is closest to phi
  virtual int binIndex( T phi) const {
    
    const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|GeneralBinFinderInPhi";

    static T epsilon = 10*std::numeric_limits<T>::epsilon();
    // Assume -pi, pi range in pi (which is the case for Geom::Phi

    LogTrace(metname) << "GeneralBinFinderInPhi::binIndex,"
		      << " Nbins: "<< theNbins;

    for (int i = 0; i< theNbins; i++) {

      T cur = theBorders[i];
      T next = theBorders[binIndex(i+1)];
      T phi_ = phi;

      LogTrace(metname) << "bin: " << i 
			<< " border min " << cur << " border max: " << next << " phi: "<< phi_;

      if ( cur > next ) // we are crossing the pi edge: so move the edge to 0!
	{
	  cur = positiveRange(cur);
	  next = positiveRange(next);
	  phi_ = positiveRange(phi_); 
	}
      if (phi_ > cur-epsilon && phi_ < next) return i;
    }
    throw cms::Exception("UnexpectedState") << "GeneralBinFinderInPhi::binIndex( T phi) bin not found!";
  }
  
  /// Returns an index in the valid range, modulo Nbins
  virtual int binIndex( int i) const {
    int ind = i % (int)theNbins;
    return (ind < 0) ? ind+theNbins : ind;
  }

  /// the middle of the bin in radians
  virtual T binPosition( int ind) const {
    return theBins[binIndex(ind)];
  }


private:
  int theNbins;
  std::vector<T> theBorders;
  std::vector<T> theBins;

  // returns a positive angle; does NOT reduce the range to 2 pi
  inline T positiveRange (T phi) const
  {
    return (phi > 0) ? phi : phi + Geom::twoPi();
  }

};
#endif

