#ifndef RBorderFinder_H
#define RBorderFinder_H

/** \class RBorderFinder
 *  Find the R binning of a list of detector according to several 
 *  definitions.
 *
 *  $Date: 2007/03/07 13:20:54 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - INFN Torino
 */

#include <DataFormats/GeometrySurface/interface/BoundingBox.h>
#include <DataFormats/GeometrySurface/interface/GeometricSorting.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <TrackingTools/DetLayers/interface/simple_stat.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <vector>
//#include <iostream>

class RBorderFinder {
public:
  
  typedef ForwardDetRing Det; //FIXME!!!
  typedef geomsort::ExtractR<Det,float> DetR;

  RBorderFinder(std::vector<const Det*> theDets) 
    : theNbins(theDets.size()), isRPeriodic_(false), isROverlapping_(false)
  {
    precomputed_value_sort(theDets.begin(), theDets.end(), DetR());

    std::vector<ConstReferenceCountingPointer<BoundDisk> > disks(theNbins);
    for ( int i = 0; i < theNbins; i++ ) {
      disks[i] = 
	dynamic_cast<const BoundDisk*> (&(theDets[i]->surface()));
      if (disks[i]==0) {
	throw cms::Exception("UnexpectedState") << "RBorderFinder: implemented for BoundDisks only";
      }
    }


    if (theNbins==1) { // Trivial case
      isRPeriodic_ = true; // meaningless in this case
      theRBorders.push_back(disks.front()->innerRadius());
      theRBins.push_back((disks.front()->outerRadius()+disks.front()->innerRadius()));
//       std::cout << "RBorderFinder:  theNbins " << theNbins << std::endl
// 		<< " C: " << theRBins[0]
// 		<< " Border: " << theRBorders[0] << std::endl;
    } else { // More than 1 bin
      double step = (disks.back()->innerRadius() -
		     disks.front()->innerRadius())/(theNbins-1);
      std::vector<double> spread;
      std::vector<std::pair<double,double> > REdge;
      REdge.reserve(theNbins);
      theRBorders.reserve(theNbins);
      theRBins.reserve(theNbins);
      spread.reserve(theNbins);
    
      for ( int i = 0; i < theNbins; i++ ) {
	theRBins.push_back((disks[i]->outerRadius()+disks[i]->innerRadius())/2.);
	spread.push_back(theRBins.back() - (theRBins[0] + i*step));
	REdge.push_back(std::pair<double,double>(disks[i]->innerRadius(),
						 disks[i]->outerRadius()));
      }
      
      theRBorders.push_back(REdge[0].first);
      for (int i = 1; i < theNbins; i++) {
	// Average borders of previous and next bins
	double br = (REdge[(i-1)].second + REdge[i].first)/2.;
	theRBorders.push_back(br);
      }
      
      for (int i = 1; i < theNbins; i++) {
	if (REdge[i].first - REdge[i-1].second < 0) {
	  isROverlapping_ = true;
	  break;
	}
      }
    
      double rms = stat_RMS(spread); 
      if ( rms < 0.01*step) { 
	isRPeriodic_ = true;
      }
    
//       std::cout << "RBorderFinder:  theNbins " << theNbins
// 		<< " step: " << step << " RMS " << rms << std::endl;
//       for (int idbg = 0; idbg < theNbins; idbg++) {
// 	std::cout << "L: " << REdge[idbg].first
// 		  << " C: " << theRBins[idbg]
// 		  << " R: " << REdge[idbg].second
// 		  << " Border: " << theRBorders[idbg]
// 		  << " SP: " << spread[idbg] << std::endl;
//       }
    }

    //Check that everything is proper
    if ((int)theRBorders.size() != theNbins || (int)theRBins.size() != theNbins) 
      throw cms::Exception("UnexpectedState") << "RBorderFinder consistency error";
}

  // Construct from a std::vector of Det*. 
  // Not tested, and do not work if the Dets are rings since 
  // position().perp() gives 0...
//   RBorderFinder(std::vector<Det*> theDets) 
//     : theNbins(theDets.size()), isRPeriodic_(false), isROverlapping_(false)
//   {
//     sort(theDets.begin(), theDets.end(), DetLessR());

//     double step = (theDets.back()->position().perp() -
// 		   theDets.front()->position().perp())/(theNbins-1);
//     std::vector<double> spread(theNbins);
//     std::vector<std::pair<double,double> > REdge;
//     REdge.reserve(theNbins);
//     theRBorders.reserve(theNbins);
//     theRBins.reserve(theNbins);
//     for ( int i = 0; i < theNbins; i++ ) {
//       theRBins.push_back(theDets[i]->position().perp());
//       spread.push_back(theDets[i]->position().perp()
// 	- (theDets[0]->position().perp() + i*step));

//       const BoundPlane * plane = 
// 	dynamic_cast<const BoundPlane*>(&theDets[i]->surface());
//       if (plane==0) {
// 	throw DetLogicError("RBorderFinder: det surface is not a BoundPlane");
//       }
      
//       std::vector<GlobalPoint> dc = 
// 	BoundingBox().corners(*plane);

//       float rmin(dc.front().perp());
//       float rmax(rmin); 
//       for (std::vector<GlobalPoint>::const_iterator pt=dc.begin();
// 	   pt!=dc.end(); pt++) {
// 	float r = (*pt).perp();	
// 	//	float z = pt->z();
// 	rmin = min( rmin, r);
// 	rmax = max( rmax, r);
//       }

//       // in addition to the corners we have to check the middle of the 
//       // det +/- length/2 since the min (max) radius for typical fw dets
//       // is reached there 
//       float rdet = theDets[i]->position().perp();
//       float len = theDets[i]->surface().bounds().length();
//       rmin = min( rmin, rdet-len/2.F);
// //   theRmax = max( theRmax, rdet+len/2.F);

//       REdge.push_back(make_std::pair(rmin,rmax));
//     }
//     // ...
//   }
  

  virtual ~RBorderFinder(){};

  /// Returns true if the Dets are periodic in R.
  inline bool isRPeriodic() const { return isRPeriodic_; }
  
  /// Returns true if any 2 of the Det overlap in R.
  inline bool isROverlapping() const { return isROverlapping_; }

  /// The borders, defined for each det as the middle between its lower 
  /// edge and the previous Det's upper edge.
  inline std::vector<double> RBorders() const { return theRBorders; }

  /// The centers of the Dets.
  inline std::vector<double> RBins() const { return theRBins; }

  //  inline std::vector<double> etaBorders() {}
  //  inline std::vector<double> zBorders() {}


private:
  int theNbins;
  bool isRPeriodic_;
  bool isROverlapping_;
  std::vector<double> theRBorders;
  std::vector<double> theRBins;

  inline int binIndex( int i) const {
    return std::min( std::max( i, 0), theNbins-1);
  }
};
#endif

