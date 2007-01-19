#ifndef PhiBorderFinder_H
#define PhiBorderFinder_H

/** \class PhiBorderFinder
 *  Find the phi binning of a list of detector according to several 
 *  definitions.
 *
 *  $Date: 2006/06/13 08:46:03 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - INFN Torino
 */


#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Vector/interface/Phi.h>
#include <Geometry/Surface/interface/BoundingBox.h>
#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/Surface/interface/GeometricSorting.h>
#include <TrackingTools/DetLayers/interface/simple_stat.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <vector>

class PhiBorderFinder {
public:
  
  typedef DetRod Det; //FIXME!!!
  typedef geomsort::ExtractPhi<Det,float> DetPhi;


  PhiBorderFinder(std::vector<const Det*> theDets) 
    : theNbins(theDets.size()), isPhiPeriodic_(false), isPhiOverlapping_(false) {
    precomputed_value_sort(theDets.begin(), theDets.end(), DetPhi());

    double step = 2.*Geom::pi()/theNbins;
    std::vector<double> spread(theNbins);
    std::vector<std::pair<double,double> > phiEdge;
    phiEdge.reserve(theNbins);
    thePhiBorders.reserve(theNbins);
    thePhiBins.reserve(theNbins);
    for ( unsigned int i = 0; i < theNbins; i++ ) {
      thePhiBins.push_back(theDets[i]->position().phi());
      spread.push_back(theDets[i]->position().phi()
	- (theDets[0]->position().phi() + i*step));

      ConstReferenceCountingPointer<BoundPlane> plane = 
	dynamic_cast<const BoundPlane*>(&theDets[i]->surface());
      if (plane==0) {
	//FIXME
	throw cms::Exception("UnexpectedState") << ("PhiBorderFinder: det surface is not a BoundPlane");
      }
      
      std::vector<GlobalPoint> dc = 
	BoundingBox().corners(*plane);

      float phimin(999.), phimax(-999.);
      for (std::vector<GlobalPoint>::const_iterator pt=dc.begin();
	   pt!=dc.end(); pt++) {
	float phi = (*pt).phi();
//	float z = pt->z();
	if (phi < phimin) phimin = phi;
	if (phi > phimax) phimax = phi;
      }
      if (phimin*phimax < 0. &&           //Handle pi border:
	  phimax - phimin > Geom::pi()) { //Assume that the Det is on
                                          //the shortest side 
	std::swap(phimin,phimax);
      }
      phiEdge.push_back(std::pair<double,double>(phimin,phimax));
      
    }
    
    for (unsigned int i = 0; i < theNbins; i++) {
      Geom::Phi<double> br(positiveRange
		      (positiveRange(phiEdge[binIndex(i-1)].second)
		       +positiveRange(phiEdge[i].first))/2.);
      thePhiBorders.push_back(br);
    }
  
    for (unsigned int i = 0; i < theNbins; i++) {
      if (Geom::Phi<double>(phiEdge[i].first)
	  - Geom::Phi<double>(phiEdge[binIndex(i-1)].second) < 0) {
	isPhiOverlapping_ = true;
	break;
      }
    }

    double rms = stat_RMS(spread); 
    if ( rms < 0.01*step) { 
      isPhiPeriodic_ = true;
    }

    //Check that everything is proper
    if (thePhiBorders.size() != theNbins || thePhiBins.size() != theNbins) 
      //FIXME
      throw cms::Exception("UnexpectedState") << "PhiBorderFinder: consistency error";
  }
  

  virtual ~PhiBorderFinder(){};

  inline unsigned int nBins() {return theNBins;}

  /// Returns true if the Dets are periodic in phi.
  inline bool isPhiPeriodic() const { return isPhiPeriodic_; }
  
  /// Returns true if any 2 of the Det overlap in phi.
  inline bool isPhiOverlapping() const { return isPhiOverlapping_; }

  /// The borders, defined for each det as the middle between its lower 
  /// edge and the previous Det's upper edge.
  inline std::vector<double>& phiBorders() const { return thePhiBorders; }

  /// The centers of the Dets.
  inline std::vector<double>& phiBins() const { return thePhiBins; }

  //  inline std::vector<double> etaBorders() {}
  //  inline std::vector<double> zBorders() {}

private:
  unsigned int theNbins;
  bool isPhiPeriodic_;
  bool isPhiOverlapping_;
  std::vector<double> thePhiBorders;
  std::vector<double> thePhiBins;

  inline double positiveRange (double phi) const
  {
    return (phi > 0) ? phi : phi + Geom::twoPi();
  }

  int binIndex( int i) const {
    int ind = i % theNbins;
    return (ind < 0) ? ind+theNbins : ind;
  }


};
#endif

