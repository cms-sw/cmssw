#ifndef PhiBorderFinder_H
#define PhiBorderFinder_H

/** \class PhiBorderFinder
 *  Find the phi binning of a list of detector according to several 
 *  definitions.
 *
 *  $Date: 2007/01/19 11:57:44 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - INFN Torino
 */


#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Vector/interface/Phi.h>
#include <Geometry/Surface/interface/BoundingBox.h>
#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/Surface/interface/GeometricSorting.h>
#include <TrackingTools/DetLayers/interface/simple_stat.h>
#include <FWCore/Utilities/interface/Exception.h>

// FIXME: remove this include
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

class PhiBorderFinder {
public:
  
  typedef DetRod Det; //FIXME!!!
  typedef geomsort::ExtractPhi<Det,float> DetPhi;


  PhiBorderFinder(std::vector<const Det*> theDets) 
    : theNbins(theDets.size()), isPhiPeriodic_(false), isPhiOverlapping_(false) {
    precomputed_value_sort(theDets.begin(), theDets.end(), DetPhi());

    const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|PhiBorderFinder";

    double step = 2.*Geom::pi()/theNbins;

    LogTrace(metname) << "RecoMuonDetLayers::PhiBorderFinder "
		      << "step w: " << step << " # of bins: " << theNbins;
    
    std::vector<double> spread(theNbins);
    std::vector<std::pair<double,double> > phiEdge;
    phiEdge.reserve(theNbins);
    thePhiBorders.reserve(theNbins);
    thePhiBins.reserve(theNbins);
    for ( unsigned int i = 0; i < theNbins; i++ ) {
      thePhiBins.push_back(theDets[i]->position().phi());
      spread.push_back(theDets[i]->position().phi()
	- (theDets[0]->position().phi() + i*step));
      
      LogTrace(metname) << "bin: " << i << " phi bin: " << thePhiBins[i]
			<< " spread: " <<  spread[i];
      

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

	LogTrace(metname) << "Plane corner "<< *pt
			  << " phi: " << phi << " phi min: " << phimin << " phi max: " << phimax; 

      }
      if (phimin*phimax < 0. &&           //Handle pi border:
	  phimax - phimin > Geom::pi()) { //Assume that the Det is on
                                          //the shortest side 
	LogTrace(metname) << "Swapping...";

	std::swap(phimin,phimax);
      }
      phiEdge.push_back(std::pair<double,double>(phimin,phimax));

      LogTrace(metname) << "Final phi edges: " << phimin << " " << phimax;

    }
    
    LogTrace(metname) << "Creates the phi borders";
    for (unsigned int i = 0; i < theNbins; i++) {
      Geom::Phi<double> br(positiveRange
		      (positiveRange(phiEdge[binIndex(i-1)].second)
		       +positiveRange(phiEdge[i].first))/2.);
      //  thePhiBorders.push_back(br);

       LogTrace(metname) << "bin: " << i << " binIndex(i-1): " << binIndex(i-1)
			 << " phiEdge.second: " << phiEdge[binIndex(i-1)].second
			 << " pos range: " << positiveRange(phiEdge[binIndex(i-1)].second)
			 << "\n"
			 << "phiEdge.first: " << phiEdge[i].first
			 << " pos range: " << positiveRange(phiEdge[i].first)
			 << " sum: " << positiveRange(phiEdge[binIndex(i-1)].second) + positiveRange(phiEdge[i].first) 
			 << " final result: " << br;

       Geom::Phi<double> firstEdge(positiveRange(phiEdge[i].first));
       Geom::Phi<double> secondEdge(positiveRange(phiEdge[binIndex(i-1)].second));
       
       double firstEdge2 = firstEdge.value();
       double secondEdge2 = secondEdge.value();

       Geom::Phi<double> mean( (firstEdge + secondEdge)/2. );      
       Geom::Phi<double> mean2( (firstEdge2 + secondEdge2)/2. );
       
       //       if ( (phiEdge[i].first * phiEdge[binIndex(i-1)].second < 0)  &&
       //    ( (fabs(phiEdge[i].first) > Geom::pi() && fabs(phiEdge[binIndex(i-1)].second) < Geom::pi() ) ||
       //	      (fabs(phiEdge[i].first) < Geom::pi() && fabs(phiEdge[binIndex(i-1)].second) > Geom::pi() ) ))
       //	 mean2 = Geom::pi() - mean2;
       
       if ( phiEdge[i].first * phiEdge[binIndex(i-1)].second < 0 ){
	 double pos1 = positiveRange(phiEdge[i].first);
	 double pos2 = positiveRange(phiEdge[binIndex(i-1)].second);
	 if ( (pos1 > pos2 && (pos1-pos2) < Geom::pi()) ||
	      (pos2 > pos1 && (pos2-pos2) < Geom::pi()) )
	   mean2 = Geom::pi() - mean2;
	   // mean2 = Geom::Phi<double>( (pos1+pos2)/2 );
       } 


       thePhiBorders.push_back(mean2);
       
       LogTrace(metname) << "Alternative procedure: "
			 << " first edge: " << firstEdge << " second edge: " << secondEdge
			 << " mean (final result): " << mean;
       LogTrace(metname) << " first edge2: " << firstEdge2 << " second edge2: " << secondEdge2
			 << " mean2 (final result): " << mean2;
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

  inline unsigned int nBins() {return theNbins;}

  /// Returns true if the Dets are periodic in phi.
  inline bool isPhiPeriodic() const { return isPhiPeriodic_; }
  
  /// Returns true if any 2 of the Det overlap in phi.
  inline bool isPhiOverlapping() const { return isPhiOverlapping_; }

  /// The borders, defined for each det as the middle between its lower 
  /// edge and the previous Det's upper edge.
  inline const std::vector<double>& phiBorders() const { return thePhiBorders; }

  /// The centers of the Dets.
  inline const std::vector<double>& phiBins() const { return thePhiBins; }

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

