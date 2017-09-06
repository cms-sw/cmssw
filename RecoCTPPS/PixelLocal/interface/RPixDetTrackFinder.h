/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoCTPPS_PixelLocal_RPixDetTrackFinder_H
#define RecoCTPPS_PixelLocal_RPixDetTrackFinder_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "RecoCTPPS/PixelLocal/interface/RPixDetPatternFinder.h"


#include "CLHEP/Vector/ThreeVector.h"

#include <vector>
#include <map>

class RPixDetTrackFinder{

	public:
		RPixDetTrackFinder(edm::ParameterSet const& parameterSet): parameterSet_(parameterSet), romanPotId_(CTPPSPixelDetId(0, 2, 3, 0)) {}
		//romanPotId_ is needed to be defined in order to 

		virtual ~RPixDetTrackFinder();

		void setHits(const std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane> > hitMap) {hitMap_ = hitMap; }
  		virtual void findTracks()=0;
  		virtual void initialize()=0;
		void clear(){
			hitMap_.clear();
			localTrackVector_.clear();
		}
		std::vector<CTPPSPixelLocalTrack> getLocalTracks() {return localTrackVector_; }
  		void setRomanPotId(CTPPSPixelDetId rpId) {romanPotId_ = rpId;};
		void setPlaneRotationMatrices(std::map<CTPPSPixelDetId, TMatrixD> planeRotationMatrixMap) { planeRotationMatrixMap_ = planeRotationMatrixMap; }
		void setPointOnPlanes(std::map<CTPPSPixelDetId, CLHEP::Hep3Vector> planePointMap) { planePointMap_ = planePointMap; }
  		void setListOfPlanes(std::vector<uint32_t> listOfAllPlanes) { listOfAllPlanes_ = listOfAllPlanes; } 
  		void setZ0(double z0) { z0_ = z0; }


	protected:
		edm::ParameterSet parameterSet_;
		std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane> > hitMap_;
		std::vector<CTPPSPixelLocalTrack>  localTrackVector_;
		CTPPSPixelDetId  romanPotId_;
		std::map<CTPPSPixelDetId, TMatrixD> planeRotationMatrixMap_;
  		std::map<CTPPSPixelDetId, CLHEP::Hep3Vector> planePointMap_;
 		uint32_t numberOfPlanesPerPot_;
  		std::vector<uint32_t> listOfAllPlanes_;
  		double z0_;

};

#endif
