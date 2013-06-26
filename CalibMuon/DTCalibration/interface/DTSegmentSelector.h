#ifndef CalibMuon_DTCalibration_DTSegmentSelector_h
#define CalibMuon_DTCalibration_DTSegmentSelector_h

/*
 *  $Date: 2010/11/18 20:33:10 $
 *  $Revision: 1.4 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>

class DTRecSegment4D;
class DTRecHit1D;
class DTStatusFlag;

class DTSegmentSelector {
   public:
      DTSegmentSelector(edm::ParameterSet const& pset):
         checkNoisyChannels_(pset.getParameter<bool>("checkNoisyChannels")),
         minHitsPhi_(pset.getParameter<int>("minHitsPhi")),
         minHitsZ_(pset.getParameter<int>("minHitsZ")),
         maxChi2_(pset.getParameter<double>("maxChi2")),
         maxAnglePhi_(pset.getParameter<double>("maxAnglePhi")),
         maxAngleZ_(pset.getParameter<double>("maxAngleZ")) {
      }
      ~DTSegmentSelector() {}
      bool operator() (DTRecSegment4D const&, edm::Event const&, edm::EventSetup const&);
    
   private:
      bool checkNoisySegment(edm::ESHandle<DTStatusFlag> const&, std::vector<DTRecHit1D> const&);

      bool checkNoisyChannels_;
      int minHitsPhi_;
      int minHitsZ_;
      double maxChi2_;
      double maxAnglePhi_;
      double maxAngleZ_;
};

#endif
