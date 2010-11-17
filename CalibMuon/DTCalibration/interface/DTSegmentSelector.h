#ifndef CalibMuon_DTCalibration_DTSegmentSelector_h
#define CalibMuon_DTCalibration_DTSegmentSelector_h

/*
 *  $Date: 2010/11/16 19:07:34 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

class DTRecSegment2D;
class DTRecSegment4D;
class DTStatusFlag;

class DTSegmentSelector {
   public:
      DTSegmentSelector(edm::ParameterSet const& pset):
         checkNoisyChannels_(pset.getParameter<bool>("checkNoisyChannels")),
         maxChi2_(pset.getParameter<double>("maxChi2")),
         maxAnglePhi_(pset.getParameter<double>("maxAnglePhi")),
         maxAngleZ_(pset.getParameter<double>("maxAngleZ")) {
      }
      ~DTSegmentSelector() {}
      bool operator() (edm::Event const&, edm::EventSetup const&, DTRecSegment4D const&);
    
   private:
      bool checkNoisySegment(edm::ESHandle<DTStatusFlag> const&, DTRecSegment2D const&);

      bool checkNoisyChannels_;
      double maxChi2_;
      double maxAnglePhi_;
      double maxAngleZ_;
};

#endif
