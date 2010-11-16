#ifndef CalibMuon_DTCalibration_DTSegmentSelector_h
#define CalibMuon_DTCalibration_DTSegmentSelector_h

/*
 *  $Date: 2010/06/28 09:48:01 $
 *  $Revision: 1.6 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

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
      template <class T>
      bool checkNoisySegment(edm::ESHandle<DTStatusFlag> const&, T const&);

      bool checkNoisyChannels_;
      double maxChi2_;
      double maxAnglePhi_;
      double maxAngleZ_;
};

#endif
