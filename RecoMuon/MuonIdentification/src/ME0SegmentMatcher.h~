#ifndef ME0Segment_ME0SegmentMatcher_h
#define ME0Segment_ME0SegmentMatcher_h

/** \class ME0SegmentMatcher 
 * Produces a collection of ME0Segment's in endcap muon ME0s. 
 *
 * $Date: 2010/03/11 23:48:11 $
 *
 * \author David Nash
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

class FreeTrajectoryState;
class MagneticField;
class ME0SegmentMatcher : public edm::EDProducer {
public:
    /// Constructor
    explicit ME0SegmentMatcher(const edm::ParameterSet&);
    /// Destructor
    ~ME0SegmentMatcher();
    /// Produce the ME0Segment collection
    virtual void produce(edm::Event&, const edm::EventSetup&);

    FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
				   int , const AlgebraicSymMatrix66& ,
				   const MagneticField* );

    FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
				   int , const AlgebraicSymMatrix55& ,
				   const MagneticField* );

    void getFromFTS(const FreeTrajectoryState& ,
		  GlobalVector& , GlobalVector& , 
		  int& , AlgebraicSymMatrix66& );

private:

    int iev; // events through
    
};

#endif
