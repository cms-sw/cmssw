#ifndef EmulatedME0Segment_EmulatedME0SegmentProducer_h
#define EmulatedME0Segment_EmulatedME0SegmentProducer_h

/** \class EmulatedME0SegmentProducer 
 * Produces a collection of EmulatedME0Segment's in endcap muon ME0s. 
 *
 * $Date: 2010/03/11 23:48:11 $
 * \author David Nash
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include <DataFormats/MuonReco/interface/EmulatedME0Segment.h>



class FreeTrajectoryState;
class MagneticField;
class TRandom3;

class EmulatedME0SegmentProducer : public edm::EDProducer {
public:
    /// Constructor
    explicit EmulatedME0SegmentProducer(const edm::ParameterSet&);
    /// Destructor
    ~EmulatedME0SegmentProducer();
    /// Produce the EmulatedME0Segment collection
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

    void RotateCovMatrix(const AlgebraicMatrix&, const AlgebraicSymMatrix&,int, AlgebraicSymMatrix&);


private:
    TRandom3 * Rand;
    int iev; 
};

#endif
