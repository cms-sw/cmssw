#ifndef ME0Segment_ME0SegmentMatcher_h
#define ME0Segment_ME0SegmentMatcher_h

/** \class ME0SegmentMatcher 
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
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

//For Debugging

#include "TH1.h" 
#include <TH2.h>
#include "TFile.h"
#include <TProfile.h>
#include "TStyle.h"
#include <TCanvas.h>


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

    
    virtual void beginRun(edm::Run const&, edm::EventSetup const&);



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

    edm::ESHandle<ME0Geometry> me0Geom;

  
};

#endif
