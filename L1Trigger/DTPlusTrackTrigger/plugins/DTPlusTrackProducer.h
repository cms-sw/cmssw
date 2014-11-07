#ifndef DTPlusTrackProducer_h
#define DTPlusTrackProducer_h

/*! \class DTPlusTrackProducer
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief EDProducer of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/interface/CachedProducts.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "L1Trigger/DTPlusTrackTrigger/interface/DTUtilities.h"

class DTPlusTrackProducer : public edm::EDProducer
{
  public :

    explicit DTPlusTrackProducer( const edm::ParameterSet& );
    ~DTPlusTrackProducer();

  private :

    virtual void beginJob();
    virtual void beginRun( const edm::Run&, const edm::EventSetup& );
    virtual void produce( edm::Event&, const edm::EventSetup& );
    virtual void endJob();

    /// Input tag
    edm::InputTag TTStubsInputTag;
    edm::InputTag TTTracksInputTag;

    /// bool flags
    bool useTSTheta;
    bool useRoughTheta;

    /// How many sigmas for the stub matching?
    double numSigmasStub;
    double numSigmasTk;
    double numSigmasPt;

    /// Min Pt of L1Tracks for matching
    double minL1TrackPt;

    /// Some constraints for finding the Pt with several methods
    double minRInvB;
    double maxRInvB;
    double station2Correction;
    bool thirdMethodAccurate;

    /// The L1 DT products
    BtiTrigsCollection*     outputBtiTrigs;
    TSPhiTrigsCollection*   outputTSPhiTrigs;
    TSThetaTrigsCollection* outputTSThetaTrigs;
    DTMatchesCollection*    outputDTMatches;

    /// The collection of matches
    std::map< unsigned int, std::vector< DTMatch* > >* tempDTMatchContainer;

  protected :

    DTTrig* theDTTrigger;
    const edm::ParameterSet pSetDT; /// needed by DTTrigger

};

#endif

