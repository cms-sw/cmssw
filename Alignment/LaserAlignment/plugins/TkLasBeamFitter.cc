/**\class TkLasBeamFitter TkLasBeamFitter.cc Alignment/LaserAlignment/plugins/TkLasBeamFitter.cc

  Original Authors:  Gero Flucke/Kolja Kaschube
           Created:  Wed May  6 08:43:02 CEST 2009
           $Id: TkLasBeamFitter.cc,v 1.1 2009/05/11 10:01:28 flucke Exp $

 Description: Fitting LAS beams with track model and providing TrajectoryStateOnSurface for hits.

 Implementation:
    - TkLasBeamCollection read from edm::Run
    - currently all done in beginRun(..),
      but should move to endRun(..) to allow a correct sequence with 
      production of TkLasBeamCollection in LaserAlignment::endRun(..)
*/


// system include files
#include <memory>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// data formats
// for edm::InRun
#include "DataFormats/Provenance/interface/BranchType.h"
// laser data formats
#include "DataFormats/LaserAlignment/interface/TkLasBeam.h"
#include "DataFormats/LaserAlignment/interface/TkFittedLasBeam.h"

// further includes
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"


//
// class declaration
//

class TkLasBeamFitter : public edm::EDProducer {
public:
  explicit TkLasBeamFitter(const edm::ParameterSet &config);
  ~TkLasBeamFitter();
  
  // virtual void beginJob() {}
  virtual void produce(edm::Event &event, const edm::EventSetup &setup);
  //  virtual void beginRun(edm::Run &run, const edm::EventSetup &setup);
  virtual void endRun(edm::Run &run, const edm::EventSetup &setup);
  // virtual void endJob() {}

private:
  /// Fit 'beam' using infor from its base class TkLasBeam and set its parameters.
  /// Also fill 'tsoses' with TSOS for each LAS hit. 
  bool fitBeam(TkFittedLasBeam &beam, std::vector<TrajectoryStateOnSurface> &tsoses);

  // ----------member data ---------------------------
  const edm::InputTag src_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
TkLasBeamFitter::TkLasBeamFitter(const edm::ParameterSet &iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src"))
{
  // declare the products to produce
  this->produces<TkFittedLasBeamCollection, edm::InRun>();
  this->produces<TsosVectorCollection, edm::InRun>();
  
  //now do what ever other initialization is needed
}

//---------------------------------------------------------------------------------------
TkLasBeamFitter::~TkLasBeamFitter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

//---------------------------------------------------------------------------------------
// ------------ method called to produce the data  ------------
void TkLasBeamFitter::produce(edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  // Nothing per event!
}

//---------------------------------------------------------------------------------------
// ------------ method called at end of each run  ---------------------------------------
void TkLasBeamFitter::endRun(edm::Run &run, const edm::EventSetup &setup)
{
  edm::Handle<TkLasBeamCollection> lasBeams;
  run.getByLabel(src_, lasBeams);

  // Create output collections - they are parallel.
  // (edm::Ref etc. and thus edm::AssociationVector are not supported for edm::Run...)
  std::auto_ptr<TkFittedLasBeamCollection> fittedBeams(new TkFittedLasBeamCollection);
  // One std::vector<TSOS> for each TkFittedLasBeam:
  std::auto_ptr<TsosVectorCollection> tsosesVec(new TsosVectorCollection);

  // Loop on input collection creating products (delegated to algorithm class?):
  for (TkLasBeamCollection::const_iterator iBeam = lasBeams->begin(), iEnd = lasBeams->end();
       iBeam != iEnd; ++iBeam) {
    fittedBeams->push_back(TkFittedLasBeam(*iBeam));
    tsosesVec->push_back(std::vector<TrajectoryStateOnSurface>());

    if (!this->fitBeam(fittedBeams->back(), tsosesVec->back())) {
      edm::LogError("BadFit") 
	<< "Problems fitting TkLasBeam, id " << fittedBeams->back().getBeamId() << ".";
      fittedBeams->pop_back(); // remove last entry added just before
      tsosesVec->pop_back();   // dito
    }
  }

  // Finally put fitted beams and TSOS vectors into run
  run.put(fittedBeams);
  run.put(tsosesVec);
}

//---------------------------------------------------------------------------------------
bool TkLasBeamFitter::fitBeam(TkFittedLasBeam &beam, std::vector<TrajectoryStateOnSurface> &tsoses)
{
  // - fit 'beam'
  // - set its parameters
  // - calculate TSOS for each hit, adding them in order of hits to 'tsoses'
  //
  // Dummy implementation so far...

  unsigned int paramType = 0;
  std::vector<TkFittedLasBeam::Scalar> params(3);      // two local, one global
  std::vector<TkFittedLasBeam::Scalar> derivatives(3); // dito
  unsigned int firstFixedParam = 2; // 3 parameters, but 0 and 1 local, while 2 is global/fixed
  AlgebraicSymMatrix parCov(2, 1); // 2x2 identity for 'free' part
  float chi2 = 10.;
  // Set fit results:
  beam.setParameters(paramType, params, parCov, derivatives, firstFixedParam, chi2);

  // Calculate TSOSes for each hit with the fitted parameters:
  tsoses.clear(); // just to be sure...
  for (std::vector<SiStripLaserRecHit2D>::const_iterator iHit = beam.begin(), iEnd = beam.end();
       iHit != iEnd; ++iHit) {
    TrajectoryStateOnSurface tsosOfHit; // = ... here calculate it...
    tsoses.push_back(tsosOfHit);
  }
  
  return true; // return false in case of problems
}

//---------------------------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(TkLasBeamFitter);
