#include "GeneratorInterface/GenFilters/interface/EMEnrichingFilter.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CLHEP/Vector/LorentzVector.h"


using namespace edm;
using namespace std;


EMEnrichingFilter::EMEnrichingFilter(const edm::ParameterSet& iConfig) { 
  
  ParameterSet filterPSet=iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet");
  
  EMEAlgo_=new EMEnrichingFilterAlgo(filterPSet, consumesCollector());

}

EMEnrichingFilter::~EMEnrichingFilter() {
}


bool EMEnrichingFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  
  bool result=EMEAlgo_->filter(iEvent, iSetup);

  return result;

}


