#include "TRandom3.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/SurveyAnalysis/test/SurveyInputDummy.h"

SurveyInputDummy::SurveyInputDummy(const edm::ParameterSet& cfg):
  theRandomizeValue( cfg.getParameter<bool>("randomizeValue") )
{
  typedef std::vector<edm::ParameterSet> ParameterSets;
 
  const ParameterSets& errors = cfg.getParameter<ParameterSets>("errors");

  unsigned int nError = errors.size();

  for (unsigned int i = 0; i < nError; ++i)
  {
    const edm::ParameterSet& error = errors[i];

    theErrors[AlignableObjectId::stringToId( error.getParameter<std::string>("level") )]
      = error.getParameter<double>("value");
  }
}

void SurveyInputDummy::analyze(const edm::Event&, const edm::EventSetup& setup)
{
  if (theFirstEvent) {
    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    setup.get<IdealGeometryRecord>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    edm::ESHandle<TrackerGeometry> tracker;
    setup.get<TrackerDigiGeometryRecord>().get( tracker );
    
    Alignable* ali = new AlignableTracker( &*tracker, tTopo );
    
    addSurveyInfo(ali);
    addComponent(ali);

    theFirstEvent = false;
  }
}

void SurveyInputDummy::addSurveyInfo(Alignable* ali)
{
  static TRandom3 rand;

  const align::Alignables& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo(comp[i]);

  align::ErrorMatrix cov; // default 0

  std::map<align::StructureType, double>::const_iterator e = theErrors.find( ali->alignableObjectId() );

  if (theErrors.end() != e)
  {
    double error = e->second;

    if (theRandomizeValue)
    {
      double x = rand.Gaus(0., error);
      double y = rand.Gaus(0., error);
      double z = rand.Gaus(0., error);
      double a = rand.Gaus(0., error);
      double b = rand.Gaus(0., error);
      double g = rand.Gaus(0., error);

      align::EulerAngles angles(3);

      angles(1) = a; angles(2) = b; angles(3) = g;

      ali->move( ali->surface().toGlobal( align::LocalVector(x, y, z) ) );
      ali->rotateInLocalFrame( align::toMatrix(angles) );
    }

    cov = ROOT::Math::SMatrixIdentity();
    cov *= error * error;
  }

  ali->setSurvey( new SurveyDet(ali->surface(), cov) );
}
