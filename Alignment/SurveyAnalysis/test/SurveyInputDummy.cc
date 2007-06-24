#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "Alignment/SurveyAnalysis/test/SurveyInputDummy.h"

SurveyInputDummy::SurveyInputDummy(const edm::ParameterSet& cfg)
{
  typedef std::vector<edm::ParameterSet> ParameterSets;
 
  static AlignableObjectId idMap;

  const ParameterSets& errors = cfg.getParameter<ParameterSets>("errors");

  unsigned int nError = errors.size();

  for (unsigned int i = 0; i < nError; ++i)
  {
    const edm::ParameterSet& error = errors[i];

    theErrors[idMap.nameToType( error.getParameter<std::string>("level") )]
      = error.getParameter<double>("value");
  }
}

void SurveyInputDummy::beginJob(const edm::EventSetup& setup)
{
  edm::ESHandle<DDCompactView> view;
  edm::ESHandle<GeometricDet>  geom;

  setup.get<IdealGeometryRecord>().get(view);
  setup.get<IdealGeometryRecord>().get(geom);

  TrackerGeometry* tracker =
    TrackerGeomBuilderFromGeometricDet().build(&*view, &*geom);

  Alignable* ali = new AlignableTracker(&*geom, tracker);

  addSurveyInfo(ali);
  addComponent(ali);
}

void SurveyInputDummy::addSurveyInfo(Alignable* ali)
{
  const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo(comp[i]);

  align::ErrorMatrix cov; // default 0

  std::map<StructureType, double>::const_iterator e = theErrors.find( (StructureType)ali->alignableObjectId() );

  if (theErrors.end() != e)
  {
    cov = ROOT::Math::SMatrixIdentity();
    cov *= e->second * e->second;
  }

  ali->setSurvey( new SurveyDet(ali->surface(), cov) );
}
