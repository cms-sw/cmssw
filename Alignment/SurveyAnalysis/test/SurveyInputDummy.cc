#include "TRandom3.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  Alignable* ali = new AlignableTracker;

  addSurveyInfo(ali);
  addComponent(ali);
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
    double error =  e->second;

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

    cov = ROOT::Math::SMatrixIdentity();
    cov *= error * error;
  }

  ali->setSurvey( new SurveyDet(ali->surface(), cov) );
}
