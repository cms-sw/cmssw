/** \class SurveyInputDummy
 *
 *  For uploading some random survey values and pseudo-dummy errors to DB.
 *
 *  If randomizeValue is true, the survey value of a structure in a level
 *  is randomly selected from a Gaussian distribution of mean given by the
 *  ideal geometry and width = "value" (e.g. width = 5e-4 for a Panel).
 *  
 *  The covariance matrix for all structures of a level will be diagonal
 *  given by value^2 * identity.
 *
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TRandom3.h"

#include <map>

class SurveyInputDummy : public SurveyInputBase {
public:
  SurveyInputDummy(const edm::ParameterSet&);

  /// Read ideal tracker geometry from DB
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  /// Add survey info to an alignable
  void addSurveyInfo(Alignable*);

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const bool theRandomizeValue;  // randomize survey values if true

  std::map<align::StructureType, double> theErrors;
};

SurveyInputDummy::SurveyInputDummy(const edm::ParameterSet& cfg)
    : tTopoToken_(esConsumes()),
      tkGeomToken_(esConsumes()),
      theRandomizeValue(cfg.getParameter<bool>("randomizeValue")) {
  typedef std::vector<edm::ParameterSet> ParameterSets;

  const ParameterSets& errors = cfg.getParameter<ParameterSets>("errors");

  unsigned int nError = errors.size();

  // FIXME: - currently defaulting to RunI as this was the previous behaviour
  //        - check this, when resurrecting this code in the future
  AlignableObjectId alignableObjectId{AlignableObjectId::Geometry::General};

  for (unsigned int i = 0; i < nError; ++i) {
    const edm::ParameterSet& error = errors[i];

    theErrors[alignableObjectId.stringToId(error.getParameter<std::string>("level"))] =
        error.getParameter<double>("value");
  }
}

void SurveyInputDummy::analyze(const edm::Event&, const edm::EventSetup& setup) {
  if (theFirstEvent) {
    //Retrieve tracker topology from geometry
    const TrackerTopology* const tTopo = &setup.getData(tTopoToken_);
    const TrackerGeometry* tracker = &setup.getData(tkGeomToken_);

    Alignable* ali = new AlignableTracker(tracker, tTopo);

    addSurveyInfo(ali);
    addComponent(ali);

    theFirstEvent = false;
  }
}

void SurveyInputDummy::addSurveyInfo(Alignable* ali) {
  static TRandom3 rand;

  const align::Alignables& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
    addSurveyInfo(comp[i]);

  align::ErrorMatrix cov;  // default 0

  std::map<align::StructureType, double>::const_iterator e = theErrors.find(ali->alignableObjectId());

  if (theErrors.end() != e) {
    double error = e->second;

    if (theRandomizeValue) {
      double x = rand.Gaus(0., error);
      double y = rand.Gaus(0., error);
      double z = rand.Gaus(0., error);
      double a = rand.Gaus(0., error);
      double b = rand.Gaus(0., error);
      double g = rand.Gaus(0., error);

      align::EulerAngles angles(3);

      angles(1) = a;
      angles(2) = b;
      angles(3) = g;

      ali->move(ali->surface().toGlobal(align::LocalVector(x, y, z)));
      ali->rotateInLocalFrame(align::toMatrix(angles));
    }

    cov = ROOT::Math::SMatrixIdentity();
    cov *= error * error;
  }

  ali->setSurvey(new SurveyDet(ali->surface(), cov));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SurveyInputDummy);
