/** \class SurveyTest
 *
 *  Analyser module for testing.
 *
 *  $Date: 2007/10/10 20:54:07 $
 *  $Revision: 1.6 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

class SurveyTest : public edm::one::EDAnalyzer<> {
public:
  SurveyTest(const edm::ParameterSet&);

  virtual void beginJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

private:
  void getTerminals(align::Alignables& terminals, Alignable* ali);

  bool theBiasFlag;  // true for biased residuals

  unsigned int theIterations;  // number of iterations

  std::string theAlgorithm;   // points or sensor residual
  std::string theOutputFile;  // name of output file

  std::vector<align::StructureType> theHierarchy;
};

SurveyTest::SurveyTest(const edm::ParameterSet& cfg)
    : theBiasFlag(cfg.getUntrackedParameter<bool>("bias", false)),
      theIterations(cfg.getParameter<unsigned int>("iterator")),
      theAlgorithm(cfg.getParameter<std::string>("algorith")),
      theOutputFile(cfg.getParameter<std::string>("fileName")) {
  typedef std::vector<std::string> Strings;

  const Strings& hierarchy = cfg.getParameter<Strings>("hierarch");

  // FIXME: - currently defaulting to RunI as this was the previous behaviour
  //        - check this, when resurrecting this code in the future
  AlignableObjectId alignableObjectId{AlignableObjectId::Geometry::General};

  for (unsigned int l = 0; l < hierarchy.size(); ++l) {
    theHierarchy.push_back(alignableObjectId.stringToId(hierarchy[l]));
  }
}

void SurveyTest::beginJob() {
  Alignable* det = SurveyInputBase::detector();

  align::Alignables sensors;

  getTerminals(sensors, det);

  std::map<std::string, SurveyAlignment*> algos;

  algos["points"] = new SurveyAlignmentPoints(sensors, theHierarchy);
  algos["sensor"] = new SurveyAlignmentSensor(sensors, theHierarchy);

  algos[theAlgorithm]->iterate(theIterations, theOutputFile, theBiasFlag);

  for (std::map<std::string, SurveyAlignment*>::iterator i = algos.begin(); i != algos.end(); ++i)
    delete i->second;
}

void SurveyTest::getTerminals(align::Alignables& terminals, Alignable* ali) {
  const auto& comp = ali->components();

  unsigned int nComp = comp.size();

  if (nComp > 0)
    for (unsigned int i = 0; i < nComp; ++i) {
      getTerminals(terminals, comp[i]);
    }
  else
    terminals.push_back(ali);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SurveyTest);
