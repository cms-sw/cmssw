/** \class SurveyTest
 *
 *  Analyzer module for testing.
 *
 *  $Date: 2007/01/29 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include <iomanip>

#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentPoints.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentSensor.h"
#include "Alignment/SurveyAnalysis/test/AlignableTest.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SurveyTest.h"

using namespace align;

SurveyTest::SurveyTest(const edm::ParameterSet& cfg):
  theConfig(cfg)
{
}

Alignable* SurveyTest::create(const std::string& parName)
{
  typedef std::vector<double>      Doubles;
  typedef std::vector<std::string> Strings;

  static const Doubles zero3Vector(3, 0.);
  static const Doubles zero6Vector(6, 0.);
  static const Strings emptyString;

  edm::ParameterSet pars = theConfig.getParameter<edm::ParameterSet>(parName);

  Doubles center = pars.getUntrackedParameter<Doubles>("center", zero3Vector);
  Doubles angles = pars.getUntrackedParameter<Doubles>("angles", zero3Vector);
  Doubles shifts = pars.getUntrackedParameter<Doubles>("shifts", zero6Vector);

  PositionType pos(center[0], center[1], center[2]);
  EulerAngles ang(3);

  ang(1) = angles[0], ang(2) = angles[1]; ang(3) = angles[2];

  AlignableSurface surf( pos, toMatrix(ang) );

  surf.setWidth ( pars.getUntrackedParameter<double>("width" , 0.) );
  surf.setLength( pars.getUntrackedParameter<double>("length", 0.) );

          int type = pars.getParameter<int>        ("typeId");
//   std::string name = pars.getParameter<std::string>("object");

  AlignableTest* ali = new AlignableTest(surf, type);
  SurveyDet* svy = new SurveyDet(surf);

  theSensors.push_back(ali);
  theSurveys.push_back(svy);

  Strings comp = pars.getUntrackedParameter<Strings>("compon", emptyString);

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
  {
    ali->addComponent( create(comp[i]) );
  }

  ang(1) = shifts[3], ang(2) = shifts[4]; ang(3) = shifts[5];

  ali->setSurvey(svy);
  ali->move( surf.toGlobal( align::LocalVector(shifts[0], shifts[1], shifts[2]) ) );
  ali->rotateInLocalFrame( toMatrix(ang) );

  return ali;
}

void getTerminals(std::vector<Alignable*>& terminals,
		  Alignable* ali)
{
  const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  if (nComp > 0)
    for (unsigned int i = 0; i < nComp; ++i)
    {
      getTerminals(terminals, comp[i]);
    }
  else
    terminals.push_back(ali);
}

void SurveyTest::beginJob(const edm::EventSetup&)
{
  Alignable* det = create( theConfig.getParameter<std::string>("detector") );

  std::vector<Alignable*> sensors;

  getTerminals(sensors, det);

  std::map<std::string, SurveyAlignment*> aligns;

  aligns["points"] = new SurveyAlignmentPoints(sensors);
  aligns["sensor"] = new SurveyAlignmentSensor(sensors);

  aligns[theConfig.getParameter<std::string>("algorith")]
    ->iterate( theConfig.getParameter<unsigned int>("iterator"),
	       theConfig.getParameter<std::string> ("fileName") );

  for (std::map<std::string, SurveyAlignment*>::iterator i = aligns.begin();
       i != aligns.end(); ++i) delete i->second;
}

void SurveyTest::analyze(const edm::Event&, const edm::EventSetup&)
{
  std::cout << "SurveyTest: Nothing to analyze." << std::endl;
}

void SurveyTest::endJob()
{
  for (unsigned int i = 0; i < theSensors.size(); ++i) delete theSensors[i];
  for (unsigned int i = 0; i < theSurveys.size(); ++i) delete theSurveys[i];

  theSensors.clear();
  theSurveys.clear();

  std::cout << "SurveyTest: Sensors and Surveys deleted." << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SurveyTest); //define this as a plug-in
