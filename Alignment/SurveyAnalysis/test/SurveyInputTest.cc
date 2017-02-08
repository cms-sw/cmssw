#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"

#include "Alignment/SurveyAnalysis/test/SurveyInputTest.h"

using namespace align;

SurveyInputTest::SurveyInputTest(const edm::ParameterSet& cfg):
  theConfig(cfg)
{
}

void SurveyInputTest::beginJob()
{
  addComponent( create( theConfig.getParameter<std::string>("detector") ) );
}

Alignable* SurveyInputTest::create(const std::string& parName)
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
  Doubles errors = pars.getUntrackedParameter<Doubles>("errors", zero6Vector);

  ErrorMatrix cov;

  for (unsigned int i = 0; i < 6; ++i) cov(i, i) = errors[i];

  PositionType pos(center[0], center[1], center[2]);
  EulerAngles ang(3);

  ang(1) = angles[0], ang(2) = angles[1]; ang(3) = angles[2];

  AlignableSurface surf( pos, toMatrix(ang) );

  surf.setWidth ( pars.getUntrackedParameter<double>("width" , 0.) );
  surf.setLength( pars.getUntrackedParameter<double>("length", 0.) );

          int type = pars.getParameter<int>        ("typeId");
  std::string name = pars.getParameter<std::string>("object");

  Alignable* ali = new AlignableComposite( type,AlignableObjectId::stringToId(name),
					   surf.rotation() );

  Strings comp = pars.getUntrackedParameter<Strings>("compon", emptyString);

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
  {
    ali->addComponent( create(comp[i]) );
  }

  ang(1) = shifts[3], ang(2) = shifts[4]; ang(3) = shifts[5];

  ali->setSurvey( new SurveyDet(surf, cov) );
  ali->move( surf.toGlobal( align::LocalVector(shifts[0], shifts[1], shifts[2]) ) );
  ali->rotateInLocalFrame( toMatrix(ang) );

  return ali;
}
