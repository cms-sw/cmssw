#include <sstream>

#include "TNtuple.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
// #include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"

#include "Alignment/SurveyAnalysis/interface/SurveyOutput.h"

using namespace align;

SurveyOutput::SurveyOutput(const std::vector<Alignable*>& alignables,
			   const std::string& fileName):
  theAlignables(alignables),
  theFile(fileName.c_str(), "RECREATE")
{
}

void SurveyOutput::write(unsigned int iter)
{
  std::ostringstream o;

  o << 't' << iter;

  TNtuple* nt = new TNtuple(o.str().c_str(), "", "x:y:z:a:b:g");

  int N = theAlignables.size();

  for (int i = 0; i < N; ++i)
  {
    Alignable* ali = theAlignables[i];

    const PositionType& pos0 = ali->survey()->position();
    const PositionType& pos1 = ali->globalPosition();
    const RotationType& rot0 = ali->survey()->rotation();
    const RotationType& rot1 = ali->globalRotation();

    align::GlobalVector shifts = pos1 - pos0;
    EulerAngles angles = toAngles( rot0.multiplyInverse(rot1) );

    nt->Fill( shifts.x(), shifts.y(), shifts.z(),
	      angles(1), angles(2), angles(3) );
//     const AlgebraicVector& pars = ali->alignmentParameters()->parameters();

//     nt->Fill(pars[0], pars[1], pars[2], pars[3], pars[4], pars[5]);
  }

  theFile.Write();

  delete nt;
}
