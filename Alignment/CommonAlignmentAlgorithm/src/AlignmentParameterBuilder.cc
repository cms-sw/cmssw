/** \file AlignableParameterBuilder.cc
 *
 *  $Date: 2007/03/02 12:16:56 $
 *  $Revision: 1.14 $

*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/SelectionUserVariables.h"

// This class's header

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"


//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker) :
  theAlignables(), theAlignableTracker(alignableTracker), theAlignableMuon(0)
{
}

//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker, 
                                                     AlignableMuon* alignableMuon) :
  theAlignables(), theAlignableTracker(alignableTracker), theAlignableMuon(alignableMuon)
{
}


//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker,
                                                     const edm::ParameterSet &pSet) :
  theAlignables(), theAlignableTracker(alignableTracker), theAlignableMuon(0)
{
  this->addSelections(pSet.getParameter<edm::ParameterSet>("Selector"));
}


//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker,
                                                     AlignableMuon* alignableMuon,
                                                     const edm::ParameterSet &pSet) :
  theAlignables(), theAlignableTracker(alignableTracker), theAlignableMuon(alignableMuon)
{
  this->addSelections(pSet.getParameter<edm::ParameterSet>("Selector"));
}


//__________________________________________________________________________________________________
unsigned int AlignmentParameterBuilder::addSelections(const edm::ParameterSet &pSet)
{

  AlignmentParameterSelector selector( theAlignableTracker, theAlignableMuon );
  const unsigned int addedSets = selector.addSelections(pSet);

  const std::vector<Alignable*> &alignables = selector.selectedAlignables();
  const std::vector<std::vector<char> > &paramSels = selector.selectedParameters();

  std::vector<Alignable*>::const_iterator iAli = alignables.begin();
  std::vector<std::vector<char> >::const_iterator iParamSel = paramSels.begin();
  unsigned int nHigherLevel = 0;

  while (iAli != alignables.end() && iParamSel != paramSels.end()) {
    std::vector<bool> boolParSel;
    bool charSelIsGeneral = this->decodeParamSel(*iParamSel, boolParSel);
    if (this->add(*iAli, boolParSel)) ++nHigherLevel;
    if (charSelIsGeneral) this->addFullParamSel((*iAli)->alignmentParameters(), *iParamSel);

    ++iAli;
    ++iParamSel;
  }

  edm::LogInfo("Alignment") << "@SUB=AlignmentParameterBuilder::addSelections"
                            << " Added " << addedSets << " set(s) of alignables with "
                            << theAlignables.size() << " alignables in total,"
                            << " of which " << nHigherLevel << " are higher level.";
   
  return addedSets;
}

//__________________________________________________________________________________________________
bool AlignmentParameterBuilder::add(Alignable *alignable, const std::vector<bool> &sel)
{ 

  const AlgebraicVector par(RigidBodyAlignmentParameters::N_PARAM, 0);
  const AlgebraicSymMatrix cov(RigidBodyAlignmentParameters::N_PARAM, 0);

  // Which kind of AlignmentParameters must be selectable once we have other parametrisations:
  AlignmentParameters *paras = new RigidBodyAlignmentParameters(alignable, par, cov, sel);
  alignable->setAlignmentParameters(paras);
  theAlignables.push_back(alignable);

  const int aliTypeId = alignable->alignableObjectId();
  const bool isHigherLevel = (aliTypeId != AlignableObjectId::AlignableDet
			      && aliTypeId != AlignableObjectId::AlignableDetUnit);
  return isHigherLevel;
}


//__________________________________________________________________________________________________
unsigned int AlignmentParameterBuilder::add(const std::vector<Alignable*> &alignables,
                                            const std::vector<bool> &sel)
{

  unsigned int nHigherLevel = 0;

  for (std::vector<Alignable*>::const_iterator iAli = alignables.begin();
       iAli != alignables.end(); ++iAli) {
    if (this->add(*iAli, sel)) ++nHigherLevel;
  }

  return nHigherLevel;
}


//__________________________________________________________________________________________________
void AlignmentParameterBuilder::fixAlignables(int n)
{

  if (n<1 || n>3) {
    edm::LogError("BadArgument") << " n = " << n << " is not in [1,3]";
    return;
  }

  std::vector<Alignable*> theNewAlignables;
  int i=0;
  int imax = theAlignables.size();
  for ( std::vector<Alignable*>::const_iterator ia=theAlignables.begin();
        ia!=theAlignables.end();  ia++ ) 
	{
	  i++;
	  if ( n==1 && i>1 ) 
		theNewAlignables.push_back(*ia);
	  else if ( n==2 && i>1 && i<imax ) 
		theNewAlignables.push_back(*ia);
	  else if ( n==3 && i>2 && i<imax) 
		theNewAlignables.push_back(*ia);
	}

  theAlignables = theNewAlignables;

  edm::LogInfo("Alignment") << "@SUB=AlignmentParameterBuilder::fixAlignables"
                            << "removing " << n << " alignables, so that " 
                            << theAlignables.size() << " alignables left";
}

//__________________________________________________________________________________________________
bool AlignmentParameterBuilder::decodeParamSel(const std::vector<char> &paramSelChar,
                                               std::vector<bool> &result) const
{
  result.clear();
  bool anyNon01 = false;

  for (unsigned int pos = 0; pos < paramSelChar.size(); ++pos) {

    switch (paramSelChar[pos]) {
    default:
      anyNon01 = true;
      // no break;
    case '1':
      result.push_back(true);
      break;
    case '0':
      result.push_back(false);
      break;
    }
  }

  return anyNon01;
}

//__________________________________________________________________________________________________
bool AlignmentParameterBuilder::addFullParamSel(AlignmentParameters *aliParams,
                                                const std::vector<char> &fullSel) const
{
  if (!aliParams) return false;

  aliParams->setUserVariables(new SelectionUserVariables(fullSel));

  return true;
}
