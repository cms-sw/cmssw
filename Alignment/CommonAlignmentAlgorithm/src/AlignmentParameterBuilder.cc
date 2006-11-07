/** \file AlignableParameterBuilder.cc
 *
 *  $Date: 2006/11/07 10:22:56 $
 *  $Revision: 1.9 $
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeRigidBodyAlignmentParameters.h"

//#include "Alignment/TrackerAlignment/interface/AlignableTracker.h" not needed since only forwarded

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

// This class's header

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"


//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker) :
  theAlignables(), theAlignableTracker(alignableTracker)
{
}

//__________________________________________________________________________________________________
AlignmentParameterBuilder::AlignmentParameterBuilder(AlignableTracker* alignableTracker,
						     const edm::ParameterSet &pSet) :
  theAlignables(), theAlignableTracker(alignableTracker)
{
  this->addSelections(pSet.getParameter<edm::ParameterSet>("AlignmentParameterSelector"));
}

//__________________________________________________________________________________________________
unsigned int AlignmentParameterBuilder::addSelections(const edm::ParameterSet &pSet)
{

   AlignmentParameterSelector selector(theAlignableTracker);
   const unsigned int addedSets = selector.addSelections(pSet);

   const std::vector<Alignable*> &alignables = selector.selectedAlignables();
   const std::vector<std::vector<bool> > &paramSels = selector.selectedParameters();

   std::vector<Alignable*>::const_iterator iAli = alignables.begin();
   std::vector<std::vector<bool> >::const_iterator iParamSel = paramSels.begin();
   unsigned int nHigherLevel = 0;

   while (iAli != alignables.end() && iParamSel != paramSels.end()) {
     if (this->add(*iAli, *iParamSel)) ++nHigherLevel;
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

  AlgebraicVector par(RigidBodyAlignmentParameters::N_PARAM, 0);
  AlgebraicSymMatrix cov(RigidBodyAlignmentParameters::N_PARAM, 0);
  bool isHigherLevel = false;
 
  AlignableDet *alidet = dynamic_cast<AlignableDet*>(alignable);
  AlignmentParameters *paras = 0;
  if (alidet != 0) { // alignable Det
    paras = new RigidBodyAlignmentParameters(alignable, par, cov, sel);
  } else { // higher level object
    paras = new CompositeRigidBodyAlignmentParameters(alignable, par, cov, sel);
    isHigherLevel = true;
  }

  alignable->setAlignmentParameters(paras);
  theAlignables.push_back(alignable);

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

