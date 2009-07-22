// @(#)root/hist:$Id: RscConstrArrayFiller.h,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

/// ConstrArrayFiller : The mother class of the RooStatsCms Tools

/**
\class ConstrArrayFiller
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott (grgory.schott<at>cern.ch) - Universitaet Karlsruhe 
Fill the array of constraints considering the Constraints in a collection of 
RscTotModels and reading from the datacard the correlations.
**/

#ifndef __RscConstrArrayFiller__
#define __RscConstrArrayFiller__

#include "TString.h"

#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"

#include "RscTool.h"
#include "RscCombinedModel.h"

class RscConstrArrayFiller : public RscTool  {

  public:

    /// Constructor
    RscConstrArrayFiller(char* name, 
                         char* title, 
                         RscTotModel* model1,
                         RscTotModel* model2=0,
                         RscTotModel* model3=0,
                         RscTotModel* model4=0,
                         RscTotModel* model5=0,
                         RscTotModel* model6=0,
                         RscTotModel* model7=0,
                         RscTotModel* model8=0,
                         bool verbosity=true);

    /// Constructor with RscCombinedModel
    RscConstrArrayFiller(char* name, 
                         char* title, 
                         RscCombinedModel* combo,
                         bool verbosity=true);

    /// Constructor with RscCombinedModel
    RscConstrArrayFiller(char* name, 
                         char* title, 
                         RooArgList constraints_list,
                         bool verbosity=true);

    /// Fill the array reading from the card
    void fill (ConstrBlockArray* the_array, const char* blocknamebase="");


  private:

    /// Fill the single block
    NLLPenalty* m_fill_single_block (TString block_name);

    /// Add all the list elements that are not present
    void m_add_selective(RooArgList* list);

    /// Pop constraint from the m_uncorr_constraints list 
    Constraint* m_pop(TString name);

    /// Internal representation of the uncorrelated constraints
    RooArgList m_constraints;

    /// Internal representation of the correlated constraints
    RooArgList m_corr_constraints;

    /// Verbosity flag
    bool m_verbose;


};

#endif

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
