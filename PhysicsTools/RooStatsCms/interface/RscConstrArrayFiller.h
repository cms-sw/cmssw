// @(#)root/hist:$Id: RscConstrArrayFiller.h,v 1.4 2009/05/15 09:55:43 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

/// ConstrArrayFiller : The mother class of the RooStatsCms Tools

/**
\class ConstrArrayFiller
$Revision: 1.4 $
$Date: 2009/05/15 09:55:43 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott (grgory.schott<at>cern.ch) - Universitaet Karlsruhe 
Fill the array of constraints considering the Constraints in a collection of 
RscTotModels and reading from the datacard the correlations.
**/

#ifndef __RscConstrArrayFiller__
#define __RscConstrArrayFiller__

#include "TString.h"

#include "RooArgList.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "ConstrBlockArray.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"
#endif

#include "RscTool.h"
#include "RscCombinedModel.h"

class RscConstrArrayFiller : public RscTool  {

  public:

    /// Constructor
    RscConstrArrayFiller(const char* name, 
                         const char* title, 
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
    RscConstrArrayFiller(const char* name, 
                         const char* title, 
                         RscCombinedModel* combo,
                         bool verbosity=true);

    /// Constructor with RscCombinedModel
    RscConstrArrayFiller(const char* name, 
                         const char* title, 
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


//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(RscConstrArrayFiller,1)
#endif
};

#endif

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
