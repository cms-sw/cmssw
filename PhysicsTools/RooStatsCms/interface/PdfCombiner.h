// @(#)root/hist:$Id: PdfCombiner.h,v 1.3 2009/04/15 11:10:45 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

/// PdfCombiner : a class to combine models

/**
\class PdfCombiner
$Revision: 1.3 $
$Date: 2009/04/15 11:10:45 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott (grgory.schott<at>cern.ch) - Universitaet Karlsruhe 
This class is meant to represent the combination of models.
The idea is to have it behave like a pdf container.
**/

#ifndef __PdfCombiner__
#define __PdfCombiner__

//#include <stdlib.h>
#include <string>

#include "RooAbsPdf.h"
#include "RooArgList.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooSimultaneous.h"

class PdfCombiner : public TNamed {

  public:
    /// Constructor
    PdfCombiner(std::string name);

    /// Constructor
    PdfCombiner(std::string name, RooCategory* cat);

    /// Default Constructor
    PdfCombiner();

    /// Destructor
    ~PdfCombiner();

    /// Method to add a model and its variable to the combination
    int add(RooAbsPdf* model, RooRealVar* x);

    /// Method to add a model and its variable(s) to the combination
    int add(RooAbsPdf* model, RooArgList* var_list);

    /// Print to screen the info about the combination
    void Print(const char* option="");

    /// Returns the pdf of the combination.
    RooAbsPdf* getPdf(int index=-1);

    /// Return the list of variables used
    RooArgList getVars();

    /// Get the category
    RooCategory* getCategory();

    /// Get the number of combined pdfs
    int getCombinedPdfs(){return m_model_collection->getSize();};

    /// Set the verbosity of the object
    void setVerbosity(bool verbose);

  private:

    /// A flag to keep race of the provenance of the category
    const bool m_external_category;

    /// The verbosity flag
    bool m_verbose;

    /// Buffer for the combined model
    RooSimultaneous* m_combined_model_buffer;

    /// Container for the models
    RooArgList* m_model_collection;

    /// Container for the models vars
    RooArgList* m_variables_collection;

    /// Category tocreate the RooSimultaneousPdf
    RooCategory* m_category;



};

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009
