// @(#)root/hist:$Id: RscCombinedModel.h,v 1.1 2009/01/06 12:18:37 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

/// RscCombinedModel : a class to combine models described by RscTotModel instances.

/**
\class RscCombinedModel
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott (grgory.schott<at>cern.ch) - Universitaet Karlsruhe 

This class is meant to represent the combination of models.
The idea is to have it behave like a RscTotModel objects container.
If in the card a parameter is cited multiple times with the same name, it will 
be considered as the same 
parameter (\b same \b name-same \b pointer \b mechanism). To be specific, 
to the same name will correspond the same address in the memory.</br>
This mechanism provides the possibility to get the maximum statistical 
advantage combining the inputs of different analyses. Indeed, one can consider 
simultaneously different channels so to infere information on a parameter which 
is common to all of them, i.e. Higgs cross-section, integrated luminosity, etc.
Here we quote a sample card which allows to reproduce the combination of 
the H->ZZ->4leptons analyses in the CMS Physics Technical Design Report. 
The comments begin with a "#".
In the card the user can see how to:
 - Specify a model
 - Specify a constraint
 - Express a yield as a product of multiple factors
 - Use the same name for a variable to express the fact that the correlation is 
100%. In the example a variable "scale" is used to bind together all the 
backgrounds. A more appropriate quantity could have been the lumi. 
At the end of this page it is also shown how to correlate gaussian variables.

\verbatim
# Combination card:
#
# Models:
#
# 1) H --> ZZ --> 2mu 2e
# 2) H --> ZZ --> 2mu 2e
# 3) H --> ZZ --> 4e

# Constraints:
# The constraint on a variable called "var" must be espressed in a variable
# called var_constraint. The syntax for the constraints of different shape are
# possible:
# - Gaussian:
#   example: var_constraint= "Gaussian, 10, 0.3"
#   This line generates a gaussian constraint whose mean is 10 and the sigma is 
#   the 30%. If the mean is 0, the sigma is read as an absolute value.


################################################################################
# The combined model
################################################################################
// Here we specify the names of the models built down in the card that we want
// to be combined
[hzz4l]
    model = combined
    components = hzz_4mu, hzz_4e, hzz_2mu2e

################################################################################
# H -> ZZ -> 4mu
################################################################################

[hzz_4mu]
    variables = x
    x = 0 L(0 - 1)

[hzz_4mu_sig]
    hzz_4mu_sig_yield = 62.78 L(0 - 200)

[hzz_4mu_sig_x]
    model = yieldonly

[hzz_4mu_bkg]

    yield_factors_number = 2

    yield_factor_1 = scale
    scale = 1 C L (0 - 3)
    scale_constraint = Gaussian,1,0.041

    yield_factor_2 = bkg_4mu
    bkg_4mu = 19.93 C


[hzz_4mu_bkg_x]
    model = yieldonly

################################################################################
# H -> ZZ -> 2mu 2e
################################################################################

[hzz_2mu2e]
    variables = x
    x = 0 L(0 - 1)

[hzz_2mu2e_sig]
    model = yieldonly
    hzz_2mu2e_sig_yield = 109.30 L(0 - 200)
[hzz_2mu2e_sig_x]
    model = yieldonly

[hzz_2mu2e_bkg] 
    yield_factors_number = 2

    yield_factor_1 = scale
    scale = 1 C L (0 - 3)
    scale_constraint = Gaussian,1,0.041

    yield_factor_2 = bkg_2mu2e
    bkg_2mu2e = 48.6 C

[hzz_2mu2e_bkg_x]
    model = yieldonly



################################################################################
# H -> ZZ -> 4e
################################################################################

# Here you can see an example about how a Yield can be set to be composed of 
# different factors, so to be able to fit for parameters like: lumi, xsections..
# E.g. Yield = lumi*xsec*eff


[hzz_4e]
    variables = x
    x = 0 L(0 - 1)

[hzz_4e_sig]
    hzz_4e_sig_yield = 38.20 L(0 - 200)

[hzz_4e_sig_x]
    model = yieldonly

[hzz_4e_bkg]

    yield_factors_number = 2

    yield_factor_1 = scale
    scale = 1 C L (0 - 3)
    scale_constraint = Gaussian,1,0.041

    yield_factor_2 = bkg_4e
    bkg_4e = 17.29 C

[hzz_4e_bkg_x]
    model = yieldonly
\endverbatim

This card gives rise to a model, that, once built can be represented in the following scheme:

\image html combined_model_diagram.png

Observe how the variable scale is common to the three backgrounds.\n

To produce blocks of three gaussian correlated variables with correlation 
coefficients different than one, the snippet we should have added is the 
following:

\verbatim
################################################################################
# The correlations
################################################################################

# The correlations are expressed in blocks. Each blocks contains variables.
# Only 3 or 2 variables can be grouped in a block.
# At first the names of the variables are listed, then the values of the 
# correlations coefficients.
# The correlation coefficient 1 represent the correlation between the variables 
# 1-2 and so on, as listed below:
#   - corr between var 1 and 2 = correlation_value1
#   - corr between var 1 and 3 = correlation_value2
#   - corr between var 2 and 3 = correlation_varue3


[constraints_block_1]

correlation_variable1 = hzz_2mu2e_bkg_yield
correlation_variable2 = hzz_4mu_bkg_yield
correlation_variable3 = hzz_4e_bkg_yield

correlation_value1 = 0.99 C
correlation_value2 = 0.99 C
correlation_value3 = 0.99 C
\endverbatim

And to correlate only two:
\verbatim

[constraints_block_1]

correlation_variable1 = hzz_2mu2e_bkg_yield
correlation_variable2 = hzz_4mu_bkg_yield

correlation_value1 = 0.99 C
\endverbatim


**/

#ifndef __RscCombinedModel__
#define __RscCombinedModel__


#include "TNamed.h"
#include "TString.h"
#include "TList.h"

#include "PhysicsTools/RooStatsCms/interface/RscTotModel.h"
#include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"
#include "PhysicsTools/RooStatsCms/interface/PdfCombiner.h"

#include "RooAbsPdf.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooWorkspace.h"

/// Enumerator for SB or B
enum ComponentCode {kBKG, kSIGBKG, kSIG};


class RscCombinedModel : public TNamed  {

  public:

    /// Constructor
    RscCombinedModel(const char* name,
                     const char* title,
                     RscTotModel* model_1,
                     RscTotModel* model_2=0,
                     RscTotModel* model_3=0,
                     RscTotModel* model_4=0,
                     RscTotModel* model_5=0,
                     RscTotModel* model_6=0,
                     RscTotModel* model_7=0,
                     RscTotModel* model_8=0,
                     RscTotModel* model_9=0,
                     RscTotModel* model_10=0);

    /// Constructor from datacard
    RscCombinedModel(const char* combined_model_name);


    /// Destructor
    ~RscCombinedModel();

    /// Print to screen the info about the combination
    void print(const char* option="");

    /// Returns the pdf of the combination.
    RooAbsPdf* getPdf();

    /// Returns the pdf of the combination.
    RooAbsPdf* getSigPdf();

    /// Returns the pdf of the combination.
    RooAbsPdf* getBkgPdf();

    /// Return the Category of the combinedmodel
    RooCategory* getCategory(ComponentCode code);

    /// Returns the TotModel.
    RscTotModel* getModel (int index);

    /// Returns the TotModel.
    RscTotModel* getModel (char* name);

    /// Set the verbosity of the object
    void setVerbosity(bool verbose);

    /// Get Parameter from the combination
    RooRealVar* getParameter(TString name);

    /// Get all the parameters from the combination
    RooArgList getParameters();

    /// Get the variables of the models involved in the combination
    RooArgList getVars();

    /// Get the constraints from the combination
    RooArgList getConstraints();

    /// Get the number of combined models
    int getSize(){return m_models_number;};

    /// get the Workspace
    RooWorkspace* getWorkspace(){return m_workspace;};


  private:
    /// Expand the string of names of models in case combined models are there
    TString m_expand_components(TString combined_model_name_s);

    /// Check if the model is combined
    bool m_is_combined(const char* combined_model_name);

    /// Method to add a model and its variable to the combination
    void m_add(RscTotModel* model);

    /// Flag for contents owning
    bool m_own_contents;

    /// The sig Pdf combiner
    PdfCombiner* m_sigPdf_combiner;

    /// The bkg Pdf combiner
    PdfCombiner* m_bkgPdf_combiner;

    /// The Pdf combiner
    PdfCombiner* m_pdf_combiner;

    /// The buffer for the total pdf
    RooAbsPdf* m_pdf_buf;

    /// The buffer for the signal pdf
    RooAbsPdf* m_sigPdf_buf;

    /// The buffer for the background pdf
    RooAbsPdf* m_bkgPdf_buf;

    /// The verbosity flag
    bool m_verbose;

    /// The internal representation of the combined RscTotModels
    TList m_models_list;

    /// Number of models collected
    int m_models_number;

    /// RooArgList of the constraints
    RooArgList* m_constraints;

    /// Find variable by name in a RooArgSet
    RooRealVar* m_find_in_set(RooArgSet* set,TString name);

    /// The internal RooWorkspace of the combined model istance
    RooWorkspace* m_workspace;

    // For Cint
    //ClassDef(RscCombinedModel,1)
};

#endif
