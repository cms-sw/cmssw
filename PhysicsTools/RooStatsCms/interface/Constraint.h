/// Constraint: the constraint

/**
\class Constraint
$Revision: 1.3 $
$Date: 2009/02/25 12:59:59 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class describes the single constraint.
Two possible types of constraints are available:
 - Gaussian (can be correlated)
 - LogNormal (the 2nd parameter stands for the percentage of the uncertainity)

The syntax to add them is:

\verbatim
# lognormal
scale = 1 L (0 - 20)
scale_constraint = LogNormal,1,0.15

# Gaussian
hzz_4mu_bkg_yield =0.5 L(0 - 200)
hzz_4mu_bkg_yield_constraint = Gaussian,0.5,0.10
\endverbatim

For more detail see RscCombinedModel.
**/

#ifndef __Constraint__
#define __Constraint__


#include "TNamed.h"
#include "TString.h"

#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"

#include "PhysicsTools/RooStatsCms/interface/NLLPenalty.h"

class Constraint : public NLLPenalty,public RooRealVar {

  public:

    Constraint& operator=(const Constraint& c){return *(Constraint*)c.Clone();};


    /// Constructor
    Constraint (const char* name,
                const char* title,
                double value,
                double minValue,
                double maxValue,
                const char* description,
                const char* unit="");

    /// Constructor
    Constraint (RooRealVar& var, const char* description);

    /// Default Constructor
    Constraint ();

    /// Clone for the RooWorkspace
    TObject* clone(const char* newname) const{
        RooRealVar dummy(*(static_cast<const RooRealVar*> (this)));
        return new Constraint(dummy, m_description->Data());
        };

    /// Clone function for Rooworkspace
    TObject* Clone(const char* newname=0) const { 
        return clone(newname?newname:GetName()) ;};

    /// Print the relevant information
    void print(const char* options="");

    /// Fluctuate parameters
    void fluctuate();

    /// Restore parameters original values
    void restore();

    /// Get NLL string
    TString getNLLstring(){return m_NLL_string;};

    /// Get NLL string for background
    TString getBkgNLLstring();

    /// Get NLL terms in a list
    RooArgList getNLLterms(){return *m_parameters;}

    /// Get NLL terms in a list for background
    RooArgList getBkgNLLterms();

    /// Get the distribution
    RooAbsPdf* getDistribution(){return m_distribution;};

    /// Get the original value
    double getOriginalValue(){return m_original_value;};

    /// Set fixed
    void setFixed(bool fix);

    /// Destructor
    ~Constraint();


  private:

    /// Init the distribution of the constraint
    void m_init_distr(const char* description);

    /// Read params in the description string
    void m_stringToParams(const char* distr_description,
                          std::string& distr_name,
                          std::vector<double>& param_vals);

    /// An hit or miss generator: RooFit does not provide it.
    double m_generate();

    /// Parameters of constraint distributions
    RooRealVar* m_mean;
    RooRealVar* m_sigma;

    /// Distribution name
    TString m_distr_name;

    /// Original value, to restore it after fluctuations
    double m_original_value;

    /// Constraint distribution
    RooAbsPdf* m_distribution;

    /// List of the distribution parameters
    RooArgList* m_parameters;

    /// Penalty NLL string
    TString m_NLL_string;

    /// Description of the distribution
    TString* m_description;

    // For Cint
    //ClassDef(Constraint,1) 
 };

#endif
