// @(#)root/hist:$Id: Constraint.cc,v 1.1 2009/01/06 12:22:43 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "TRandom.h"

#include "PhysicsTools/RooStatsCms/interface/Constraint.h"

#include "RooGaussian.h"
#include "RooGenericPdf.h"
#include "RooRandom.h"

#include "RooDataSet.h"

/*----------------------------------------------------------------------------*/

/**
Construct the RooRealVar and the distribution of the constraint according to 
the string in the description.
**/

Constraint::Constraint (RooRealVar& var, const char* description)
    :RooRealVar(var),
     m_description(0){
    m_description = new TString(description);

    setVerbosity(false);

    m_original_value=var.getVal();

    m_init_distr(description);

    }

/*----------------------------------------------------------------------------*/

/**
Construct the RooRealVar and the distribution of the constraint according to 
the string in the description.
**/

Constraint::Constraint(const char* name,
                       const char* title,
                       double value,
                       double minValue,
                       double maxValue,
                       const char* description,
                       const char* unit)
    :RooRealVar(name, title, value, minValue, maxValue, unit),
     m_description(0){

    setVerbosity(false);

    m_description = new TString(description);

    m_original_value=value;

    m_init_distr(description);

    }

/*----------------------------------------------------------------------------*/

Constraint::Constraint ()
    :RooRealVar("Variable_name","Variable_title",0,-1,1,""),
     m_description(0){

    setVerbosity(false);

    m_description = new TString("Gaussian,0,1");

    m_original_value=0;

    m_init_distr(m_description->Data());

    }

/*----------------------------------------------------------------------------*/

/**
Build the distribution according to the description string.
The penalty term of the lognormal distribution is privated of the 
log(log(sigma)) term since sigma can be less than the neper number.
This has almost no impact since the NLL differences are almost always 
considered.
**/

void Constraint::m_init_distr(const char* description){


    // Read the description and fill the parameters values
    std::string distr_name;
    std::vector<double> param_vals;
    m_stringToParams(description, distr_name, param_vals);

    m_distr_name=distr_name.c_str();

    // Allocate the list of variables
    m_parameters=new RooArgList ("parameter_list");
    m_parameters->add(*((RooRealVar*)this));

    // Choose the right distribution

    // 1. Gaussian Distribution

    if (distr_name=="Gaussian"){

        // Allocate the variables for the gaussian

        TString mean_name(GetName());
        mean_name+="_gaussian_constr_mean";
        m_mean =
               new RooRealVar (mean_name.Data(),mean_name.Data(),param_vals[0]);

        m_parameters->add(*m_mean);

        TString sigma_name(GetName());
        sigma_name+="_gaussian_constr_sigma";
        double sigma_val;
        if (param_vals[0]==0)
            sigma_val=param_vals[1];
        else
            sigma_val=param_vals[1]*param_vals[0];
        m_sigma =
             new RooRealVar (sigma_name.Data(),sigma_name.Data(),sigma_val);

        m_parameters->add(*m_sigma);

        // Build the distribution
        TString distr_name=GetName();
        distr_name+="_gaussian";
        m_distribution=new RooGaussian (distr_name.Data(),
                                        distr_name.Data(),
                                        *(static_cast<RooRealVar*>(this)),
                                        *m_mean,
                                        *m_sigma);

        // Write down the nll penalty term
        TString this_name=GetName();
        m_NLL_string= " + 0.5*(("+this_name+"-"+mean_name+")/"+sigma_name+")**2";

        }

    // Log Normal Distribution
    else if (distr_name=="LogNormal"){
        // Allocate the variables for the gaussian

        TString mean_name=GetName();
        mean_name+="_lognormal_constr_mean";
        m_mean =
               new RooRealVar (mean_name.Data(),mean_name.Data(),param_vals[0]);

        m_parameters->add(*m_mean);

        TString sigma_name=GetName();
        sigma_name+="_lognormal_constr_sigma";
        double sigma_val;
        sigma_val=param_vals[1]+1;
        m_sigma =
             new RooRealVar (sigma_name.Data(),sigma_name.Data(),sigma_val);

        m_parameters->add(*m_sigma);

        // Build the distribution
        TString distr_name=GetName();
        distr_name+="_lognormal";
        m_distribution= new RooGenericPdf
            (distr_name.Data(),
             "lognormal",
             "(1/@0)*exp(-0.5*((log(@0)-log(@1))/log(@2))**2) ",//Roofit provides normalisation!
             RooArgSet(*(static_cast<RooRealVar*>(this)),*m_mean,*m_sigma));

        // Write down the nll penalty term
        TString this_name=GetName();

        m_NLL_string = " + log("+this_name+")";
        m_NLL_string += " + 0.5 * ( (log("+this_name+")-log("+mean_name+"))/log("+sigma_name+") )**2";

        }

    // N.
    else{
        std::cout << "[Constraint::Constraint] "
                  << " no distribution " << distr_name 
                  << " recognised, aborting...\n";
        abort();
        }

    }

/*----------------------------------------------------------------------------*/

/**
A simple parser.It separates the values contained in distr_description using 
the fact that they are comma separated and fill the distr_name and the vector 
of parameters names.
**/

void Constraint::m_stringToParams(const char* distr_description, 
                                  std::string& distr_name, 
                                  std::vector<double>& param_vals){

    // Convert to a string and add a last comma
    std::string descr(distr_description);
    descr+=",";
    //std::cout << descr << std::endl;

    // Separate at commas
    unsigned int index=0;
    char current_char=descr[index];

    // first value is the name, the rest paramvals
    do{distr_name+=current_char;
       index++;
       current_char=descr[index];
      }while(current_char!=',');

    index++;
    current_char=descr[index];

    std::string tmp("");
    while(index<descr.length()){
        tmp="";
        do{tmp+=current_char;
           index++;
           current_char=descr[index];
          }while(current_char!=',');
        index++;
        current_char=descr[index];
        //std::cout << "Param is " << tmp << "\n";
        param_vals.push_back(atof(tmp.c_str()));
        }

    if (param_vals.size()==0){
        std::cout << "[Constraint::StringToParams] " 
               << "0 parameters detected. " 
               << "The syntax is <distribution>, <param1 val> [param2 val]..\n"
               << "Aborting\n";
        abort();
        }
    }

/*----------------------------------------------------------------------------*/

void Constraint::print(const char* options){
    std::cout << m_distr_name.Data() << " Constraint " << GetName() << ":\n"
              << "\nParameters:\n";
    RooRealVar* par;
    for (int i=0;i<m_parameters->getSize();++i){
        par=(RooRealVar*) &((*m_parameters)[i]);
        std::cout << " - " << par->ClassName() << " object " << par->GetName()
                  << " " << par->getVal() << std::endl;
        }
    std::cout << "\nDistribution:\n";
    m_distribution->Print(options);
    }

/*----------------------------------------------------------------------------*/

/**
The value of the nuisance parameter is fluctuated according the limits of the 
very parameter and its distribution. This happens throwing a simple random 
number.
**/
void Constraint::fluctuate(){
    double newval=m_generate();
    if (is_verbose())
        std::cout << "[Constraint::fluctuate] "
                  << getVal() << " --> " << newval << std::endl;
    this->setVal(newval);
    }

/*----------------------------------------------------------------------------*/

/// Restore the value of the parameter from a stream
/**
Brings back the original value. This is done to be able to restore the values 
after a Fit or a fluctuation to avoid strange behaviours in long loops, e.g. 
feed minuit with the last fitted value.
**/
void Constraint::restore(){

    if (is_verbose())
        std::cout << "[Constraint::restore] "
                  << getVal() << " --> " << m_original_value << std::endl;
    this->setVal(m_original_value);
    }

/*----------------------------------------------------------------------------*/

void Constraint::setFixed(bool fix){
    setConstant(fix);
    }
/*----------------------------------------------------------------------------*/

double Constraint::m_generate(){
    RooDataSet* evt=m_distribution->generate(*this,1);
    const RooArgSet *arg=evt->get();
    double rndm=arg->getRealValue(this->GetName());
    delete evt;

   return rndm;
//     double rnd=gRandom->Gaus(m_mean->getVal(),m_sigma->getVal());
//     return rnd;

}

/*----------------------------------------------------------------------------*/

/**
If it is a constraint on the signal, return " + 0" !
**/

TString Constraint::getBkgNLLstring(){

    if (TString(GetName()).Contains(SIG_KEYWORD)!=0)
        return TString(" + 0");
    else
        return getNLLstring();

    }

/*----------------------------------------------------------------------------*/

/**
If it is a constraint on the signal, return an empty list !
**/

RooArgList Constraint::getBkgNLLterms(){

    if (TString(GetName()).Contains(SIG_KEYWORD)!=0)
        return RooArgList();
    else
        return getNLLterms();

    }

/*----------------------------------------------------------------------------*/

Constraint::~Constraint(){

    //cout << m_parameters->isOwning() << std::endl;
    delete m_description;

    delete m_mean;
    delete m_sigma;

    delete m_parameters;

    delete m_distribution;
    }

/*----------------------------------------------------------------------------*/

/// To build the cint dictionaries
//ClassImp(Constraint)

/*----------------------------------------------------------------------------*/
