// @(#)root/hist:$Id: ConstrBlock2.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"

#include "RooFormulaVar.h"

#include "PhysicsTools/RooStatsCms/interface/ConstrBlock2.h"


/*----------------------------------------------------------------------------*/

/**
Constructor of the class
**/

ConstrBlock2::ConstrBlock2 (const char* name,
                            const char* title,
                            double corr12,
                            Constraint* constr1,
                            Constraint* constr2):
    TNamed(name,title){

    setVerbosity(true);

    // The list to store the parameters involved
    m_parameters = new RooArgList("Parameters");


    // The correlation term
    TString corr_name=GetName();
    corr_name+="_corr";
    m_corr=new RooRealVar(corr_name.Data(),
                         corr_name.Data(),
                         corr12);

    m_parameters->add(*m_corr);

    m_parameters->add(constr1->getNLLterms());
    m_parameters->add(constr2->getNLLterms());

    m_constr_list =  new RooArgList("Constr list");

    m_constr_list->add(*constr1);
    m_constr_list->add(*constr2);

    RooRealVar* mean1=static_cast<RooRealVar*>(&(constr1->getNLLterms())[1]);
    RooRealVar* mean2=static_cast<RooRealVar*>(&(constr2->getNLLterms())[1]);

    RooRealVar* sigma1=static_cast<RooRealVar*>(&(constr1->getNLLterms())[2]);
    RooRealVar* sigma2=static_cast<RooRealVar*>(&(constr2->getNLLterms())[2]);

    TString res1_n="res1_";
    res1_n+=GetName();
    RooFormulaVar* res1=new RooFormulaVar (res1_n.Data(),
                                           "((@0-@1)/@2)",
                                           RooArgList(*constr1,*mean1,*sigma1));


    TString res2_n="res2_";
    res2_n+=GetName();
    RooFormulaVar* res2=new RooFormulaVar (res2_n.Data(),
                                           "((@0-@1)/@2)",
                                           RooArgList(*constr2,*mean2,*sigma2));

//     RooFormulaVar* rho2=new RooFormulaVar ("rho2",
//                                            "@0**2",
//                                             RooArgList(*m_corr));

    m_parameters->add(RooArgList(*res1,*res2));

    m_NLL_string  = " + (0.5/(1-";
    m_NLL_string += corr_name;
    m_NLL_string += "**2))*("+res1_n+"**2+"+res2_n+"**2-2*";
    m_NLL_string += corr_name;
    m_NLL_string += "*"+res1_n+"*"+res2_n+")";

//     m_NLL_string  = " + (0.5/(1-";
//     m_NLL_string += corr_name;
//     m_NLL_string += "**2))*(res1**2+res2**2-2*";
//     m_NLL_string += corr_name;
//     m_NLL_string += "*res1*res2)";


/* TOO COMPLICATED DOES NOT COMPILE
    // Build the NLL term

    TString constr1_name=constr1->GetName();

    TString term1_string="((";
    term1_string+=constr1_name;
    term1_string+="-";
    term1_string+=constr1_name;
    term1_string+="_gaussian_constr_mean )/";
    term1_string+=constr1_name;
    term1_string+="_gaussian_constr_sigma)";


    TString constr2_name=constr2->GetName();

    TString term2_string="((";
    term2_string+=constr2_name;
    term2_string+="-";
    term2_string+=constr2_name;
    term2_string+="_gaussian_constr_mean )/";
    term2_string+=constr2_name;
    term2_string+="_gaussian_constr_sigma)";

    TString rho2="("+corr_name+"/("+
                 constr1_name+"_gaussian_constr_sigma*"+
                 constr2_name+"_gaussian_constr_sigma*"+
                 "))";

    m_NLL_string=" + 0.5/(1-"+rho2+"**2)*("+
                 term1_string+"**2+"+
                 term2_string+"**2-2*"+rho2+
                 "*"+term1_string+"*"+term2_string+")";

TOO COMPLICATED DOES NOT COMPILE */


    }

/*----------------------------------------------------------------------------*/

void ConstrBlock2::print(const char* options){
    std::cout << "ConstrBlock " << GetName() << ":\n"
              << "\nParameters:\n";
    RooRealVar* par;
    for (int i=0;i<m_parameters->getSize();++i){
        par=(RooRealVar*) &((*m_parameters)[i]);
        std::cout << " - " << par->ClassName() << " object " << par->GetName()
                  << " " << par->getVal() << std::endl;
        }
    }

/*----------------------------------------------------------------------------*/

void ConstrBlock2::restore(){
    Constraint* constr;
    for (int i=0;i<m_constr_list->getSize();++i){
        constr=(Constraint*) &((*m_constr_list)[i]);
        constr->restore();
        }
    }

/*----------------------------------------------------------------------------*/

void ConstrBlock2::setFixed(bool fix){
    Constraint* constr;
    for (int i=0;i<m_constr_list->getSize();++i){
        constr=(Constraint*) &((*m_constr_list)[i]);
        constr->setConstant(fix);
        }
    }

/*----------------------------------------------------------------------------*/

/**
We realy heavily on the orders of the input parameters:
Correlation-constr1-mean-sigma-constr2-mean-sigma
**/

void ConstrBlock2::fluctuate(){

    double m[2][2];

    // Fill the matrix
    double sigma[2];

    Constraint* constr1=(Constraint*)&(*m_constr_list)[0];
    Constraint* constr2=(Constraint*)&(*m_constr_list)[1];


    sigma[0]=((RooRealVar*)&((constr1->getNLLterms())[2]))->getVal();
    sigma[1]=((RooRealVar*)&((constr2->getNLLterms())[2]))->getVal();;

    // diagonal terms
    for (int i=0;i<2;++i)
        m[i][i]=sigma[i]*sigma[i];

    // non diagonal terms
    m[1][0]=m[0][1]=sigma[0]*sigma[1]*m_corr->getVal();

    // Correlated generation
    // Create the as:
    float a11=sqrt(m[0][0]);

    float a21=m[1][0]/a11;
    float a22=sqrt(m[1][1]-a21*a21);

    if (is_verbose())
    std::cout << "[ConstrBlock2::fluctuate] "
              << "The a_ij:\n"
              << " - a11 = " << a11 << std::endl
              << " - a22 = " << a22 << std::endl
              << " - a21 = " << a21 << std::endl;

    // Shoot 3 random numbers centered in the mean of the constraint
    float u[2];
    if (is_verbose())
        std::cout << "[ConstrBlock2::fluctuate] "
                  << " The us:\n";
    for(int i=0;i<2;i++){
        u[i]=NLLPenalty::random_generator.Gaus(0,1);
        if (is_verbose())
            std::cout << " - u[i]" << u[i]<< std::endl;
        }

    // and finally the 2 numbers!
    double y[2];
    y[0]=a11*u[0]+constr1->getOriginalValue();
    y[1]=a21*u[0]+a22*u[1]+constr2->getOriginalValue();

    if (is_verbose())
        std::cout << "[ConstrBlock2::fluctuate] \n"
                  << "  - constr1_val : "
                  << constr1->getOriginalValue() << " --- " << y[0] << std::endl
                  << "  - constr2_val : "
                  << constr2->getOriginalValue() << " --- " << y[1] << std::endl;


    if (y[0] < constr1->getMin() or
        y[1] < constr2->getMin() or
        y[0] > constr1->getMax() or
        y[1] > constr2->getMax()){
        std::cerr << "[ConstrBlock2::fluctuate]"
                  << " Correlated variable outside limits... Regenerating.\n";
        fluctuate();
        }
    else{
        constr1->setVal(y[0]);
        constr2->setVal(y[1]);
        }

    }

/*----------------------------------------------------------------------------*/

/**
If only one param is for signal, return the terms of the other, as uncorrelated.
If both are for signal, return "0".
**/

TString ConstrBlock2::getBkgNLLstring(){

    bool c1_is_sig=TString((*m_constr_list)[0].GetName()).Contains(SIG_KEYWORD);
    bool c2_is_sig=TString((*m_constr_list)[1].GetName()).Contains(SIG_KEYWORD);

    // Both are for signal:
    if (c1_is_sig and c2_is_sig)
        return TString(" + 0");

    // param 0 is for signal
    if ((not c1_is_sig) and c2_is_sig)
        return ((Constraint*)&((*m_constr_list)[0]))->getNLLstring();

    // param 1 is for signal
    if (c1_is_sig and (not c2_is_sig))
        return ((Constraint*)&((*m_constr_list)[1]))->getNLLstring();

    // both are for bkg
    return getNLLstring();

    }

/*----------------------------------------------------------------------------*/

/**
Three cases: sig-sig, sig-bkg and bkg-bkg. Return accordingly.
**/

RooArgList ConstrBlock2::getBkgNLLterms(){

    bool c1_is_sig=TString((*m_constr_list)[0].GetName()).Contains(SIG_KEYWORD);
    bool c2_is_sig=TString((*m_constr_list)[1].GetName()).Contains(SIG_KEYWORD);

    // Both are for signal:
    if (c1_is_sig and c2_is_sig)
        return RooArgList();

    // param 0 is for signal
    if ((not c1_is_sig) and c2_is_sig)
        return ((Constraint*)&((*m_constr_list)[0]))->getNLLterms();

    // param 1 is for signal
    if (c1_is_sig and ( not c2_is_sig))
        return ((Constraint*)&((*m_constr_list)[1]))->getNLLterms();

    // both are for bkg
    return getNLLterms();

    }

/*----------------------------------------------------------------------------*/
ConstrBlock2::~ConstrBlock2(){
    if (m_parameters!=NULL)
        delete m_parameters;
    if (m_constr_list!=NULL)
        delete m_constr_list;
    if (m_corr!=NULL)
        delete m_corr;
    }
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
