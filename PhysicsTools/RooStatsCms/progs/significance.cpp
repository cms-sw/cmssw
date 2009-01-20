#include <iostream>

#include "TH1F.h"
#include "TString.h"
#include "TApplication.h"
#include "TCanvas.h"

#include "RooGlobalFunc.h"

#include "Minus2LnQCalculator.h"
#include "RscConstrArrayFiller.h"
#include "RscCombinedModel.h"


int main(int argc, char** argv){



if (argc<4){
    std::cout << "Usage: \n"
              << argv[0] << " card combined_model n_toys [max_sign=15]\n";
    return  1;
    }

TString card_name(argv[1]);
TString model_name(argv[2]);
int nToyMC=atoi(argv[3]);
float max_sign=15;
if (argc==5)
    max_sign=atof(argv[4]);


RscAbsPdfBuilder::setDataCard(card_name.Data());
RscCombinedModel combi(model_name.Data());


// Get the pdfs
RooAbsPdf* sb_model=combi.getPdf();
RooAbsPdf* b_model=combi.getBkgPdf();

// Get the vars
RooArgList vars(combi.getVars());

// Get the penalties: different for sig and bkg
ConstrBlockArray constr_array("ConstrArray", "The array of constraints");
RscConstrArrayFiller filler("ConstrFiller", "The array Filler", combi.getConstraints());
filler.fill(&constr_array,model_name.Data());
TString NLL_penalties=constr_array.getNLLstring();
RooArgList NLL_terms=constr_array.getNLLterms();
TString NLL_bkg_penalties=constr_array.getBkgNLLstring();
RooArgList NLL_bkg_terms=constr_array.getBkgNLLterms();

/*
Calculate significance with the profiling method:
    - Add to Likelihood penalities according to systematics
    - One fit
*/
TH1F h_sign_prof("prof_sign","Significance: profiling method",1000,0,max_sign);
// generate the dataset you expect
RooAbsData* data_exp = (RooAbsData*)(sb_model->generate(vars,(int)sb_model->expectedEvents(vars)));
Minus2LnQCalculator sqrt_m2lnQ_constr_calc(*sb_model, *b_model, NLL_penalties, NLL_terms, NLL_bkg_penalties, NLL_bkg_terms, *data_exp);
double significance_prof = sqrt_m2lnQ_constr_calc.getSqrtValue(true);
h_sign_prof.Fill(significance_prof);

/*
Calculate significance with the marginalisation method:
    - Cover phasespace with MC generation
    - No fits!
*/
TH1F h_sign_marg("marginalisation_sign","Significance: marginalisation method",1000,0,max_sign);
RooAbsData* data_m;
double significance=0;
for (int i=0; i<nToyMC; i++){

    constr_array.fluctuate();
    data_m = (RooAbsData*)sb_model->generate(vars,RooFit::Extended());
    constr_array.restore();

    Minus2LnQCalculator sqrt_m2lnQ_constr_calc(*sb_model, *b_model, *data_m);
    significance = sqrt_m2lnQ_constr_calc.getSqrtValue(false);
    h_sign_marg.Fill(significance);
    //delete data_m;
    }

// Basically here we do the Make up of the plots

TApplication theapp ("Significance calculation",&argc,argv);

TCanvas cc;
cc.Divide(2);
cc.cd(1);

h_sign_marg.DrawNormalized();

TGraph sign_prof(1);
sign_prof.SetPoint(0,significance_prof,0);
sign_prof.SetMarkerStyle(22);
sign_prof.SetMarkerSize(2);
sign_prof.SetMarkerColor(kRed);
sign_prof.Draw("PSame");

cc.cd(2);
h_sign_prof.Draw();

cout << "\n\nSignificance with profile method = " << significance_prof << endl;
cout << "\n\nSignificance without sys (mean) = " << h_sign_marg.GetMean() << endl << endl;

theapp.Run();

}
