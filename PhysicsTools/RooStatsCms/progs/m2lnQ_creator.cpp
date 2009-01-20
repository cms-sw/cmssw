/**
A very genral program that builds the two distributions of -2lnQ in the signal 
plus background and background only hypotheses for a combined model.
It takes as input the card name, the model name and the number of toys.
It dumps a png image of the plot and a rootfile with the LimitResults object.
**/
#include <iostream>

#include "TString.h"

#include "RooGlobalFunc.h"
#include "RooRandom.h"

#include "LimitCalculator.h"
#include "RscCombinedModel.h"
#include "RscConstrArrayFiller.h"

int main(int argc, char** argv){

    // Parse input

    if (argc == 2 and TString(argv[1]).Contains("-h"))
        std::cout << "\nm2lnQ_creator:\n"
                  << "A very genral program that builds the two distributions of -2lnQ in the signal\n" 
                  << "plus background and background only hypotheses for a combined model.\n"
                  << "It takes as input the card name, the model name and the number of toys.\n"
                  << "It dumps a png image of the plot and a rootfile with the LimitResults object.\n";

    if (argc != 4){
        std::cerr << "\nUsage:\n " << argv[0] << " card model ntoys\n\n";
        if (not TString(argv[1]).Contains("-h"))
            std::cerr << "\n  Type " << argv[0] << " -h for more details.\n\n";
        return 1;
        }

    TString card_name(argv[1]);
    TString model_name(argv[2]);
    int n_toys = atoi (argv[3]);
    TString n_toys_s (argv[3]);

    RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;

    // Get the model from the card
    RscAbsPdfBuilder::setDataCard(card_name.Data());
    RscCombinedModel model(model_name.Data());

    // Get the pdf of the model(sb and bonly components)
    RooAbsPdf* sb_model=model.getPdf();
    RooAbsPdf* b_model=model.getBkgPdf();



    // The object that collects all the constraints and the blocks of correlated constraints
    ConstrBlockArray constr_array("ConstrArray", "The array of constraints");
    constr_array.setVerbosity(true);

    // Facility to read the card and fill the constraints array properly
    RscConstrArrayFiller filler("ConstrFiller", "The array Filler", model.getConstraints());
    filler.fill(&constr_array,model_name.Data());

    RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

    RooArgList l (model.getVars());
    l.Print("v");

    // The -2lnQ calculator
    LimitCalculator calc("m2lnQmethod","-2lnQ method",sb_model,b_model,&l,&constr_array);

    // Launch the calculation
    bool fluctuate = true;
    //bool fluctuate = false;
    RooRandom::randomGenerator()->SetSeed(1);
    RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;
    LimitResults* res = calc.calculate (n_toys,fluctuate);
    RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

    // Get theplot and impose to observe the median of the SB distribution
    LimitPlot* p = res->getPlot("","",300);

    // Now dump the stuff on disk, creating some meaningful names
    TString image_name((model_name+"_m2lnQ_distrinutions_"+n_toys_s+".png").Data());
    p->draw();
    p->dumpToImage(image_name.Data());

    TString file_name((model_name+"_m2lnQ_distrinutions_"+n_toys_s+".root").Data());
    TFile ofile(file_name.Data(),"RECREATE");
    res->Write((model_name+"_m2lnq_distributions_"+n_toys_s).Data());
    ofile.Close();

    delete p;
    delete res;
    }
