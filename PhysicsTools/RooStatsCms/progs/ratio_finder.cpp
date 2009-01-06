/**
A very general program that scans a variable in the model inside a max and a 
min value so to obtain a desired CLs value. The class used is RatioFinder.
At the end of the procedure the RatioFinderResults object is written to a root 
file and a png image of the evolution of the scan is dumped.
Useful to answer questions like:
 - I want to do a green/yellow plot "a la Tevatron" where I exclude a certain R factor on the SM cross section
 - I want to know how much lumi I need for a CLs of X percent
**/
#include <iostream>
#include <sstream>

#include "TString.h"
#include "TFile.h"

#include "RooGlobalFunc.h"

#include "RatioFinder.h"
#include "RscCombinedModel.h"
#include "RscConstrArrayFiller.h"


int main(int argc, char** argv){

    // Parse input
    if (argc == 2 and TString(argv[1]).Contains("-h"))
        std::cout << "A very general program that scans a variable in the model inside a max and a\n"
                  << "min value so to obtain a desired CLs value. The class used is RatioFinder.\n"
                  << "At the end of the procedure the RatioFinderResults object is written to a root\n"
                  << "file and a png image of the evolution of the scan is dumped.\n"
                  << "Useful to answer questions like:\n"
                  << " - I want to do a green/yellow plot \"a la Tevatron\" where "
                  <<   "I exclude a certain R factor on the SM cross section\n"
                  << " - I want to know how much lumi I need for a CLs of X percent\n";

    if (argc != 9){
        std::cerr << "Usage:\n " << argv[0]
                  << " card model ntoys n_sigmas minR maxR deltaR varname\n";
        if (not TString(argv[1]).Contains("-h"))
            std::cerr << "\n  Type " << argv[0] << " -h for more details.\n\n";
        return 1;
        }

    TString card_name(argv[1]);
    TString model_name(argv[2]);
    int n_toys = atoi (argv[3]);
    TString n_toys_s (argv[3]);
    double n_sigma= atof (argv[4]);
    double minR= atof (argv[5]);
    double maxR= atof (argv[6]);
    double deltaR= atof (argv[7]);
    TString varname (argv[8]);

    RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;

    // Get the model from the card
    RscAbsPdfBuilder::setDataCard(card_name.Data());
    RscCombinedModel model(model_name.Data());

    // Get the pdf of the model(sb and bonly components)
    RooAbsPdf* sb_model=model.getPdf();
    RooAbsPdf* b_model=model.getBkgPdf();

    // The object that collects all the constraints and the blocks of correlated constraints
    ConstrBlockArray constr_array("ConstrArray", "The array of constraints");
    constr_array.setVerbosity(false);

    // Facility to read the card and fill the constraints array properly
    RscConstrArrayFiller filler("ConstrFiller", "The array Filler", model.getConstraints());
    filler.fill(&constr_array);

    RooArgList l (model.getVars());
    //l.Print("v");

    RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

    // Build the finder object

    TString finderanme("Finder_deltaR");
    finderanme+=deltaR;
    finderanme+="_";
    finderanme+=model_name;

    finderanme.ReplaceAll(" ","");

    RatioFinder finder(finderanme.Data(),"Finder HWW",sb_model,b_model,varname.Data(),l,&constr_array);

    RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;
    double desired_CL=.05;
    RatioFinderResults* res = finder.findRatio(n_toys,minR,maxR,n_sigma,desired_CL,deltaR,true);
    RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

    res->print();
    double inter_R = res->getInterpolatedRatio("Linear");

    std::cout << "\nInterpolated Ratio = " << inter_R << std::endl<< std::endl;

    RatioFinderPlot* p=res->getPlot();

    p->draw();

    // Save to file

    std::stringstream s;
    s << "ratioscan_" << model_name.Data() << "_ntoys" << n_toys << "_nsigma" << n_sigma << "_" << minR << "-" << maxR << "_" << varname.Data() << ".png";
    p->dumpToImage(s.str().c_str());

    std::stringstream ss;
    ss << "ratioscan_" << model_name.Data() << "_ntoys" << n_toys << "_nsigma" << n_sigma << "_" << minR << "-" << maxR << "_" << varname.Data() << ".root";

    TFile ofile(ss.str().c_str(),"RECREATE");
    res->Write();
    ofile.Close();

}
