#include <cstdlib>
#include <string>
#include <iostream>
#include <numeric>
#include <functional>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>

#include <TString.h>

#include <Alignment/OfflineValidation/interface/CompareAlignments.h>
#include <Alignment/OfflineValidation/interface/PlotAlignmentValidation.h>
#include <Alignment/OfflineValidation/interface/TkAlStyle.h>

using namespace std;
using namespace AllInOneConfig;

namespace pt = boost::property_tree;

std::string getVecTokenized(pt::ptree& tree, const std::string& name, const std::string& token){
    std::string s;

    for(std::pair<std::string, pt::ptree> childTree : tree.get_child(name)){
        s+= childTree.second.get_value<std::string>() + token;
    }

    return s.substr(0, s.size()-token.size());
}

int merge (int argc, char * argv[]){
    // parse the command line
    Options options;
    options.helper(argc, argv);
    options.parser(argc, argv);

    //Read in AllInOne json config
    pt::ptree main_tree;
    pt::read_json(options.config, main_tree);

    pt::ptree alignments = main_tree.get_child("alignments");
    pt::ptree validation = main_tree.get_child("validation");

    //Read all configure variables and set default for missing keys
    std::string methods = validation.get_child_optional("methods") ? getVecTokenized(validation, "methods", ",") : "median,rmsNorm";
    std::string curves = validation.get_child_optional("curves") ? getVecTokenized(validation, "curves", ",") : "plain";

    int minimum = validation.get_child_optional("minimum") ? validation.get<int>("minimum") : 15;

    bool useFit = validation.get_child_optional("usefit") ? validation.get<bool>("usefit") : false;
    bool bigText = validation.get_child_optional("bigtext") ? validation.get<bool>("bigtext") : false;

    TkAlStyle::legendheader = validation.get_child_optional("legendheader") ? validation.get<std::string>("legendheader") : "";
    TkAlStyle::legendoptions = validation.get_child_optional("legendoptions") ? getVecTokenized(validation, "legendoptions", " ") : "mean rms";
    TkAlStyle::set(INTERNAL, NONE, "", validation.get<std::string>("customrighttitle"));

    std::vector<int> moduleids;

    if(validation.get_child_optional("moduleid")){
        for(std::pair<std::string, pt::ptree> childTree : validation.get_child("moduleid")){
            moduleids.push_back(childTree.second.get_value<int>());
        }
    } 

    //Set configuration string for CompareAlignments class
    TString filesAndLabels;

    for(std::pair<std::string, pt::ptree> childTree : alignments){
        filesAndLabels += childTree.second.get<std::string>("file") + "/DMR.root=" + childTree.second.get<std::string>("title") + "|" + childTree.second.get<std::string>("color") + "|" + childTree.second.get<std::string>("style") + " , ";
    }

    filesAndLabels.Remove(filesAndLabels.Length()-3);

    //Do file comparisons
    CompareAlignments comparer(main_tree.get<std::string>("output"));
    comparer.doComparison(filesAndLabels);

    //Create plots
    gStyle->SetTitleH(0.07);
    gStyle->SetTitleW(1.00);
    gStyle->SetTitleFont(132);

    PlotAlignmentValidation plotter(bigText);

    for(std::pair<std::string, pt::ptree> childTree : alignments){
        plotter.loadFileList((childTree.second.get<std::string>("file") + "/DMR.root").c_str(), childTree.second.get<std::string>("title"), childTree.second.get<int>("color"), childTree.second.get<int>("style"));
    }

    plotter.setOutputDir(main_tree.get<std::string>("output"));
    plotter.useFitForDMRplots(useFit);
    plotter.setTreeBaseDir("TrackHitFilter");
    plotter.plotDMR(methods, minimum, curves);
    plotter.plotSurfaceShapes("coarse");
    plotter.plotChi2((main_tree.get<std::string>("output") + "/" + "result.root").c_str());

    for(const int& moduleid : moduleids) {
        plotter.residual_by_moduleID(moduleid);
    }

    return EXIT_SUCCESS; 
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main (int argc, char * argv[])
{
    return exceptions<merge>(argc, argv);
}
#endif
