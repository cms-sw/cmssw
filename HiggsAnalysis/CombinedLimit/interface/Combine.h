#ifndef HiggsAnalysis_CombinedLimit_Combine_h
#define HiggsAnalysis_CombinedLimit_Combine_h
#include <TString.h>
#include <boost/program_options.hpp>

class TDirectory;
class TTree;
class LimitAlgo;
class RooWorkspace;
class RooAbsData;

extern Float_t t_cpu_, t_real_;
//RooWorkspace *writeToysHere = 0;
extern TDirectory *writeToysHere;
extern TDirectory *readToysFromHere;
extern LimitAlgo * algo, * hintAlgo ;
extern int verbose;
extern bool withSystematics;
extern float cl;

class Combine {
    public:
        Combine() ;

        const boost::program_options::options_description & options() const { return options_; }    
        void applyOptions(const boost::program_options::variables_map &vm) ;

        void run(TString hlfFile, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys);

    private:
        bool mklimit(RooWorkspace *w, RooAbsData &data, double &limit) ;

        boost::program_options::options_description options_;

        float rMin_, rMax_;
        bool compiledExpr_;
};


#endif
