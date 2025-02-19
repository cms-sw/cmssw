class RooAbsData;
class RooAbsCollection;
namespace RooStats { class ModelConfig; }

namespace asimovutils {
    /// Generate asimov dataset from nominal value of nuisance parameters
    RooAbsData * asimovDatasetNominal(RooStats::ModelConfig *mc, double poiValue=0.0, int verbose=0) ;
    /// Generate asimov dataset from best fit value of nuisance parameters, and fill in snapshot of corresponding global observables
    RooAbsData * asimovDatasetWithFit(RooStats::ModelConfig *mc, RooAbsData &realdata, RooAbsCollection &snapshot, double poiValue=0.0, int verbose=0) ;
}

