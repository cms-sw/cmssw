#ifndef Base_PileUp_h
#define Base_PileUp_h

#include <string>
#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/EventID.h"


class TFile;
class TH1F;

namespace CLHEP {
  class RandPoissonQ;
  class RandPoisson;
}



namespace edm {
  class PileUp {
  public:
    typedef VectorInputSource::EventPrincipalVector EventPrincipalVector;
    explicit PileUp(ParameterSet const& pset, int const minb, int const maxb, double averageNumber, TH1F* const histo, const bool playback);
    ~PileUp();

    void readPileUp(std::vector<EventPrincipalVector> & result,std::vector<std::vector<edm::EventID> > &ids);

    double averageNumber() const {return averageNumber_;}
    bool poisson() const {return poisson_;}
    bool doPileup() {return none_ ? false :  averageNumber_>0.;}
    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
      input_->dropUnwantedBranches(wantedBranches);
    }
    void endJob () {
      input_->doEndJob();
    }

  private:
    std::string const type_;
    int const minBunch_;
    int const maxBunch_;
    double const averageNumber_;
    int const intAverage_;
    TH1F* const histo_;
    bool const histoDistribution_;
    bool const probFunctionDistribution_;
    bool const poisson_;
    bool const fixed_;
    bool const none_;
    bool manage_OOT_;
    bool poisson_OOT_;
    bool fixed_OOT_;
    int  intFixed_OOT_;

    VectorInputSource * const input_;
    CLHEP::RandPoissonQ *poissonDistribution_;
    CLHEP::RandPoisson  *poissonDistr_OOT_;


    TH1F *h1f;
    TH1F *hprobFunction;
    TFile *probFileHisto;
    
    //playback info
    bool playback_;

    // sequential reading
    bool sequential_;
    
    // read the seed for the histo and probability function cases
    int seed_;
  };
}

#endif
