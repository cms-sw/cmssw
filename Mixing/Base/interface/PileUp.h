#ifndef Base_PileUp_h
#define Base_PileUp_h

#include <string>
#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/EventID.h"

namespace CLHEP {
  class RandPoissonQ;
}

namespace edm {
  class PileUp {
  public:
    typedef VectorInputSource::EventPrincipalVector EventPrincipalVector;
    explicit PileUp(ParameterSet const& pset, int const minb, int const maxb, double averageNumber, const bool playback);
    ~PileUp();

    void readPileUp(std::vector<EventPrincipalVector> & result,std::vector<edm::EventID> &ids, std::vector<int> &fileNrs,std::vector<unsigned int> & nrEvents);

    double averageNumber() const {return averageNumber_;}
    bool poisson() const {return poisson_;}
    bool doPileup() {return none_ ? false :  averageNumber_>0.;}
    void setWantedBranches(std::vector<std::string> const * branches) {wantedBranches_=branches;}
    const std::vector<std::string> * getWantedBranchNames() const {return wantedBranches_;}

  private:
    std::string const type_;
    int const minBunch_;
    int const maxBunch_;
    double const averageNumber_;
    int const intAverage_;
    bool const poisson_;
    bool const fixed_;
    bool const none_;
    VectorInputSource * const input_;
    CLHEP::RandPoissonQ *poissonDistribution_;
    const std::vector<std::string> *wantedBranches_;

    //playback info
    bool playback_;
  };
}

#endif
