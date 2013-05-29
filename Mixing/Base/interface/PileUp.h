#ifndef Base_PileUp_h
#define Base_PileUp_h

#include <string>
#include <vector>
#include <boost/bind.hpp>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"

#include "TRandom.h"
#include "TFile.h"
#include "TH1F.h"

class TFile;
class TH1F;

namespace CLHEP {
  class RandPoissonQ;
  class RandPoisson;
}



namespace edm {
  class PileUp {
  public:
    explicit PileUp(ParameterSet const& pset, double averageNumber, TH1F* const histo, const bool playback);
    ~PileUp();

    template<typename T>
      void readPileUp(std::vector<edm::EventID> &ids, T eventOperator, const int NumPU );

    template<typename T>
      void playPileUp(const std::vector<edm::EventID> &ids, T eventOperator);

    double averageNumber() const {return averageNumber_;}
    bool poisson() const {return poisson_;}
    bool doPileUp() {return none_ ? false :  averageNumber_>0.;}
    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
      input_->dropUnwantedBranches(wantedBranches);
    }
    void endJob () {
      input_->doEndJob();
    }

    void CalculatePileup(int MinBunch, int MaxBunch, std::vector<int>& PileupSelection, std::vector<float>& TrueNumInteractions);

    //template<typename T>
    // void recordEventForPlayback(EventPrincipal const& eventPrincipal,
	//			  std::vector<edm::EventID> &ids, T& eventOperator);

  private:
    std::string const type_;
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



  template<typename T>
  class RecordEventID
  {
  private:
    std::vector<edm::EventID>& ids_;
    T& eventOperator_;
    int eventCount ;
  public:
    RecordEventID(std::vector<edm::EventID>& ids, T& eventOperator)
      : ids_(ids), eventOperator_(eventOperator), eventCount( 0 ) {}
    void operator()(EventPrincipal const& eventPrincipal) {
      ids_.push_back(eventPrincipal.id());
      eventOperator_(eventPrincipal, ++eventCount);
    }
  };


  /*! Generates events from a VectorInputSource.
   *  This function decides which method of VectorInputSource 
   *  to call: sequential, random, or pre-specified.
   *  The ids are either ids to read or ids to store while reading.
   *  eventOperator has a type that matches the eventOperator in
   *  VectorInputSource::loopRandom.
   */
  template<typename T>
  void
    PileUp::readPileUp(std::vector<edm::EventID> &ids, T eventOperator, const int pileEventCnt) {

    // One reason PileUp is responsible for recording event IDs is
    // that it is the one that knows how many events will be read.
    ids.reserve(pileEventCnt);
    RecordEventID<T> recorder(ids,eventOperator);
    if (sequential_) {
      // boost::bind creates a functor from recordEventForPlayback
      // so that recordEventForPlayback can insert itself before
      // the original eventOperator.

      input_->loopSequential(pileEventCnt, recorder);
      //boost::bind(&PileUp::recordEventForPlayback<T>,
      //                    boost::ref(*this), _1, boost::ref(ids),
      //                             boost::ref(eventOperator))
      //  );
        
    } else  {
      input_->loopRandom(pileEventCnt, recorder);
      //               boost::bind(&PileUp::recordEventForPlayback<T>,
      //                             boost::ref(*this), _1, boost::ref(ids),
      //                             boost::ref(eventOperator))
      //                 );
    }
  }



  template<typename T>
  void
    PileUp::playPileUp(const std::vector<edm::EventID> &ids, T eventOperator) {
    //TrueNumInteractions.push_back( ids.size() ) ;
    input_->loopSpecified(ids,eventOperator);
  }



  /*! Record the event ID and pass the call on to the eventOperator.
   */
  /*  template<typename T>
    void recordEventForPlayback(EventPrincipal const& eventPrincipal,
                             std::vector<edm::EventID> &ids, T& eventOperator)
    {
      ids.push_back(eventPrincipal.id());
      eventOperator(eventPrincipal);
    }
  */
}


#endif
