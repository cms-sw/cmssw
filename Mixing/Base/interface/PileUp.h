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
      void readPileUp(std::vector<edm::EventID> &ids, T eventOperator,
		      std::vector<float>& TrueBXCount, const int bx );

    template<typename T>
      void playPileUp(const std::vector<edm::EventID> &ids, T eventOperator,
		      std::vector<float>& TrueBXCount);

    double averageNumber() const {return averageNumber_;}
    bool poisson() const {return poisson_;}
    bool doPileUp() {return none_ ? false :  averageNumber_>0.;}
    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
      input_->dropUnwantedBranches(wantedBranches);
    }
    void endJob () {
      input_->doEndJob();
    }

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
    PileUp::readPileUp(std::vector<edm::EventID> &ids, T eventOperator,
		       std::vector<float>& TrueNumInteractions, const int bx) {

    // if we are managing the distribution of out-of-time pileup separately, select the distribution for bunch
    // crossing zero first, save it for later.

    int nzero_crossing = -1;
    double Fnzero_crossing = -1;

    // This is a number of events per source per bunch crossing.
    size_t pileEventCnt=0;

    if(manage_OOT_) {
      if (none_){
	nzero_crossing = 0;
      }else if (poisson_){
	nzero_crossing =  poissonDistribution_->fire() ;
      }else if (fixed_){
	nzero_crossing =  intAverage_ ;
      }else if (histoDistribution_ || probFunctionDistribution_){
	double d = histo_->GetRandom();
	//n = (int) floor(d + 0.5);  // incorrect for bins with integer edges
	Fnzero_crossing =  d;
      }

      if(bx==0 && !poisson_OOT_) { 
	pileEventCnt = nzero_crossing ;
	TrueNumInteractions.push_back( nzero_crossing );
      }
      else{
	if(poisson_OOT_) {
	  pileEventCnt = poissonDistr_OOT_->fire(Fnzero_crossing) ;
	  TrueNumInteractions.push_back( Fnzero_crossing );
	}
	else {
	  pileEventCnt = intFixed_OOT_ ;
	  TrueNumInteractions.push_back( intFixed_OOT_ );
	}  
      }
    } 
    else {
      if (none_){
        pileEventCnt = 0;
	TrueNumInteractions.push_back( 0. );
      }else if (poisson_){
        pileEventCnt = poissonDistribution_->fire();
	TrueNumInteractions.push_back( averageNumber_ );
      }else if (fixed_){
        pileEventCnt = intAverage_;
	TrueNumInteractions.push_back( intAverage_ );
      }else if (histoDistribution_ || probFunctionDistribution_){
        double d = histo_->GetRandom();
        pileEventCnt = int(d);
	TrueNumInteractions.push_back( d );
      }
    }

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
    PileUp::playPileUp(const std::vector<edm::EventID> &ids, T eventOperator,
		       std::vector<float>& TrueNumInteractions) {
    TrueNumInteractions.push_back( ids.size() ) ;
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
