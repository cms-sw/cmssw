#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalRecHitProducer_HH
/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: $
 *  $Date: $
 *  $Revision: $
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"

// forward declaration
class EcalRecHitProducer : public edm::EDProducer {

  public:
    explicit EcalRecHitProducer(const edm::ParameterSet& ps);
    ~EcalRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    std::string uncalibRecHitProducer_; // name of module/plugin/producer making uncalib rechits
    std::string uncalibRecHitCollection_; // secondary name given to collection of uncalib rechits
    std::string rechitCollection_; // secondary name to be given to collection of hits

    EcalRecHitAbsAlgo* algo_;

    int nMaxPrintout_; // max # of printouts
    int nEvt_; // internal counter of events

    bool counterExceeded() const { return ( (nEvt_>nMaxPrintout_) || (nMaxPrintout_<0) ) ; }
};
#endif
