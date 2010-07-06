#ifndef SecondaryInput_SecondaryProducer_h
#define SecondaryInput_SecondaryProducer_h

/** \class SecondaryProducer
 *
 * \author Bill Tanenbaum
 *
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Sources/interface/VectorInputSource.h"

namespace edm {
  class SecondaryProducer: public EDProducer {
  public:

    /** standard constructor*/
    explicit SecondaryProducer(ParameterSet const& pset);

    /**Default destructor*/
    virtual ~SecondaryProducer();

    /**Cumulates the pileup events onto this event*/
    virtual void produce(Event& e1, EventSetup const& c);


  private:

    virtual void put(Event &) {}

    virtual void endJob() {secInput_->doEndJob();}

    boost::shared_ptr<VectorInputSource> makeSecInput(ParameterSet const& ps);

    boost::shared_ptr<VectorInputSource> secInput_;

    bool sequential_;

    bool specified_;

    bool firstEvent_;

    bool firstLoop_;

    EventNumber_t expectedEventNumber_;
  };
}//edm

#endif
