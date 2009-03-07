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
  class SecondaryProducer: public edm::EDProducer {
  public:

    /** standard constructor*/
    explicit SecondaryProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~SecondaryProducer();

    /**Cumulates the pileup events onto this event*/
    virtual void produce(edm::Event& e1, const edm::EventSetup& c);


  private:

    virtual void put(edm::Event &) {}

    virtual void endJob() {secInput_->doEndJob();}

    boost::shared_ptr<VectorInputSource> makeSecInput(ParameterSet const& ps);

    boost::shared_ptr<VectorInputSource> secInput_;
  };
}//edm

#endif
