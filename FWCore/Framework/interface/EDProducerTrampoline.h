#ifndef FWCore_Framework_EDProducerTrampoline_h
#define FWCore_Framework_EDProducerTrampoline_h



#include "FWCore/Framework/interface/EDProducer.h"


namespace edm {
  template<typename T, int Nmax>
  class EDProducerTrampoline : public EDProducer {
  public:

    EDProducerTrampoline() : m_objNum(seqNum()++){}


  private:
    static int & seqNum() { static int l=0; return l;}

    template<int N>
    void produceN(edm::Event& e, const edm::EventSetup& es)  __attribute__ ((noinline));

    virtual void produceChild(Event&, EventSetup const&) = 0;

    template<int N>
    void switchIt(edm::Event& e, const edm::EventSetup& es) {
      if ( m_objNum==N) produceN<N>(e,es); else switch<N+1>(e,es);
    }
    template<>
    void switchIt<Nmax>(edm::Event& e, const edm::EventSetup& es) {
      produceN<Nmax>(e,es);
    }

    void produce(edm::Event& e, const edm::EventSetup& es) final {
      switchIt<0>(e,es);
    }

  private:
    int m_objNum;

  };


  template<typename T, int Nmax>
  template<int N>
  void EDProducerTrampoline<T, Nmax>::produceN(edm::Event& e, const edm::EventSetup& es)
  { asm (""); this->produceChild(e,es);}
  

} // namespace edm
