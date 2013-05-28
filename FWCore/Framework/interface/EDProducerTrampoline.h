#ifndef FWCore_Framework_EDProducerTrampoline_h
#define FWCore_Framework_EDProducerTrampoline_h



#include "FWCore/Framework/interface/EDProducer.h"
#include<sstream>
#include "FWCore/Framework/interface/Event.h"


namespace edm {
  namespace detailsTrampoline {   
    template<int N>
    struct SwitchIt {
      template<typename T> 
      static void op(T & p, edm::Event& e, const edm::EventSetup& es) {
      if ( p.objNum()>=N) p. template produceN<N>(e,es); else SwitchIt<N-1>::op(p,e,es);
     }
   };
   template<> 
   struct SwitchIt<0> {
      template<typename T>
      static void op(T&p, edm::Event& e, const edm::EventSetup& es) {
        p. template produceN<0>(e,es);
      }
    };
  
  }


  template<typename T, int Nmax>
  class EDProducerTrampoline : public EDProducer {
  public:
    using Self=EDProducerTrampoline<T,Nmax>;

    EDProducerTrampoline() : m_objNum(seqNum()++){}
    int objNum() const { return m_objNum;}

  private:
    static int & seqNum() { static int l=0; return l;}
public:
    template<int N>
    void produceN(edm::Event& e, const edm::EventSetup& es)  __attribute__ ((noinline));
private:
    virtual void produceChild(Event&, EventSetup const&) = 0;

    void produce(edm::Event& e, const edm::EventSetup& es) final {
      detailsTrampoline::SwitchIt<Nmax>::op(*this,e,es);
    }

  private:
    int m_objNum;

  };


  template<typename T, int Nmax>
  template<int N>
  void EDProducerTrampoline<T, Nmax>::produceN(edm::Event& e, const edm::EventSetup& es)
  { asm (""); 
    if(e.run()==0) { std::stringstream ss; ss<< "make sure this is not optimized away " << N; throw ss.str().c_str(); }
    this->produceChild(e,es);
}
  

} // namespace edm
#endif
