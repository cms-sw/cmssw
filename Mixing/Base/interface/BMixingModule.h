#ifndef BMixingModule_h
#define BMixingModule_h

/** \class BMixingModule
 *
 * BMixingModule is the EDProducer subclass 
 * which fills the CrossingFrame object
 * It is the baseclass for all modules mnixing events 
 *
 * \author Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
 *
 * \version   1st Version June 2005
 * \version   2nd Version Sep 2005

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Mixing/Base/interface/PileUp.h"


namespace edm {
  class BMixingModule : public edm::EDProducer {
    public:

      /** standard constructor*/
      explicit BMixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~BMixingModule();

      /**Cumulates the pileup events onto this event*/
      virtual void produce(edm::Event& e1, const edm::EventSetup& c);

    private:

      virtual void createnewEDProduct() {std::cout << "BMixingModule::createnewEDProduct must be overwritten!" << std::endl;}
      void merge(const int bcr, const std::vector<Event *> vec);
      virtual void addSignals(const edm::Event &e) {;}
      virtual void addPileups(const int bcr, edm::Event*) {;}
      virtual void put(edm::Event &e) {;}

    protected:
      int bunchSpace_;
      static int trackoffset;
      static int vertexoffset;

      PileUp input_;
    };
}//edm

#endif
