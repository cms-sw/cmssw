#ifndef BMixingModule_h
#define BMixingModule_h

/** \class BMixingModule
 *
 * BMixingModule is the EDProducer subclass 
 * which fills the BCrossingFrame object
 * It is the baseclass for all modules mnixing events 
 *
 * \author Ursula Berthon, LLR Palaiseau
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
#include "FWCore/Framework/interface/SecondaryInputSource.h"
#include "Mixing/Base/interface/PUGenerator.h"


namespace edm
{
  class BMixingModule : public edm::EDProducer
    {
    public:

      /** standard constructor*/
      explicit BMixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~BMixingModule();

      /**Cumulates the pileup events onto this event*/
      virtual void produce(edm::Event& e1, const edm::EventSetup& c);


    private:
      double average_;
      PUGenerator *generator_;

      virtual void createnewEDProduct() {std::cout<<"BMixingModule::createnewEDProduct must be overwritten!"<<std::endl;}
      virtual void getEvents(const unsigned int nrEvents);
      void merge(const int bcr, const std::vector<Event *> vec);
      virtual void addSignals(edm::Event &e) {;}
      virtual void addPileups(const int bcr, edm::Event*) {;}
      virtual void put(edm::Event &e
) {;}

      boost::shared_ptr<SecondaryInputSource> makeSecInput(ParameterSet const& ps);
    protected:
      int minbunch_;
      int maxbunch_;
      int bunchSpace_;

      boost::shared_ptr<SecondaryInputSource> secInput_;
      std::vector<Event *> eventVector_;

    };
}//edm

#endif
