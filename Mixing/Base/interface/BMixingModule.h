#ifndef BMixingModule_h
#define BMixingModule_h

/** \class BMixingModule
 *
 * BMixingModule is the EDProducer subclass 
 * which fills the CrossingFrame object
 * It is the baseclass for all modules mixing events
 *
 * \author Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
 *
 * \version   1st Version June 2005
 * \version   2nd Version Sep 2005

 *
 ************************************************************/

#include <vector>
#include <memory>

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/MixingRcd.h"


namespace edm {
  class BMixingModule : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
    public:
      /** standard constructor*/
      explicit BMixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~BMixingModule();

      /**Cumulates the pileup events onto this event*/
      virtual void produce(edm::Event& e1, const edm::EventSetup& c) override;

      virtual void initializeEvent(const edm::Event& event, const edm::EventSetup& setup) {}

      // edm::Event is non-const because digitizers put their products into the Event.
      virtual void finalizeEvent(edm::Event& event, const edm::EventSetup& setup) {}

      virtual void beginRun(const edm::Run& r, const edm::EventSetup& setup) override;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& setup) override;

      virtual void endRun(const edm::Run& r, const edm::EventSetup& setup) override;
      virtual void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& setup) override;

      // to be overloaded by dependent class
      virtual void reload(const edm::EventSetup & setup){};

      // Should 'averageNumber' return 0 or 1 if there is no mixing? It is the average number of
      // *crossings*, including the hard scatter, or the average number of overlapping events?
      // We have guessed 'overlapping events'.
      double averageNumber() const {return inputSources_[0] ? inputSources_[0]->averageNumber() : 0.0; }
      // Should 'poisson' return 0 or 1 if there is no mixing? See also averageNumber above.
      bool poisson() const {return inputSources_[0] ? inputSources_[0]->poisson() : 0.0 ;}

      virtual void createnewEDProduct();
      virtual void checkSignal(const edm::Event &e);
      virtual void addSignals(const edm::Event &e,const edm::EventSetup& c) {}
      virtual void addPileups(const int bcr, EventPrincipal *ep, unsigned int eventId,unsigned int worker, const edm::EventSetup& c) {}
      virtual void setBcrOffset ();
      virtual void setSourceOffset (const unsigned int s);
      virtual void put(edm::Event &e,const edm::EventSetup& c) {}
      virtual void doPileUp(edm::Event &e, const edm::EventSetup& c);

  protected:
      void setupPileUpEvent(const edm::EventSetup& setup);
      void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);
      virtual void beginJob() override;
      virtual void endJob() override;
      //      std::string type_;
      int bunchSpace_;
      int vertexOffset_;
      bool checktof_;
      int minBunch_;
      int maxBunch_;
      bool const mixProdStep1_;	       	
      bool const mixProdStep2_;
      	
      bool readDB_;
      bool playback_;
      const static unsigned int maxNbSources_;
      std::vector<std::string> sourceNames_;
      bool doit_[4];//FIXME
      std::vector< float > TrueNumInteractions_;

      unsigned int eventId_;

      // input, cosmics, beamhalo_plus, beamhalo_minus
      std::vector<std::shared_ptr<PileUp> > inputSources_;

      void update(edm::EventSetup const&);
      edm::ESWatcher<MixingRcd> parameterWatcher_;
  };

}//edm

#endif
