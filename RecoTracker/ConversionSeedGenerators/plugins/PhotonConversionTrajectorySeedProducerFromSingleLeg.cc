#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhotonConversionTrajectorySeedProducerFromSingleLegAlgo.h"
//#include "UserUtilities/TimingPerformance/interface/TimeReport.h"

class dso_hidden PhotonConversionTrajectorySeedProducerFromSingleLeg final : public edm::stream::EDProducer<> {
public:
  PhotonConversionTrajectorySeedProducerFromSingleLeg(const edm::ParameterSet& );
  ~PhotonConversionTrajectorySeedProducerFromSingleLeg(){delete _theFinder;}
  PhotonConversionTrajectorySeedProducerFromSingleLeg(const PhotonConversionTrajectorySeedProducerFromSingleLeg&)=delete;
  PhotonConversionTrajectorySeedProducerFromSingleLeg& operator=(const PhotonConversionTrajectorySeedProducerFromSingleLeg&)=delete;
  void produce(edm::Event& , const edm::EventSetup& ) override;

private:
  std::string _newSeedCandidates;
  PhotonConversionTrajectorySeedProducerFromSingleLegAlgo *_theFinder;
};


PhotonConversionTrajectorySeedProducerFromSingleLeg::
PhotonConversionTrajectorySeedProducerFromSingleLeg(const edm::ParameterSet& conf)
  : _newSeedCandidates(conf.getParameter<std::string>( "newSeedCandidates"))
{
  _theFinder = new PhotonConversionTrajectorySeedProducerFromSingleLegAlgo(conf,
  	consumesCollector());
  produces<TrajectorySeedCollection>(_newSeedCandidates);
}


void PhotonConversionTrajectorySeedProducerFromSingleLeg::produce(edm::Event& ev, const edm::EventSetup& es)
{


  std::auto_ptr<TrajectorySeedCollection> result( new TrajectorySeedCollection() );  

  _theFinder->find(ev,es,*result);
  result->shrink_to_fit();
  ev.put(result, _newSeedCandidates);  


}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonConversionTrajectorySeedProducerFromSingleLeg);
