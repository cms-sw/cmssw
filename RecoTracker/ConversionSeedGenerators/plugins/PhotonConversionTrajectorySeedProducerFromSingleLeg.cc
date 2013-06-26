#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/ConversionSeedGenerators/interface/PhotonConversionTrajectorySeedProducerFromSingleLegAlgo.h"
//#include "UserUtilities/TimingPerformance/interface/TimeReport.h"

class PhotonConversionTrajectorySeedProducerFromSingleLeg : public edm::EDProducer {
public:
  PhotonConversionTrajectorySeedProducerFromSingleLeg(const edm::ParameterSet& );
  ~PhotonConversionTrajectorySeedProducerFromSingleLeg(){}
  void beginRun(edm::Run const&run, const edm::EventSetup& es) override;
  void endRun(edm::Run const&run, const edm::EventSetup& es) override;
  void produce(edm::Event& , const edm::EventSetup& ) override;

private:
  edm::ParameterSet _conf;
  std::string _newSeedCandidates, _xcheckSeedCandidates;
  bool _DoxcheckSeedCandidates;
  PhotonConversionTrajectorySeedProducerFromSingleLegAlgo *_theFinder;
};


PhotonConversionTrajectorySeedProducerFromSingleLeg::
PhotonConversionTrajectorySeedProducerFromSingleLeg(const edm::ParameterSet& conf)
  : _conf(conf),
    _newSeedCandidates(conf.getParameter<std::string>( "newSeedCandidates")),
    _xcheckSeedCandidates(conf.getParameter<std::string>( "xcheckSeedCandidates") ),
    _DoxcheckSeedCandidates( conf.getParameter<bool>( "DoxcheckSeedCandidates") )
{
  _theFinder = new PhotonConversionTrajectorySeedProducerFromSingleLegAlgo(conf);
  produces<TrajectorySeedCollection>(_newSeedCandidates);
  if(_DoxcheckSeedCandidates)
    produces<TrajectorySeedCollection>(_xcheckSeedCandidates);
}


void PhotonConversionTrajectorySeedProducerFromSingleLeg::
endRun(edm::Run const&run, const edm::EventSetup& es) {
  _theFinder->clear();
}

void PhotonConversionTrajectorySeedProducerFromSingleLeg::
beginRun(edm::Run const&run, const edm::EventSetup& es)
{
  _theFinder->init();
}


void PhotonConversionTrajectorySeedProducerFromSingleLeg::produce(edm::Event& ev, const edm::EventSetup& es)
{
  //TimeMe myTest("PhotonConversionTrajectorySeedProducerFromSingleLeg::produce");

  //FIXME 
  //remove this
  //edm::CPUTimer cpu_timer;
  //cpu_timer.start();
  //--------------------------------------


  std::auto_ptr<TrajectorySeedCollection> result( new TrajectorySeedCollection() );  
  //try{
  _theFinder->analyze(ev,es);
  if(_theFinder->getTrajectorySeedCollection()->size())
    result->insert(result->end(),
		   _theFinder->getTrajectorySeedCollection()->begin(),
		   _theFinder->getTrajectorySeedCollection()->end());
  //}catch(cms::Exception& er){
  //  edm::LogError("SeedingConversion") << " Problem in the Single Leg Conversion Seed Producer " <<er.what()<<std::endl;
  //}catch(std::exception& er){
  //  edm::LogError("SeedingConversion") << " Problem in the Single Leg Conversion Seed Producer " << er.what()<<std::endl;
  //}

  
  //edm::LogInfo("debugTrajSeedFromSingleLeg") << " TrajectorySeedCollection size " << result->size();
  ev.put(result, _newSeedCandidates);  

  //FIXME 
  //remove this
  //cpu_timer.stop();
  //std::cout << "cpu timer " << cpu_timer.realTime() << " " << cpu_timer.cpuTime() << std::endl;
  //--------------------------------------

  //FIXME 
  //This is a check part that can be removed

  if(!_DoxcheckSeedCandidates)
    return;



  std::auto_ptr<TrajectorySeedCollection> resultCheck( new TrajectorySeedCollection() );
  if(_theFinder->getTrajectorySeedCollectionOfSourceTracks()->size())
    resultCheck->insert(resultCheck->end(),
			_theFinder->getTrajectorySeedCollectionOfSourceTracks()->begin(),
			_theFinder->getTrajectorySeedCollectionOfSourceTracks()->end());
  ev.put(resultCheck , _xcheckSeedCandidates);

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonConversionTrajectorySeedProducerFromSingleLeg);
