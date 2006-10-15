#ifndef JetPlusTrackProducers_h
#define JetPlusTrackProducers_h

/* Template producer to correct jet
    F.Ratnikov (UMd)
    Mar 2, 2006
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackAlgorithm.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{
  class JetPlusTrack : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit JetPlusTrack (const edm::ParameterSet& ps);

    virtual ~JetPlusTrack () {}

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
  
    JetPlusTrackAlgorithm mAlgorithm;
    edm::InputTag mInputJets;
    edm::InputTag mInputCaloTower;
    edm::InputTag mInputPVfCTF;
    std::string m_inputTrackLabel;
    
    double theRcalo;
    double theRvert;
    int theResponse;
    
  };
}


#endif
