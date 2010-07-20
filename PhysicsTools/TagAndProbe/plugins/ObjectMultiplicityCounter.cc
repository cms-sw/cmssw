//

/**
  \class    ObjectMultiplicityCounter"
  \brief    Matcher of number of reconstructed objects in the event to probe 
            
  \author   Kalanand Mishra
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class ObjectMultiplicityCounter : public edm::EDProducer {
    public:
        explicit ObjectMultiplicityCounter(const edm::ParameterSet & iConfig);
        virtual ~ObjectMultiplicityCounter() ;

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
        edm::InputTag probes_;            
        edm::InputTag objects_; 
        StringCutObjectSelector<reco::Jet,true> objCut_; // lazy parsing, to allow cutting on variables not in reco::Candidate class
};

ObjectMultiplicityCounter::ObjectMultiplicityCounter(const edm::ParameterSet & iConfig) :
    probes_(iConfig.getParameter<edm::InputTag>("probes")),
    objects_(iConfig.getParameter<edm::InputTag>("objects")),
    objCut_(iConfig.existsAs<std::string>("objectSelection") ? iConfig.getParameter<std::string>("objectSelection") : "", true)
{
    produces<edm::ValueMap<float> >();
}


ObjectMultiplicityCounter::~ObjectMultiplicityCounter()
{
}

void 
ObjectMultiplicityCounter::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    // read input
    Handle<View<reco::Candidate> > probes;
    Handle<View<reco::Jet> > objects;
    iEvent.getByLabel(probes_,  probes);
    iEvent.getByLabel(objects_, objects);

    // prepare vector for output    
    std::vector<float> values;
    
    // fill
    float count = 0.0;
    View<reco::Candidate>::const_iterator probe, endprobes = probes->end(); 
    View<reco::Jet>::const_iterator object, endobjects = objects->end();
    for (object = objects->begin(); object != endobjects; ++object) {
      if ( !(objCut_(*object)) ) continue;
      count += 1.0;
    }

    for (probe = probes->begin(); probe != endprobes; ++probe) values.push_back(count);
 
    
    // convert into ValueMap and store
    std::auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
    ValueMap<float>::Filler filler(*valMap);
    filler.insert(probes, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap);
}



#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ObjectMultiplicityCounter);
