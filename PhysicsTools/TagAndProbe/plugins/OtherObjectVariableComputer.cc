//

/**
  \class    OtherObjectVariableComputer"
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
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

template<typename T>
class OtherObjectVariableComputer : public edm::EDProducer {
    public:
        explicit OtherObjectVariableComputer(const edm::ParameterSet & iConfig);
        virtual ~OtherObjectVariableComputer() ;

        virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:
        edm::InputTag probes_;            
        edm::InputTag objects_; 
        StringObjectFunction<T,true>    objVar_;
        double default_;
        StringCutObjectSelector<T,true> objCut_; // lazy parsing, to allow cutting on variables not in reco::Candidate class
        bool doSort_;
        StringObjectFunction<T,true>    objSort_;
};

template<typename T>
OtherObjectVariableComputer<T>::OtherObjectVariableComputer(const edm::ParameterSet & iConfig) :
    probes_(iConfig.getParameter<edm::InputTag>("probes")),
    objects_(iConfig.getParameter<edm::InputTag>("objects")),
    objVar_(iConfig.getParameter<std::string>("expression")),
    default_(iConfig.getParameter<double>("default")),
    objCut_(iConfig.existsAs<std::string>("objectSelection") ? iConfig.getParameter<std::string>("objectSelection") : "", true),
    doSort_(iConfig.existsAs<std::string>("objectSortDescendingBy")),
    objSort_(doSort_ ? iConfig.getParameter<std::string>("objectSortDescendingBy") : "1", true)
{
    produces<edm::ValueMap<float> >();
}


template<typename T>
OtherObjectVariableComputer<T>::~OtherObjectVariableComputer()
{
}

template<typename T>
void 
OtherObjectVariableComputer<T>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    // read input
    Handle<View<reco::Candidate> > probes;
    Handle<View<T> > objects;
    iEvent.getByLabel(probes_,  probes);
    iEvent.getByLabel(objects_, objects);
    
    // fill
    std::vector<std::pair<double, double> > selected;
    typename View<T>::const_iterator object, endobjects = objects->end();
    for (object = objects->begin(); object != endobjects; ++object) {
      if (objCut_(*object)) {
        selected.push_back(std::pair<double, double>(objSort_(*object), objVar_(*object)));
        if (!doSort_) break; // if we take just the first one, there's no need of computing the others
      }
    }
    if (doSort_ && selected.size() > 1) std::sort(selected.begin(), selected.end()); // sorts (ascending)

    // prepare vector for output    
    std::vector<float> values(probes->size(), (selected.empty() ? default_ : selected.back().second));

    // convert into ValueMap and store
    std::auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
    ValueMap<float>::Filler filler(*valMap);
    filler.insert(probes, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap);
}


typedef OtherObjectVariableComputer<reco::Candidate> OtherCandVariableComputer;
//typedef OtherObjectVariableComputer<reco::Track>     OtherTrackVariableComputer;
//typedef OtherObjectVariableComputer<reco::Vertex>    OtherVertexVariableComputer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OtherCandVariableComputer);
//DEFINE_FWK_MODULE(OtherTrackVariableComputer);
//DEFINE_FWK_MODULE(OtherVertexVariableComputer);
