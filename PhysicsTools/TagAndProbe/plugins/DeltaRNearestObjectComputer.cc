//
// $Id: DeltaRNearestObjectComputer.cc,v 1.2 2010/02/26 22:30:50 wdd Exp $
//

/**
  \class    gDeltaRNearestObjectComputer DeltaRNearestObjectComputer.h "MuonAnalysis/MuonAssociators/interface/DeltaRNearestObjectComputer.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: DeltaRNearestObjectComputer.cc,v 1.2 2010/02/26 22:30:50 wdd Exp $
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

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class DeltaRNearestObjectComputer : public edm::EDProducer {
    public:
        explicit DeltaRNearestObjectComputer(const edm::ParameterSet & iConfig);
        virtual ~DeltaRNearestObjectComputer() ;

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
        edm::InputTag probes_;            
        edm::InputTag objects_; 
        StringCutObjectSelector<reco::Candidate,true> objCut_; // lazy parsing, to allow cutting on variables not in reco::Candidate class
};

DeltaRNearestObjectComputer::DeltaRNearestObjectComputer(const edm::ParameterSet & iConfig) :
    probes_(iConfig.getParameter<edm::InputTag>("probes")),
    objects_(iConfig.getParameter<edm::InputTag>("objects")),
    objCut_(iConfig.existsAs<std::string>("objectSelection") ? iConfig.getParameter<std::string>("objectSelection") : "", true)
{
    produces<edm::ValueMap<float> >();
}


DeltaRNearestObjectComputer::~DeltaRNearestObjectComputer()
{
}

void 
DeltaRNearestObjectComputer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    // read input
    Handle<View<reco::Candidate> > probes, objects;
    iEvent.getByLabel(probes_,  probes);
    iEvent.getByLabel(objects_, objects);

    // prepare vector for output    
    std::vector<float> values;
    
    // fill
    View<reco::Candidate>::const_iterator probe, endprobes = probes->end(), object, endobjects = objects->end();
    for (probe = probes->begin(); probe != endprobes; ++probe) {
        double dr2min = 10000;
        for (object = objects->begin(); object != endobjects; ++object) {
            if (!objCut_(*object)) continue;
            double dr2 = deltaR2(*probe, *object);
            if (dr2 < dr2min) { dr2min = dr2; }
        }
        values.push_back(sqrt(dr2min));
    }

    // convert into ValueMap and store
    std::auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
    ValueMap<float>::Filler filler(*valMap);
    filler.insert(probes, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap);
}



#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeltaRNearestObjectComputer);
