//
//

/**
  \class    NearbyCandCountComputer NearbyCandCountComputer.h "PhysicsTools/TagAndProbe/interface/NearbyCandCountComputer.h"
  \brief    Count candidates near to another candidate, write result in ValueMap

            Implementation notice: not templated, because we want to allow cuts on the pair through PATDiObjectProxy

  \author   Giovanni Petrucciani
  \version  $Id: NearbyCandCountComputer.cc,v 1.2 2010/07/09 14:03:51 gpetrucc Exp $
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
#include "PhysicsTools/PatUtils/interface/PATDiObjectProxy.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class NearbyCandCountComputer : public edm::EDProducer {
    public:
        explicit NearbyCandCountComputer(const edm::ParameterSet & iConfig);
        virtual ~NearbyCandCountComputer() ;

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    private:
        edm::EDGetTokenT<edm::View<reco::Candidate> > probesToken_;
        edm::EDGetTokenT<edm::View<reco::Candidate> > objectsToken_;
        double deltaR2_;
        StringCutObjectSelector<reco::Candidate,true> objCut_; // lazy parsing, to allow cutting on variables not in reco::Candidate class
        StringCutObjectSelector<pat::DiObjectProxy,true> pairCut_;
};

NearbyCandCountComputer::NearbyCandCountComputer(const edm::ParameterSet & iConfig) :
    probesToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("probes"))),
    objectsToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("objects"))),
    deltaR2_(std::pow(iConfig.getParameter<double>("deltaR"), 2)),
    objCut_(iConfig.existsAs<std::string>("objectSelection") ? iConfig.getParameter<std::string>("objectSelection") : "", true),
    pairCut_(iConfig.existsAs<std::string>("pairSelection") ? iConfig.getParameter<std::string>("pairSelection") : "", true)
{
    produces<edm::ValueMap<float> >();
}


NearbyCandCountComputer::~NearbyCandCountComputer()
{
}

void
NearbyCandCountComputer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    // read input
    Handle<View<reco::Candidate> > probes, objects;
    iEvent.getByToken(probesToken_,  probes);
    iEvent.getByToken(objectsToken_, objects);

    // prepare vector for output
    std::vector<float> values;

    // fill
    View<reco::Candidate>::const_iterator probe, endprobes = probes->end();
    View<reco::Candidate>::const_iterator object, beginobjects = objects->begin(), endobjects = objects->end();
    for (probe = probes->begin(); probe != endprobes; ++probe) {
        float count = 0;
        for (object = beginobjects; object != endobjects; ++object) {
            if ((deltaR2(*probe, *object) < deltaR2_) &&
                objCut_(*object) &&
                pairCut_(pat::DiObjectProxy(*probe, *object))) {
                    count++;
            }
        }
        values.push_back(count);
    }

    // convert into ValueMap and store
    std::auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
    ValueMap<float>::Filler filler(*valMap);
    filler.insert(probes, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap);
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NearbyCandCountComputer);
