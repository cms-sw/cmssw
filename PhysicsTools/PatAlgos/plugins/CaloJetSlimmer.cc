
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>

class CaloJetSlimmer : public edm::stream::EDProducer<> {

public:

CaloJetSlimmer( edm::ParameterSet const & params ):
    srcToken_(consumes<edm::View<reco::CaloJet> >( params.getParameter<edm::InputTag>("src") )),
    cut_( params.getParameter<std::string>("cut") ),
    selector_( cut_ )
    {
        produces< reco::CaloJetCollection> ();
    }
   
    virtual ~CaloJetSlimmer() {}
   
    virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
        
        auto caloJets = std::make_unique<reco::CaloJetCollection>();
        
        edm::Handle< edm::View<reco::CaloJet> > h_jets;
        iEvent.getByToken( srcToken_, h_jets);

        for (auto const& ijet : *h_jets){
           if( selector_(ijet) ){
              reco::CaloJet::Specific tmp_specific;
              const reco::Candidate::Point orivtx(0,0,0);
              tmp_specific.mEnergyFractionEm = ijet.emEnergyFraction();
              tmp_specific.mEnergyFractionHadronic = ijet.energyFractionHadronic();
              reco::CaloJet newCaloJet(ijet.p4(), orivtx, tmp_specific);
              float jetArea = ijet.jetArea();
              newCaloJet.setJetArea(jetArea);          
              caloJets->push_back(newCaloJet);
           }
         

        } 
        iEvent.put(std::move(caloJets), "");
    }


protected:
    const edm::EDGetTokenT<edm::View<reco::CaloJet> > srcToken_;
    const std::string                                cut_;
    const StringCutObjectSelector<reco::Jet>               selector_;
};


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloJetSlimmer);
