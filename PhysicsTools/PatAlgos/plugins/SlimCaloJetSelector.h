#ifndef PhysicsTools_PatAlgos_SlimCaloJetSelector_h
#define PhysicsTools_PatAlgos_SlimCaloJetSelector_h

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>

class SlimCaloJetSelector : public edm::stream::EDFilter<> {

public:

SlimCaloJetSelector( edm::ParameterSet const & params ):
    srcToken_(consumes<edm::View<reco::CaloJet> >( params.getParameter<edm::InputTag>("src") )),
    cut_( params.getParameter<std::string>("cut") ),
    filter_( params.exists("filter") ? params.getParameter<bool>("filter"): false),
    selector_( cut_ )
    {
        produces< reco::CaloJetCollection> ("selectedCaloJets");


    }
   
    virtual ~SlimCaloJetSelector() {}

    virtual void beginJob() {}
    virtual void endJob() {}
   
    virtual bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
        
        auto caloJets = std::make_unique<reco::CaloJetCollection>();
        
        edm::Handle< edm::View<reco::CaloJet> > h_jets;
        iEvent.getByToken( srcToken_, h_jets);

        for (edm::View<reco::CaloJet>::const_iterator ibegin = h_jets->begin(),
                iend = h_jets->end(), ijet = ibegin;
                ijet != iend; ++ijet ){
           if( selector_(*ijet) ){
              reco::CaloJet::Specific tmp_specific;
              const reco::Candidate::Point orivtx(0,0,0);
              tmp_specific.mTowersArea = ijet->towersArea();
              tmp_specific.mEnergyFractionEm = ijet->emEnergyFraction();
              tmp_specific.mEnergyFractionHadronic = ijet->energyFractionHadronic();

              reco::CaloJet newCaloJet(ijet->p4(), orivtx, tmp_specific);

              caloJets->push_back(newCaloJet);
           }
         

        } 
           

        bool pass = caloJets->size()>0;
        iEvent.put(std::move(caloJets), "selectedCaloJets");
        if (filter_ )
            return pass;
        else  
            return true;
    }


protected:
    const edm::EDGetTokenT<edm::View<reco::CaloJet> > srcToken_;
    const std::string                                cut_;
    const bool                                       filter_;
    const StringCutObjectSelector<reco::Jet>               selector_;



};


#endif
