// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetProducers/interface/PileupJPTJetIdAlgo.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"


// ------------------------------------------------------------------------------------------
class PileupJPTJetIdProducer : public edm::EDProducer {
public:
	explicit PileupJPTJetIdProducer(const edm::ParameterSet&);
	~PileupJPTJetIdProducer();

private:
	virtual void produce(edm::Event&, const edm::EventSetup&);
      
	virtual void beginRun(edm::Run&, edm::EventSetup const&);
	virtual void endRun(edm::Run&, edm::EventSetup const&);
	virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
	virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

	edm::InputTag jets_;
        edm::EDGetTokenT<edm::View<reco::JPTJet> > input_token_;
             bool allowMissingInputs_;
             int verbosity;
        cms::PileupJPTJetIdAlgo* pualgo;
	
};

// ------------------------------------------------------------------------------------------
PileupJPTJetIdProducer::PileupJPTJetIdProducer(const edm::ParameterSet& iConfig)
{
	jets_ = iConfig.getParameter<edm::InputTag>("jets");
	input_token_ = consumes<edm::View<reco::JPTJet> >(jets_);
        verbosity   = iConfig.getParameter<int>("Verbosity");
        allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
        pualgo = new cms::PileupJPTJetIdAlgo(iConfig); 
        pualgo->bookMVAReader();
	produces<edm::ValueMap<float> > ("JPTPUDiscriminant");
	produces<edm::ValueMap<int> > ("JPTPUId");
}



// ------------------------------------------------------------------------------------------
PileupJPTJetIdProducer::~PileupJPTJetIdProducer()
{
}


// ------------------------------------------------------------------------------------------
void
PileupJPTJetIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace std;
    using namespace reco;
    edm::Handle<View<JPTJet> > jets;
    iEvent.getByToken(input_token_, jets);
    vector<float> mva;
    vector<int> idflag;
    for ( unsigned int i=0; i<jets->size(); ++i ) {
        int b = -1;
        const JPTJet & jet = jets->at(i);

        float mvapu = pualgo->fillJPTBlock(&jet);

        mva.push_back(mvapu);

  // Get PUid type
///|eta|<2.6
//WP 95% JPT PUID > 0.3
//WP 90% JPT PUID > 0.7
//WP 80% JPT PUID > 0.9

//|eta|>=2.6
//WP 90% JPT PUID > -0.55
//WP 80% JPT PUID > -0.3
//WP 70% JPT PUID > -0.1
        
        if(fabs(jet.eta()) < 2.6 ) {
           if( mvapu > 0.3 ) b = 0;
           if( mvapu > 0.7 ) b = 1;
           if( mvapu > 0.9 ) b = 2;
        } else {
           if( mvapu > -0.55 ) b = 0;
           if( mvapu > -0.3 ) b = 1;
           if( mvapu > -0.1 ) b = 2;
        }

        idflag.push_back(b); 

       if(verbosity > 0) std::cout<<" PUID producer::Corrected JPT Jet is "<<jet.pt()<<" "<<jet.eta()<<" "<<jet.phi()<<" "<<jet.getSpecific().Zch<<std::endl; 
    
    }
     
    auto_ptr<ValueMap<float> > mvaout(new ValueMap<float>());
    ValueMap<float>::Filler mvafiller(*mvaout);
    mvafiller.insert(jets,mva.begin(),mva.end());
    mvafiller.fill();
    iEvent.put(mvaout,"JPTPUDiscriminant");

    auto_ptr<ValueMap<int> > idflagout(new ValueMap<int>());
    ValueMap<int>::Filler idflagfiller(*idflagout);
    idflagfiller.insert(jets,idflag.begin(),idflag.end());
    idflagfiller.fill();
    iEvent.put(idflagout,"JPTPUId");
       
}

// ------------------------------------------------------------------------------------------
void 
PileupJPTJetIdProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------------------------------------------------------------------------------------
void 
PileupJPTJetIdProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------------------------------------------------------------------------------------
void 
PileupJPTJetIdProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------------------------------------------------------------------------------------
void 
PileupJPTJetIdProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(PileupJPTJetIdProducer);
