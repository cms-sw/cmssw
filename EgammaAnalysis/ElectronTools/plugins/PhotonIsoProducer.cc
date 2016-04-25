// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "EgammaAnalysis/ElectronTools/interface/PFIsolationEstimator.h"
//
// class declaration
//

class PhotonIsoProducer : public edm::EDFilter {
      public:
         explicit PhotonIsoProducer(const edm::ParameterSet&);
         ~PhotonIsoProducer();
      private:
        virtual bool filter(edm::Event&, const edm::EventSetup&);

// ----------member data ---------------------------
        bool verbose_;
        edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
        edm::EDGetTokenT<reco::PhotonCollection> photonToken_;
        edm::EDGetTokenT<reco::PFCandidateCollection> particleFlowToken_;
        std::string nameIsoCh_;
        std::string nameIsoPh_;
        std::string nameIsoNh_;

        PFIsolationEstimator isolator;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PhotonIsoProducer::PhotonIsoProducer(const edm::ParameterSet& iConfig) {
        verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);
        vertexToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexTag"));
        photonToken_ = consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("photonTag"));
        particleFlowToken_ = consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("particleFlowTag"));

        nameIsoCh_ = iConfig.getParameter<std::string>("nameValueMapIsoCh");
        nameIsoPh_ = iConfig.getParameter<std::string>("nameValueMapIsoPh");
        nameIsoNh_ = iConfig.getParameter<std::string>("nameValueMapIsoNh");


        produces<edm::ValueMap<double> >(nameIsoCh_);
        produces<edm::ValueMap<double> >(nameIsoPh_);
        produces<edm::ValueMap<double> >(nameIsoNh_);

        isolator.initializePhotonIsolation(kTRUE); //NOTE: this automatically set all the correct defaul veto values
        isolator.setConeSize(0.3);

}


PhotonIsoProducer::~PhotonIsoProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool PhotonIsoProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

        std::auto_ptr<edm::ValueMap<double> > chIsoMap(new edm::ValueMap<double>() );
	edm::ValueMap<double>::Filler chFiller(*chIsoMap);

        std::auto_ptr<edm::ValueMap<double> > phIsoMap(new edm::ValueMap<double>() );
	edm::ValueMap<double>::Filler phFiller(*phIsoMap);

        std::auto_ptr<edm::ValueMap<double> > nhIsoMap(new edm::ValueMap<double>() );
	edm::ValueMap<double>::Filler nhFiller(*nhIsoMap);

	edm::Handle<reco::VertexCollection>  vertexCollection;
	iEvent.getByToken(vertexToken_, vertexCollection);

	edm::Handle<reco::PhotonCollection> phoCollection;
	iEvent.getByToken(photonToken_, phoCollection);
	const reco::PhotonCollection *recoPho = phoCollection.product();

	// All PF Candidate for alternate isolation
	edm::Handle<reco::PFCandidateCollection> pfCandidatesH;
	iEvent.getByToken(particleFlowToken_, pfCandidatesH);
	const  reco::PFCandidateCollection thePfColl = *(pfCandidatesH.product());

        std::vector<double> chIsoValues;
	std::vector<double> phIsoValues;
	std::vector<double> nhIsoValues;
        chIsoValues.reserve(phoCollection->size());
        phIsoValues.reserve(phoCollection->size());
        nhIsoValues.reserve(phoCollection->size());

	unsigned int ivtx = 0;
	reco::VertexRef myVtxRef(vertexCollection, ivtx);


        for (reco::PhotonCollection::const_iterator aPho = recoPho->begin(); aPho != recoPho->end(); ++aPho) {

          isolator.fGetIsolation(&*aPho,
		                 &thePfColl,
				 myVtxRef,
				 vertexCollection);

	  if(verbose_) {
	    std::cout << " run " << iEvent.id().run() << " lumi " << iEvent.id().luminosityBlock() << " event " << iEvent.id().event();
	    std::cout << " pt " <<  aPho->pt() << " eta " << aPho->eta() << " phi " << aPho->phi()
		      << " charge " << aPho->charge()<< " : " << std::endl;;

	    std::cout << " ChargedIso " << isolator.getIsolationCharged() << std::endl;
	    std::cout << " PhotonIso " << isolator.getIsolationPhoton() << std::endl;
	    std::cout << " NeutralHadron Iso " << isolator.getIsolationNeutral()  << std::endl;
	  }

	  chIsoValues.push_back(isolator.getIsolationCharged());
	  phIsoValues.push_back(isolator.getIsolationPhoton());
	  nhIsoValues.push_back(isolator.getIsolationNeutral());

	}

	chFiller.insert(phoCollection, chIsoValues.begin(), chIsoValues.end() );
	chFiller.fill();

	phFiller.insert(phoCollection, phIsoValues.begin(), phIsoValues.end() );
	phFiller.fill();

	nhFiller.insert(phoCollection, nhIsoValues.begin(), nhIsoValues.end() );
	nhFiller.fill();


	iEvent.put(chIsoMap,nameIsoCh_);
	iEvent.put(phIsoMap,nameIsoPh_);
	iEvent.put(nhIsoMap,nameIsoNh_);


	return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(PhotonIsoProducer);
