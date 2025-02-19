#include "DPGAnalysis/Skims/interface/EcalTangentFilter.h"

using namespace std;

EcalTangentFilter::EcalTangentFilter(const edm::ParameterSet& iConfig){
	fMuLabel  = iConfig.getParameter<std::string>("MuLabel");
	fMuonD0Min  = iConfig.getParameter<double>("MuD0Min");
	fMuonD0Max  = iConfig.getParameter<double>("MuD0Max");
	fVerbose  = iConfig.getParameter<bool>("Verbose");
	cout << "----------------------------------------------------------------------" << std::endl;
	cout << "--- EcalTangentFilter constructor" << std::endl;
	cout << "---  MuLabel:       " << fMuLabel << std::endl;
	cout << "---  MuonD0Min:     " << fMuonD0Min << std::endl;
	cout << "---  MuonD0Max:     " << fMuonD0Max << std::endl;
	cout << "---  Verbose:     " << fVerbose << std::endl;
	cout << "----------------------------------------------------------------------" << std::endl;
	fNgood = 0;
	fNtot  = 0;
}


EcalTangentFilter::~EcalTangentFilter(){
  std::cout << "==> EcalTangentFilter> Number of accepted/total events = " << fNgood << "/" << fNtot << std::endl;
}


// ------------ method called on each new Event  ------------
bool EcalTangentFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
	++fNtot;
	bool goodEvent(false);
	using namespace edm;
	using namespace reco;
	edm::Handle<reco::MuonCollection> Muons;
	iEvent.getByLabel(fMuLabel, Muons);
	for (MuonCollection::const_iterator muon=Muons->begin(); muon!=Muons->end(); ++muon){
		reco::TrackRef ref;
		if(muon->outerTrack().isNonnull()){
			ref = muon->outerTrack();
			// Cut on impact parameter
			double d0 = ref->d0();
			if(fabs(d0) > fMuonD0Min && fabs(d0) < fMuonD0Max) goodEvent = true;
		}
	}
	if(goodEvent){
		++fNgood;
		if(fVerbose) std::cout << "==> EcalTangentFilter> new event r" << iEvent.id().run() << "/ev" << iEvent.id().event() << " accepted, so far: " << fNgood << std::endl;
	}
	return goodEvent;
}

// ------------ method called once each job just before starting event loop  ------------
void EcalTangentFilter::beginJob(){
}

// ------------ method called once each job just after ending the event loop  ------------
void EcalTangentFilter::endJob(){
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalTangentFilter);
