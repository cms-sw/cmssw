#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositVetos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

#include <string>


class CandIsoDumper : public edm::EDAnalyzer {

public:
  CandIsoDumper(const edm::ParameterSet&);

  virtual ~CandIsoDumper();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
private:
   edm::InputTag src_;
};

/// constructor with config
CandIsoDumper::CandIsoDumper(const edm::ParameterSet& par) :
  src_(par.getParameter<edm::InputTag>("src"))  {
}

/// destructor
CandIsoDumper::~CandIsoDumper(){
}

/// build deposits
void CandIsoDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace reco::isodeposit;
  edm::Handle< reco::CandIsoDepositAssociationVector > hDeps;
  iEvent.getByLabel(src_, hDeps);

  static uint32_t event = 0;
  std::cout << "Dumping event " << (++event) << std::endl;

  uint32_t dep;
  for (dep = 0; dep < hDeps->size(); ++dep) {
      std::cout << "  Dumping deposit " << (dep+1) << std::endl;
      const reco::Candidate &cand = *hDeps->key(dep);
      const reco::MuIsoDeposit &val = hDeps->value(dep);
      std::cout << "      Candidate pt " << cand.pt() << ", eta " << cand.eta() << ", phi " << cand.phi() << ", energy " << cand.energy() << std::endl;
      std::cout << "      Deposit within 0.4 " << val.depositWithin(0.4, true) << "\n";
      Direction candDir(cand.eta(), cand.phi());
      AbsVetos z2v; 
      //reco::MuIsoDeposit::Vetos z2vOld; 
      //z2vOld.push_back(reco::MuIsoDeposit::Veto(candDir,0.2));
      //std::cout << "Deposit within 0.7 - 0.2: OLDV " << val.depositWithin(0.7, z2vOld) << std::endl;
      //std::cout << "Deposit within 0.7 - 0.2: OLDV " << val.depositWithin(0.7, z2vOld, true) << std::endl;
      //z2v.push_back(new ConeVeto(candDir,0.2));
      //std::cout << "Deposit within 0.7 - 0.2: ABSV " << val.depositWithin(0.7, z2v) << std::endl;
      //std::cout << "Deposit within 0.7 - 0.2: ABSV " << val.depositWithin(0.7, z2v, true) << std::endl;
      //z2v.push_back(new AngleConeVeto(candDir,0.2));
      //std::cout << "      Deposit within 0.7 - A3DV(0.2) " << val.depositWithin(0.7, z2v) << std::endl;
      for (size_t i = 0; i < z2v.size(); i++) { delete z2v[i]; }
	  std::cout << "      Dumping deposit contents: " << "\n";
	  for (reco::MuIsoDeposit::const_iterator it = val.begin(), ed = val.end(); it != ed; ++it) {
		 std::cout << "        + at (eta, phi) = (" << it->eta() << ", " << it->phi() << "): value = " << it->value() << "\n";
	  }
	  std::cout << "      -end of deposit: " << std::endl;
	  
  }
}

DEFINE_FWK_MODULE( CandIsoDumper );
