#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

#include <string>

using reco::isodeposit::Direction;

class CandIsoDumper : public edm::one::EDAnalyzer<> {
public:
  CandIsoDumper(const edm::ParameterSet&);

  virtual ~CandIsoDumper();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  edm::EDGetTokenT<reco::IsoDepositMap> srcToken_;
};

/// constructor with config
CandIsoDumper::CandIsoDumper(const edm::ParameterSet& par)
    : srcToken_(consumes<reco::IsoDepositMap>(par.getParameter<edm::InputTag>("src"))) {}

/// destructor
CandIsoDumper::~CandIsoDumper() {}

/// build deposits
void CandIsoDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace reco::isodeposit;
  edm::Handle<reco::IsoDepositMap> hDeps;
  iEvent.getByToken(srcToken_, hDeps);

  static uint32_t event = 0;
  std::cout << "Dumping event " << (++event) << std::endl;

  uint32_t dep;
  typedef reco::IsoDepositMap::const_iterator iterator_i;
  typedef reco::IsoDepositMap::container::const_iterator iterator_ii;
  iterator_i depI = hDeps->begin();
  iterator_i depIEnd = hDeps->end();
  for (; depI != depIEnd; ++depI) {
    std::vector<double> retV(depI.size(), 0);
    edm::Handle<edm::View<reco::Candidate> > candH;
    iEvent.get(depI.id(), candH);
    edm::View<reco::Candidate>::const_iterator candII;

    iterator_ii depII;
    for (dep = 0, depII = depI.begin(), candII = candH->begin(); depII != depI.end(); ++depII, ++dep, ++candII) {
      std::cout << "  Dumping deposit " << (dep + 1) << std::endl;
      const reco::Candidate& cand = *candII;
      const reco::IsoDeposit& val = *depII;
      std::cout << "      Candidate pt " << cand.pt() << ", eta " << cand.eta() << ", phi " << cand.phi() << ", energy "
                << cand.energy() << std::endl;
      std::cout << "      Deposit within 0.4 " << val.depositWithin(0.4, true) << "\n";
      reco::isodeposit::Direction candDir(cand.eta(), cand.phi());
      AbsVetos z2v;
      //reco::IsoDeposit::Vetos z2vOld;
      //z2vOld.push_back(reco::IsoDeposit::Veto(candDir,0.2));
      //std::cout << "Deposit within 0.7 - 0.2: OLDV " << val.depositWithin(0.7, z2vOld) << std::endl;
      //std::cout << "Deposit within 0.7 - 0.2: OLDV " << val.depositWithin(0.7, z2vOld, true) << std::endl;
      //z2v.push_back(new ConeVeto(candDir,0.2));
      //std::cout << "Deposit within 0.7 - 0.2: ABSV " << val.depositWithin(0.7, z2v) << std::endl;
      //std::cout << "Deposit within 0.7 - 0.2: ABSV " << val.depositWithin(0.7, z2v, true) << std::endl;
      //z2v.push_back(new AngleConeVeto(candDir,0.2));
      //std::cout << "      Deposit within 0.7 - A3DV(0.2) " << val.depositWithin(0.7, z2v) << std::endl;

      for (size_t i = 0; i < z2v.size(); i++) {
        delete z2v[i];
      }
      std::cout << "      Dumping deposit contents: "
                << "\n";
      for (reco::IsoDeposit::const_iterator it = val.begin(), ed = val.end(); it != ed; ++it) {
        std::cout << "        + at dR(eta, phi) = " << it->dR() << " (" << it->eta() << ", " << it->phi()
                  << "): value = " << it->value() << "\n";
      }
      std::cout << "      -end of deposit: " << std::endl;

    }  //!for (depII)
  }    //!for (depI)
}

DEFINE_FWK_MODULE(CandIsoDumper);
