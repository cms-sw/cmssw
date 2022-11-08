/**
 *  Test analyser: SharesInputTest
 *
 * Original Author:  Phillip Killewald
 *         Created:  Thu Jan 29 17:33:51 CET 2009
 */

#include <map>
#include <string>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "TNtuple.h"

class CSCSharesInputTest : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CSCSharesInputTest(const edm::ParameterSet &myConfig);

  ~CSCSharesInputTest();

private:
  virtual void beginJob();

  virtual void analyze(const edm::Event &myEvent, const edm::EventSetup &mySetup);

  virtual void endJob();

  edm::EDGetTokenT<CSCRecHit2DCollection> rh_token;
  edm::EDGetTokenT<edm::View<reco::Muon> > mu_token;

  std::map<std::string, uint64_t> counts_;

  edm::Service<TFileService> rootFile_;
  std::map<std::string, TNtuple *> ntuples_;
};
