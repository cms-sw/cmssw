//Copied from L1TStage2CaloAnalyzer on 2023.08.04 as it exists in CMSSW_13_1_0_pre4
//Modified by Chris McGinn to instead work for just ZDC etSums
//Contact at christopher.mc.ginn@cern.ch or cfmcginn @ github for bugs

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//For the output
#include "TTree.h"
//string for some branch handling
#include <string>

//
// class declaration
//

namespace l1t {

  class L1TZDCEtSumsAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  public:
    explicit L1TZDCEtSumsAnalyzer(const edm::ParameterSet&);
    ~L1TZDCEtSumsAnalyzer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

    //fileservice
    edm::Service<TFileService> fs_;
    //Declare a tree, member and pointer
    TTree* etSumZdcTree_;
    //Declare the etSum max and bpx max
    //MAX Here is effectively a hardcoded number - you just need to know it from the producer - realistically this is best handled by something passed but for now (2023.08.17) we hardcode
    static const int maxBPX_ = 5;
    float etSumZdcP_[maxBPX_];
    float etSumZdcM_[maxBPX_];

    // ----------member data ---------------------------
    edm::EDGetToken sumToken_;

    bool doHistos_;

    TFileDirectory evtDispDir_;
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
  L1TZDCEtSumsAnalyzer::L1TZDCEtSumsAnalyzer(const edm::ParameterSet& iConfig)
      : doHistos_(iConfig.getUntrackedParameter<bool>("doHistos", true)) {
    usesResource(TFileService::kSharedResource);
    //now do what ever initialization is needed

    // register what you consume and keep token for later access:
    edm::InputTag nullTag("None");

    edm::InputTag sumTag = iConfig.getParameter<edm::InputTag>("etSumTag");
    sumToken_ = consumes<l1t::EtSumBxCollection>(sumTag);

    edm::LogInfo("L1TZDCEtSumsAnalyzer") << "Processing " << sumTag.label() << std::endl;
  }

  //
  // member functions
  //

  // ------------ method called for each event  ------------
  void L1TZDCEtSumsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;

    //    Handle<EtSumBxCollection> sums;
    Handle<BXVector<l1t::EtSum> > sums;
    iEvent.getByToken(sumToken_, sums);

    int startBX = sums->getFirstBX();

    for (int ibx = startBX; ibx <= sums->getLastBX(); ++ibx) {
      for (auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr) {
        if (itr->getType() == l1t::EtSum::EtSumType::kZDCP)
          etSumZdcP_[ibx - startBX] = itr->hwPt();
        if (itr->getType() == l1t::EtSum::EtSumType::kZDCM)
          etSumZdcM_[ibx - startBX] = itr->hwPt();
      }
    }

    etSumZdcTree_->Fill();
  }

  // ------------ method called once each job just before starting event loop  ------------
  void L1TZDCEtSumsAnalyzer::beginJob() {
    etSumZdcTree_ = fs_->make<TTree>("etSumZdcTree", "");
    etSumZdcTree_->Branch("etSumZdcP", etSumZdcP_, ("etSumZdcP[" + std::to_string(maxBPX_) + "]/F").c_str());
    etSumZdcTree_->Branch("etSumZdcM", etSumZdcM_, ("etSumZdcM[" + std::to_string(maxBPX_) + "]/F").c_str());
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void L1TZDCEtSumsAnalyzer::endJob() {}

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  void L1TZDCEtSumsAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("etSumTag", edm::InputTag("l1tZDCEtSums", ""));
    descriptions.add("l1tZDCEtSumsAnalyzer", desc);
  }

}  // namespace l1t

using namespace l1t;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TZDCEtSumsAnalyzer);
