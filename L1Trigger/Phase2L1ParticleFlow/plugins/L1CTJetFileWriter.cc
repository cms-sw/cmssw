#include <memory>
#include <numeric>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

//
// class declaration
//

class L1CTJetFileWriter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1CTJetFileWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------constants, enums and typedefs ---------
  std::vector<edm::ParameterSet> collections_;

  size_t nFramesPerBX_;
  size_t ctl2BoardTMUX_;
  size_t gapLengthOutput_;
  size_t maxLinesPerFile_;
  std::map<l1t::demo::LinkId, std::pair<l1t::demo::ChannelSpec, std::vector<size_t>>> channelSpecsOutputToGT_;

  // ----------member functions ----------------------
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  std::vector<ap_uint<64>> encodeJets(const std::vector<l1t::PFJet> jets, unsigned nJets);
  std::vector<ap_uint<64>> encodeSums(const std::vector<l1t::EtSum> sums, unsigned nSums);

  l1t::demo::BoardDataWriter fileWriterOutputToGT_;
  std::vector<std::pair<edm::EDGetTokenT<edm::View<l1t::PFJet>>, edm::EDGetTokenT<edm::View<l1t::EtSum>>>> tokens_;
  std::vector<std::pair<bool, bool>> tokensToWrite_;
  std::vector<unsigned> nJets_;
  std::vector<unsigned> nSums_;
};

L1CTJetFileWriter::L1CTJetFileWriter(const edm::ParameterSet& iConfig)
    : collections_(iConfig.getParameter<std::vector<edm::ParameterSet>>("collections")),
      nFramesPerBX_(iConfig.getParameter<unsigned>("nFramesPerBX")),
      ctl2BoardTMUX_(iConfig.getParameter<unsigned>("TMUX")),
      maxLinesPerFile_(iConfig.getParameter<unsigned>("maxLinesPerFile")),
      channelSpecsOutputToGT_{{{"jets", 0}, {{ctl2BoardTMUX_, gapLengthOutput_}, {0}}}},
      fileWriterOutputToGT_(l1t::demo::parseFileFormat(iConfig.getParameter<std::string>("format")),
                            iConfig.getParameter<std::string>("outputFilename"),
                            iConfig.getParameter<std::string>("outputFileExtension"),
                            nFramesPerBX_,
                            ctl2BoardTMUX_,
                            maxLinesPerFile_,
                            channelSpecsOutputToGT_) {
  for (const auto& pset : collections_) {
    edm::EDGetTokenT<edm::View<l1t::PFJet>> jetToken;
    edm::EDGetTokenT<edm::View<l1t::EtSum>> mhtToken;
    unsigned nJets = pset.getParameter<unsigned>("nJets");
    unsigned nSums = pset.getParameter<unsigned>("nSums");
    nJets_.push_back(nJets);
    nSums_.push_back(nSums);
    bool writeJetToken(false), writeMhtToken(false);
    if (nJets > 0) {
      jetToken = consumes<edm::View<l1t::PFJet>>(pset.getParameter<edm::InputTag>("jets"));
      writeJetToken = true;
    }
    if (nSums > 0) {
      mhtToken = consumes<edm::View<l1t::EtSum>>(pset.getParameter<edm::InputTag>("mht"));
      writeMhtToken = true;
    }
    tokens_.emplace_back(jetToken, mhtToken);
    tokensToWrite_.emplace_back(writeJetToken, writeMhtToken);
  }
  gapLengthOutput_ = ctl2BoardTMUX_ * nFramesPerBX_ - 2 * std::accumulate(nJets_.begin(), nJets_.end(), 0) -
                     std::accumulate(nSums_.begin(), nSums_.end(), 0);
}

void L1CTJetFileWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // 1) Pack collections in the order they're specified. jets then sums within collection
  std::vector<ap_uint<64>> link_words;
  for (unsigned iCollection = 0; iCollection < collections_.size(); iCollection++) {
    if (tokensToWrite_.at(iCollection).first) {
      const auto& jetToken = tokens_.at(iCollection).first;
      // 2) Encode jet information onto vectors containing link data
      const edm::View<l1t::PFJet>& jets = iEvent.get(jetToken);
      std::vector<l1t::PFJet> sortedJets;
      sortedJets.reserve(jets.size());
      std::copy(jets.begin(), jets.end(), std::back_inserter(sortedJets));

      std::stable_sort(
          sortedJets.begin(), sortedJets.end(), [](l1t::PFJet i, l1t::PFJet j) { return (i.hwPt() > j.hwPt()); });
      const auto outputJets(encodeJets(sortedJets, nJets_.at(iCollection)));
      link_words.insert(link_words.end(), outputJets.begin(), outputJets.end());
    }

    if (tokensToWrite_.at(iCollection).second) {
      // 3) Encode sums onto vectors containing link data
      const auto& mhtToken = tokens_.at(iCollection).second;
      const edm::View<l1t::EtSum>& mht = iEvent.get(mhtToken);
      std::vector<l1t::EtSum> orderedSums;
      std::copy(mht.begin(), mht.end(), std::back_inserter(orderedSums));
      const auto outputSums(encodeSums(orderedSums, nSums_.at(iCollection)));
      link_words.insert(link_words.end(), outputSums.begin(), outputSums.end());
    }
  }
  // 4) Pack jet information into 'event data' object, and pass that to file writer
  l1t::demo::EventData eventDataJets;
  eventDataJets.add({"jets", 0}, link_words);
  fileWriterOutputToGT_.addEvent(eventDataJets);
}

// ------------ method called once each job just after ending the event loop  ------------
void L1CTJetFileWriter::endJob() {
  // Writing pending events to file before exiting
  fileWriterOutputToGT_.flush();
}

std::vector<ap_uint<64>> L1CTJetFileWriter::encodeJets(const std::vector<l1t::PFJet> jets, const unsigned nJets) {
  // Encode up to nJets jets, padded with 0s
  std::vector<ap_uint<64>> jet_words(2 * nJets, 0);  // allocate 2 words per jet
  for (unsigned i = 0; i < std::min(nJets, (uint)jets.size()); i++) {
    l1t::PFJet j = jets.at(i);
    jet_words[2 * i] = j.encodedJet()[0];
    jet_words[2 * i + 1] = j.encodedJet()[1];
  }
  return jet_words;
}

std::vector<ap_uint<64>> L1CTJetFileWriter::encodeSums(const std::vector<l1t::EtSum> sums, unsigned nSums) {
  // Need two l1t::EtSum for each GT Sum
  std::vector<ap_uint<64>> sum_words;
  for (unsigned i = 0; i < nSums; i++) {
    if (2 * i < sums.size()) {
      l1gt::Sum gtSum;
      gtSum.valid = 1;  // if the sums are sent at all, they are valid
      gtSum.vector_pt.V = sums.at(2 * i + 1).hwPt();
      gtSum.vector_phi.V = sums.at(2 * i + 1).hwPhi();
      gtSum.scalar_pt.V = sums.at(2 * i).hwPt();
      sum_words.push_back(gtSum.pack_ap());
    } else {
      sum_words.push_back(0);
    }
  }
  return sum_words;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1CTJetFileWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.addOptional<edm::InputTag>("jets");
    vpsd1.addOptional<edm::InputTag>("mht");
    vpsd1.add<uint>("nJets", 0);
    vpsd1.add<uint>("nSums", 0);
    desc.addVPSet("collections", vpsd1);
  }
  desc.add<std::string>("outputFilename");
  desc.add<std::string>("outputFileExtension", "txt");
  desc.add<uint32_t>("nJets", 12);
  desc.add<uint32_t>("nFramesPerBX", 9);
  desc.add<uint32_t>("TMUX", 6);
  desc.add<uint32_t>("maxLinesPerFile", 1024);
  desc.add<std::string>("format", "EMPv2");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CTJetFileWriter);
