
#include "HLTMuonL1TRegionalFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

HLTMuonL1TRegionalFilter::HLTMuonL1TRegionalFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      candTag_(iConfig.getParameter<edm::InputTag>("CandTag")),
      candToken_(consumes<l1t::MuonBxCollection>(candTag_)),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
      minN_(iConfig.getParameter<int>("MinN")),
      centralBxOnly_(iConfig.getParameter<bool>("CentralBxOnly")) {
  using namespace std;
  using namespace edm;

  // read in the eta-range dependent parameters
  const vector<ParameterSet> cuts = iConfig.getParameter<vector<ParameterSet> >("Cuts");
  size_t ranges = cuts.size();
  if (ranges == 0) {
    throw edm::Exception(errors::Configuration) << "Please provide at least one PSet in the Cuts VPSet!";
  }
  etaBoundaries_.reserve(ranges + 1);
  minPts_.reserve(ranges);
  qualityBitMasks_.reserve(ranges);
  for (size_t i = 0; i < ranges; i++) {
    //set the eta range
    vector<double> etaRange = cuts[i].getParameter<vector<double> >("EtaRange");
    if (etaRange.size() != 2 || etaRange[0] >= etaRange[1]) {
      throw edm::Exception(errors::Configuration) << "EtaRange must have two non-equal values in increasing order!";
    }
    if (i == 0) {
      etaBoundaries_.push_back(etaRange[0]);
    } else if (etaBoundaries_[i] != etaRange[0]) {
      throw edm::Exception(errors::Configuration)
          << "EtaRanges must be disjoint without gaps and listed in increasing eta order!";
    }
    etaBoundaries_.push_back(etaRange[1]);

    //set the minPt
    minPts_.push_back(cuts[i].getParameter<double>("MinPt"));

    //set the quality bit masks
    qualityBitMasks_.push_back(0);
    vector<unsigned int> qualities = cuts[i].getParameter<vector<unsigned int> >("QualityBits");
    for (unsigned int qualitie : qualities) {
      //       if(7U < qualities[j]){ // qualities[j] >= 0, since qualities[j] is unsigned   //FIXME: this will be updated once we have info from L1
      //         throw edm::Exception(errors::Configuration) << "QualityBits must be between 0 and 7 !";
      //       }
      qualityBitMasks_[i] |= 1 << qualitie;
    }
  }

  // dump parameters for debugging
  if (edm::isDebugEnabled()) {
    ostringstream ss;
    ss << "Constructed with parameters:" << endl;
    ss << "    CandTag = " << candTag_.encode() << endl;
    ss << "    PreviousCandTag = " << previousCandTag_.encode() << endl;
    ss << "    EtaBoundaries = \t" << etaBoundaries_[0];
    for (size_t i = 1; i < etaBoundaries_.size(); i++) {
      ss << '\t' << etaBoundaries_[i];
    }
    ss << endl;
    ss << "    MinPts =        \t    " << minPts_[0];
    for (size_t i = 1; i < minPts_.size(); i++) {
      ss << "\t    " << minPts_[i];
    }
    ss << endl;
    ss << "    QualityBitMasks =  \t    " << qualityBitMasks_[0];
    for (size_t i = 1; i < qualityBitMasks_.size(); i++) {
      ss << "\t    " << qualityBitMasks_[i];
    }
    ss << endl;
    ss << "    MinN = " << minN_ << endl;
    ss << "    saveTags= " << saveTags();
    LogDebug("HLTMuonL1TRegionalFilter") << ss.str();
  }
}

HLTMuonL1TRegionalFilter::~HLTMuonL1TRegionalFilter() = default;

void HLTMuonL1TRegionalFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("CandTag", edm::InputTag("hltGmtStage2Digis"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag("hltL1sL1SingleMu20"));
  desc.add<int>("MinN", 1);
  desc.add<bool>("CentralBxOnly", true);

  edm::ParameterSetDescription validator;
  std::vector<edm::ParameterSet> defaults(3);

  std::vector<double> etaRange;
  double minPt;
  std::vector<unsigned int> qualityBits;

  etaRange.clear();
  etaRange.push_back(-2.5);
  etaRange.push_back(+2.5);
  minPt = 20.0;
  qualityBits.clear();
  qualityBits.push_back(6);
  qualityBits.push_back(7);
  validator.add<std::vector<double> >("EtaRange", etaRange);
  validator.add<double>("MinPt", minPt);
  validator.add<std::vector<unsigned int> >("QualityBits", qualityBits);

  etaRange.clear();
  etaRange.push_back(-2.5);
  etaRange.push_back(-1.6);
  minPt = 20.0;
  qualityBits.clear();
  qualityBits.push_back(6);
  qualityBits.push_back(7);
  defaults[0].addParameter<std::vector<double> >("EtaRange", etaRange);
  defaults[0].addParameter<double>("MinPt", minPt);
  defaults[0].addParameter<std::vector<unsigned int> >("QualityBits", qualityBits);

  etaRange.clear();
  etaRange.push_back(-1.6);
  etaRange.push_back(+1.6);
  minPt = 20.0;
  qualityBits.clear();
  qualityBits.push_back(7);
  defaults[1].addParameter<std::vector<double> >("EtaRange", etaRange);
  defaults[1].addParameter<double>("MinPt", minPt);
  defaults[1].addParameter<std::vector<unsigned int> >("QualityBits", qualityBits);

  etaRange.clear();
  etaRange.push_back(+1.6);
  etaRange.push_back(+2.5);
  minPt = 20.0;
  qualityBits.clear();
  qualityBits.push_back(6);
  qualityBits.push_back(7);
  edm::ParameterSetDescription element2;
  defaults[2].addParameter<std::vector<double> >("EtaRange", etaRange);
  defaults[2].addParameter<double>("MinPt", minPt);
  defaults[2].addParameter<std::vector<unsigned int> >("QualityBits", qualityBits);

  desc.addVPSet("Cuts", validator, defaults);

  descriptions.add("hltMuonL1TRegionalFilter", desc);
}

bool HLTMuonL1TRegionalFilter::hltFilter(edm::Event& iEvent,
                                         const edm::EventSetup& iSetup,
                                         trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace trigger;
  using namespace l1t;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // get hold of all muons
  Handle<l1t::MuonBxCollection> allMuons;
  iEvent.getByToken(candToken_, allMuons);

  // get hold of muons that fired the previous level
  Handle<TriggerFilterObjectWithRefs> previousLevelCands;
  iEvent.getByToken(previousCandToken_, previousLevelCands);

  vector<MuonRef> prevMuons;
  previousLevelCands->getObjects(TriggerL1Mu, prevMuons);

  // look at all mucands,  check cuts and add to filter object
  int n = 0;

  for (int ibx = allMuons->getFirstBX(); ibx <= allMuons->getLastBX(); ++ibx) {
    if (centralBxOnly_ && (ibx != 0))
      continue;
    for (auto it = allMuons->begin(ibx); it != allMuons->end(ibx); it++) {
      MuonRef muon(allMuons, distance(allMuons->begin(allMuons->getFirstBX()), it));

      // Only select muons that were selected in the previous level
      if (find(prevMuons.begin(), prevMuons.end(), muon) == prevMuons.end())
        continue;

      //check maxEta cut
      float eta = muon->eta();
      int region = -1;
      for (size_t r = 0; r < etaBoundaries_.size() - 1; r++) {
        if (etaBoundaries_[r] <= eta && eta <= etaBoundaries_[r + 1]) {
          region = r;
          break;
        }
      }
      if (region == -1)
        continue;

      //check pT cut
      if (muon->pt() < minPts_[region])
        continue;

      //check quality cut
      if (qualityBitMasks_[region]) {
        int quality = (it->hwQual() == 0 ? 0 : (1 << it->hwQual()));
        if ((quality & qualityBitMasks_[region]) == 0)
          continue;
      }

      //we have a good candidate
      n++;
      filterproduct.addObject(TriggerL1Mu, muon);
    }
  }

  if (saveTags())
    filterproduct.addCollectionTag(candTag_);

  // filter decision
  const bool accept(n >= minN_);

  // dump event for debugging
  if (edm::isDebugEnabled()) {
    LogTrace("HLTMuonL1TRegionalFilter") << "\nHLTMuonL1TRegionalFilter -----------------------------------------------"
                                         << endl;
    LogTrace("HLTMuonL1TRegionalFilter") << "L1mu#" << '\t' << "q*pt" << '\t' << '\t' << "eta" << '\t' << "phi" << '\t'
                                         << "quality" << '\t' << "isPrev\t " << endl;
    LogTrace("HLTMuonL1TRegionalFilter") << "--------------------------------------------------------------------------"
                                         << endl;

    vector<MuonRef> firedMuons;
    filterproduct.getObjects(TriggerL1Mu, firedMuons);
    for (size_t i = 0; i < firedMuons.size(); i++) {
      l1t::MuonRef mu = firedMuons[i];
      bool isPrev = find(prevMuons.begin(), prevMuons.end(), mu) != prevMuons.end();
      LogTrace("HLTMuonL1TRegionalFilter")
          << i << '\t' << setprecision(2) << scientific << mu->charge() * mu->pt() << '\t' << fixed << mu->eta() << '\t'
          << mu->phi() << '\t' << mu->hwQual() << '\t' << isPrev << endl;
    }
    LogTrace("HLTMuonL1TRegionalFilter") << "--------------------------------------------------------------------------"
                                         << endl;
    LogTrace("HLTMuonL1TRegionalFilter") << "Decision of this filter is " << accept
                                         << ", number of muons passing = " << filterproduct.l1tmuonSize();
  }

  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMuonL1TRegionalFilter);
