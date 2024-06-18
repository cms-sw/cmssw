#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"

/**\class PFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;

class PFBlockProducer : public edm::stream::EDProducer<> {
public:
  explicit PFBlockProducer(const edm::ParameterSet&);

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// verbose ?
  const bool verbose_;
  const edm::EDPutTokenT<reco::PFBlockCollection> putToken_;

  /// Particle flow block algorithm
  PFBlockAlgo pfBlockAlgo_;
};

DEFINE_FWK_MODULE(PFBlockProducer);

void PFBlockProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // verbosity
  desc.addUntracked<bool>("verbose", false);
  // Debug flag
  desc.addUntracked<bool>("debug", false);
  //define what we are importing into particle flow
  //from the various subdetectors
  // importers are executed in the order they are defined here!!!
  //order matters for some modules (it is pointed out where this is important)
  // you can find a list of all available importers in:
  //  plugins/importers
  {
    std::vector<edm::ParameterSet> vpset;
    vpset.reserve(12);
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GSFTrackImporter");
      pset.addParameter<edm::InputTag>("source", {"pfTrackElec"});
      pset.addParameter<bool>("gsfsAreSecondary", false);
      pset.addParameter<bool>("superClustersArePF", true);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "ConvBremTrackImporter");
      pset.addParameter<edm::InputTag>("source", {"pfTrackElec"});
      pset.addParameter<bool>("vetoEndcap", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "SuperClusterImporter");
      pset.addParameter<edm::InputTag>("source_eb",
                                       {"particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"});
      pset.addParameter<edm::InputTag>(
          "source_ee", {"particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"});
      pset.addParameter<double>("maximumHoverE", 0.5);
      pset.addParameter<double>("minSuperClusterPt", 10.0);
      pset.addParameter<double>("minPTforBypass", 100.0);
      pset.addParameter<edm::InputTag>("hbheRecHitsTag", {"hbhereco"});
      pset.addParameter<int>("maxSeverityHB", 9);
      pset.addParameter<int>("maxSeverityHE", 9);
      pset.addParameter<bool>("usePFThresholdsFromDB", false);
      pset.addParameter<bool>("superClustersArePF", true);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "ConversionTrackImporter");
      pset.addParameter<edm::InputTag>("source", {"pfConversions"});
      pset.addParameter<bool>("vetoEndcap", false);
      vpset.emplace_back(pset);
    }
    // V0's not actually used in particle flow block building so far
    //NuclearInteraction's also come in Loose and VeryLoose varieties
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "NuclearInteractionTrackImporter");
      pset.addParameter<edm::InputTag>("source", {"pfDisplacedTrackerVertex"});
      pset.addParameter<bool>("vetoEndcap", false);
      vpset.emplace_back(pset);
    }
    //for best timing GeneralTracksImporter should come after
    // all secondary track importers
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GeneralTracksImporter");
      pset.addParameter<edm::InputTag>("source", {"pfTrack"});
      pset.addParameter<bool>("vetoEndcap", false);
      pset.addParameter<edm::InputTag>("muonSrc", {"muons1stStep"});
      pset.addParameter<std::string>("trackQuality", "highPurity");
      pset.addParameter<bool>("cleanBadConvertedBrems", true);
      pset.addParameter<bool>("useIterativeTracking", true);
      pset.addParameter<std::vector<double>>("DPtOverPtCuts_byTrackAlgo", {10.0, 10.0, 10.0, 10.0, 10.0, 5.0});
      pset.addParameter<std::vector<uint32_t>>("NHitCuts_byTrackAlgo", {3, 3, 3, 3, 3, 3});
      pset.addParameter<double>("muonMaxDPtOPt", 1);
      vpset.emplace_back(pset);
    }
    // secondary GSF tracks are also turned off
    // to properly set SC based links you need to run ECAL importer
    // after you've imported all SCs to the block
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "ECALClusterImporter");
      pset.addParameter<edm::InputTag>("source", {"particleFlowClusterECAL"});
      pset.addParameter<edm::InputTag>("BCtoPFCMap", {"particleFlowSuperClusterECAL:PFClusterAssociationEBEE"});
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GenericClusterImporter");
      pset.addParameter<edm::InputTag>("source", {"particleFlowClusterHCAL"});
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GenericClusterImporter");
      pset.addParameter<edm::InputTag>("source", {"particleFlowBadHcalPseudoCluster"});
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GenericClusterImporter");
      pset.addParameter<edm::InputTag>("source", {"particleFlowClusterHO"});
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GenericClusterImporter");
      pset.addParameter<edm::InputTag>("source", {"particleFlowClusterHF"});
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("importerName", "GenericClusterImporter");
      pset.addParameter<edm::InputTag>("source", {"particleFlowClusterPS"});
      vpset.emplace_back(pset);
    }
    edm::ParameterSetDescription psd;
    psd.add<std::string>("importerName", "");
    psd.add<edm::InputTag>("source", {});
    psd.add<bool>("gsfsAreSecondary", false);
    psd.add<bool>("superClustersArePF", false);
    psd.add<edm::InputTag>("source_eb", {});
    psd.add<edm::InputTag>("source_ee", {});
    psd.add<double>("maximumHoverE", 0);
    psd.add<double>("minSuperClusterPt", 0);
    psd.add<double>("minPTforBypass", 0);
    psd.add<edm::InputTag>("hbheRecHitsTag", {});
    psd.add<int>("maxSeverityHB", 0);
    psd.add<int>("maxSeverityHE", 0);
    psd.add<bool>("usePFThresholdsFromDB", false);
    psd.add<bool>("vetoEndcap", false);
    psd.add<edm::InputTag>("muonSrc", {});
    psd.add<std::string>("trackQuality", "");
    psd.add<bool>("cleanBadConvertedBrems", false);
    psd.add<bool>("useIterativeTracking", false);
    psd.add<std::vector<double>>("DPtOverPtCuts_byTrackAlgo", {});
    psd.add<std::vector<uint32_t>>("NHitCuts_byTrackAlgo", {});
    psd.add<double>("muonMaxDPtOPt", 0);
    psd.add<edm::InputTag>("BCtoPFCMap", {});
    psd.add<double>("maxDPtOPt", 0);
    psd.add<uint32_t>("vetoMode", 0);
    psd.add<edm::InputTag>("vetoSrc", {});
    psd.add<edm::InputTag>("timeValueMap", {});
    psd.add<edm::InputTag>("timeErrorMap", {});
    psd.add<edm::InputTag>("timeQualityMap", {});
    psd.add<double>("timeQualityThreshold", 0);
    psd.add<edm::InputTag>("timeValueMapGsf", {});
    psd.add<edm::InputTag>("timeErrorMapGsf", {});
    psd.add<edm::InputTag>("timeQualityMapGsf", {});
    psd.add<bool>("useTimeQuality", false);
    desc.addVPSet("elementImporters", psd, vpset);
  }
  //linking definitions
  // you can find a list of all available linkers in:
  //  plugins/linkers
  // see : plugins/kdtrees for available KDTree Types
  // to enable a KDTree for a linking pair, write a KDTree linker
  // and set useKDTree = True in the linker PSet
  //order does not matter here since we are defining a lookup table
  {
    std::vector<edm::ParameterSet> vpset;
    vpset.reserve(18);
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "PreshowerAndECALLinker");
      pset.addParameter<std::string>("linkType", "PS1:ECAL");
      pset.addParameter<bool>("useKDTree ", true);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "PreshowerAndECALLinker");
      pset.addParameter<std::string>("linkType", "PS2:ECAL");
      pset.addParameter<bool>("useKDTree ", true);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "TrackAndECALLinker");
      pset.addParameter<std::string>("linkType", "TRACK:ECAL");
      pset.addParameter<bool>("useKDTree ", true);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "TrackAndHCALLinker");
      pset.addParameter<std::string>("linkType", "TRACK:HCAL");
      pset.addParameter<bool>("useKDTree", true);
      pset.addParameter<std::string>("trajectoryLayerEntrance", "HCALEntrance");
      pset.addParameter<std::string>("trajectoryLayerExit", "HCALExit");
      pset.addParameter<int>("nMaxHcalLinksPerTrack",
                             1);  // the max hcal links per track (negative values: no restriction)
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "TrackAndHOLinker");
      pset.addParameter<std::string>("linkType", "TRACK:HO");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "ECALAndHCALLinker");
      pset.addParameter<std::string>("linkType", "ECAL:HCAL");
      pset.addParameter<double>("minAbsEtaEcal", 2.5);
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "HCALAndHOLinker");
      pset.addParameter<std::string>("linkType", "HCAL:HO");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "HFEMAndHFHADLinker");
      pset.addParameter<std::string>("linkType", "HFEM:HFHAD");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "TrackAndTrackLinker");
      pset.addParameter<std::string>("linkType", "TRACK:TRACK");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "ECALAndECALLinker");
      pset.addParameter<std::string>("linkType", "ECAL:ECAL");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "GSFAndECALLinker");
      pset.addParameter<std::string>("linkType", "GSF:ECAL");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "TrackAndGSFLinker");
      pset.addParameter<std::string>("linkType", "TRACK:GSF");
      pset.addParameter<bool>("useKDTree", false);
      pset.addParameter<bool>("useConvertedBrems", true);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "GSFAndBREMLinker");
      pset.addParameter<std::string>("linkType", "GSF:BREM");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "GSFAndGSFLinker");
      pset.addParameter<std::string>("linkType", "GSF:GSF");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "ECALAndBREMLinker");
      pset.addParameter<std::string>("linkType", "ECAL:BREM");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "GSFAndHCALLinker");
      pset.addParameter<std::string>("linkType", "GSF:HCAL");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "HCALAndBREMLinker");
      pset.addParameter<std::string>("linkType", "HCAL:BREM");
      pset.addParameter<bool>("useKDTree", false);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("linkerName", "SCAndECALLinker");
      pset.addParameter<std::string>("linkType", "SC:ECAL");
      pset.addParameter<bool>("useKDTree", false);
      pset.addParameter<bool>("SuperClusterMatchByRef", true);
      vpset.emplace_back(pset);
    }
    edm::ParameterSetDescription psd;
    psd.add<std::string>("linkerName", "");
    psd.add<std::string>("linkType", "");
    psd.add<bool>("useKDTree", false);
    psd.add<std::string>("trajectoryLayerEntrance", "");
    psd.add<std::string>("trajectoryLayerExit", "");
    psd.add<int>("nMaxHcalLinksPerTrack", 0);
    psd.add<double>("minAbsEtaEcal", 0);
    psd.add<bool>("useConvertedBrems", false);
    psd.add<bool>("SuperClusterMatchByRef", false);
    desc.addVPSet("linkDefinitions", psd, vpset);
  }
  descriptions.addWithDefaultLabel(desc);
}

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig)
    : verbose_{iConfig.getUntrackedParameter<bool>("verbose", false)}, putToken_{produces<reco::PFBlockCollection>()} {
  bool debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  pfBlockAlgo_.setDebug(debug_);

  edm::ConsumesCollector cc = consumesCollector();
  const std::vector<edm::ParameterSet>& importers = iConfig.getParameterSetVector("elementImporters");
  pfBlockAlgo_.setImporters(importers, cc);

  const std::vector<edm::ParameterSet>& linkdefs = iConfig.getParameterSetVector("linkDefinitions");
  pfBlockAlgo_.setLinkers(linkdefs);
}

void PFBlockProducer::beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  pfBlockAlgo_.updateEventSetup(es);
}

void PFBlockProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  pfBlockAlgo_.buildElements(iEvent);

  auto blocks = pfBlockAlgo_.findBlocks();

  if (verbose_) {
    ostringstream str;
    str << pfBlockAlgo_ << endl;
    str << "number of blocks : " << blocks.size() << endl;
    str << endl;

    for (auto const& block : blocks) {
      str << block << endl;
    }

    LogInfo("PFBlockProducer") << str.str() << endl;
  }

  iEvent.emplace(putToken_, blocks);
}
