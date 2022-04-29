// system includes
#include <sstream>

// user includes
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class SiStripClusterValidator : public edm::one::EDAnalyzer<> {
public:
  SiStripClusterValidator(const edm::ParameterSet& config);
  ~SiStripClusterValidator() override = default;
  virtual void endJob() override;
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  void validate(const edm::DetSetVector<SiStripCluster>&, const edm::DetSetVector<SiStripCluster>&);
  void validate(const edmNew::DetSetVector<SiStripCluster>&, const edmNew::DetSetVector<SiStripCluster>&);

private:
  inline const std::string& header() { return header_; }

  /// Input collections
  const edm::InputTag collection1Tag_;
  const edm::InputTag collection2Tag_;
  const bool dsvnew_;
  /// used to remember if there have been errors for message in endJob
  bool errors_;

  std::string header_;
};

std::ostream& operator<<(std::ostream&, const edmNew::DetSetVector<SiStripCluster>&);
std::ostream& operator<<(std::ostream&, const edm::DetSetVector<SiStripCluster>&);

SiStripClusterValidator::SiStripClusterValidator(const edm::ParameterSet& conf)
    : collection1Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection1")),
      collection2Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection2")),
      dsvnew_(conf.getUntrackedParameter<bool>("DetSetVectorNew", true)),
      errors_(false),
      header_() {
  std::stringstream ss;
  ss << " Collection1: "
     << " Type:Label:Instance:Process=\"" << (dsvnew_ ? "edmNew::DSV<SiStripClusters>" : "edm::DSV<SiStripClusters>")
     << ":" << collection1Tag_.label() << ":" << collection1Tag_.instance() << ":" << collection1Tag_.process() << "\""
     << std::endl
     << " Collection2: "
     << " Type:Label:Instance:Process=\"" << (dsvnew_ ? "edmNew::DSV<SiStripClusters>" : "edm::DSV<SiStripClusters>")
     << ":" << collection2Tag_.label() << ":" << collection2Tag_.instance() << ":" << collection2Tag_.process() << "\""
     << std::endl;
  header_ = ss.str();
  if (dsvnew_) {
    consumes<edmNew::DetSetVector<SiStripCluster> >(collection1Tag_);
    consumes<edmNew::DetSetVector<SiStripCluster> >(collection2Tag_);
  } else {
    consumes<edm::DetSetVector<SiStripCluster> >(collection1Tag_);
    consumes<edm::DetSetVector<SiStripCluster> >(collection2Tag_);
  }
}

void SiStripClusterValidator::endJob() {
  std::stringstream ss;
  ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl << header();

  if (!errors_) {
    ss << "Collections are identical in every event (assuming no exceptions thrown!)" << std::endl;
  } else {
    ss << "Differences were found" << std::endl;
  }

  if (!errors_) {
    edm::LogVerbatim("SiStripDigiValidator") << ss.str();
  } else {
    edm::LogError("SiStripDigiValidator") << ss.str();
  }
}

void SiStripClusterValidator::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  if (dsvnew_) {
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > collection1Handle;
    event.getByLabel(collection1Tag_, collection1Handle);
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > collection2Handle;
    event.getByLabel(collection2Tag_, collection2Handle);
    validate(*collection1Handle, *collection2Handle);
  } else {
    edm::Handle<edm::DetSetVector<SiStripCluster> > collection1Handle;
    event.getByLabel(collection1Tag_, collection1Handle);
    edm::Handle<edm::DetSetVector<SiStripCluster> > collection2Handle;
    event.getByLabel(collection2Tag_, collection2Handle);
    validate(*collection1Handle, *collection2Handle);
  }
}

void SiStripClusterValidator::validate(const edmNew::DetSetVector<SiStripCluster>& collection1,
                                       const edmNew::DetSetVector<SiStripCluster>& collection2) {
  /// check number of DetSets

  if (collection1.size() != collection2.size()) {
    std::stringstream ss;
    ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl
       << header() << "Collection sizes do not match! (" << collection1.size() << " and " << collection2.size() << ")";
    edm::LogError("SiStripClusterValidator") << ss.str();
    errors_ = true;
    return;
  }

  /// loop over first collection DetSets comparing them to same DetSet in other collection

  edmNew::DetSetVector<SiStripCluster>::const_iterator iDetSet1 = collection1.begin();
  edmNew::DetSetVector<SiStripCluster>::const_iterator jDetSet1 = collection1.end();
  for (; iDetSet1 != jDetSet1; ++iDetSet1) {
    /// check that it exists

    edm::det_id_type id = iDetSet1->detId();
    edmNew::DetSetVector<SiStripCluster>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      std::stringstream ss;
      ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl
         << header() << "DetSet in collection 1 with id " << id << " is missing from collection 2!";
      edm::LogError("SiStripClusterValidator") << ss.str();
      errors_ = true;
      return;
    }

    /// check that the clusters are identical

    edmNew::DetSet<SiStripCluster>::const_iterator iCluster1 = iDetSet1->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator iCluster2 = iDetSet2->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator jCluster2 = iDetSet2->end();
    for (; iCluster2 != jCluster2; ++iCluster2) {
      if (std::equal(iCluster1->amplitudes().begin(), iCluster1->amplitudes().end(), iCluster2->amplitudes().begin()) &&
          iCluster1->firstStrip() == iCluster2->firstStrip())
        iCluster1++;
    }

    if (iCluster1 != iDetSet1->end()) {
      std::stringstream ss;
      ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl
         << header() << "No match for cluster in detector " << id << " with first strip number "
         << iCluster1->firstStrip();
      edm::LogError("SiStripClusterValidator") << ss.str();
      errors_ = true;
    }
  }
}

void SiStripClusterValidator::validate(const edm::DetSetVector<SiStripCluster>& collection1,
                                       const edm::DetSetVector<SiStripCluster>& collection2) {
  /// check number of DetSets

  if (collection1.size() != collection2.size()) {
    std::stringstream ss;
    ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl
       << header() << "Collection sizes do not match! (" << collection1.size() << " and " << collection2.size() << ")";
    errors_ = true;
    return;
  }

  /// loop over first collection DetSets comparing them to same DetSet in other collection

  edm::DetSetVector<SiStripCluster>::const_iterator iDetSet1 = collection1.begin();
  edm::DetSetVector<SiStripCluster>::const_iterator jDetSet1 = collection1.end();
  for (; iDetSet1 != jDetSet1; ++iDetSet1) {
    /// check that it exists

    edm::det_id_type id = iDetSet1->detId();
    edm::DetSetVector<SiStripCluster>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      std::stringstream ss;
      ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl
         << header() << "DetSet in collection 1 with id " << id << " is missing from collection 2!";
      edm::LogError("SiStripClusterValidator") << ss.str();
      errors_ = true;
      return;
    }

    /// check that the clusters are identical

    edm::DetSet<SiStripCluster>::const_iterator iCluster1 = iDetSet1->begin();
    edm::DetSet<SiStripCluster>::const_iterator iCluster2 = iDetSet2->begin();
    edm::DetSet<SiStripCluster>::const_iterator jCluster2 = iDetSet2->end();
    for (; iCluster2 != jCluster2; ++iCluster2) {
      if (std::equal(iCluster1->amplitudes().begin(), iCluster1->amplitudes().end(), iCluster2->amplitudes().begin()) &&
          iCluster1->firstStrip() == iCluster2->firstStrip())
        iCluster1++;
    }

    if (iCluster1 != iDetSet1->end()) {
      std::stringstream ss;
      ss << "[SiStripClusterValidator::" << __func__ << "]" << std::endl
         << header() << "No match for cluster in detector " << id << " with first strip number "
         << iCluster1->firstStrip();
      edm::LogError("SiStripClusterValidator") << ss.str();
      errors_ = true;
    }
  }
}

/// Debug for SiStripCluster collection

std::ostream& operator<<(std::ostream& ss, const edmNew::DetSetVector<SiStripCluster>& clusters) {
  edmNew::DetSetVector<SiStripCluster>::const_iterator ids = clusters.begin();
  edmNew::DetSetVector<SiStripCluster>::const_iterator jds = clusters.end();
  for (; ids != jds; ++ids) {
    std::stringstream sss;
    uint16_t nd = 0;
    edmNew::DetSet<SiStripCluster>::const_iterator id = ids->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator jd = ids->end();
    for (; id != jd; ++id) {
      nd++;
      if (uint16_t(id - ids->begin()) < 10) {
        sss << uint16_t(id - ids->begin()) << "/" << id->firstStrip() << "/" << id->amplitudes().size() << ", ";
      }
    }
    std::stringstream ss;
    ss << "[SiStripClusterValidator::" << __func__ << "]"
       << " DetId: " << ids->detId() << " size: " << ids->size() << " clusters: " << nd
       << " index/first-strip/width: " << sss.str() << std::endl;
    edm::LogError("SiStripClusterValidator") << ss.str();
  }
  return ss;
}

/// Debug for SiStripCluster collection

std::ostream& operator<<(std::ostream& ss, const edm::DetSetVector<SiStripCluster>& clusters) {
  edm::DetSetVector<SiStripCluster>::const_iterator ids = clusters.begin();
  edm::DetSetVector<SiStripCluster>::const_iterator jds = clusters.end();
  for (; ids != jds; ++ids) {
    std::stringstream sss;
    uint16_t nd = 0;
    edm::DetSet<SiStripCluster>::const_iterator id = ids->begin();
    edm::DetSet<SiStripCluster>::const_iterator jd = ids->end();
    for (; id != jd; ++id) {
      nd++;
      if (uint16_t(id - ids->begin()) < 10) {
        sss << uint16_t(id - ids->begin()) << "/" << id->firstStrip() << "/" << id->amplitudes().size() << ", ";
      }
    }
    std::stringstream ss;
    ss << "[SiStripClusterValidator::" << __func__ << "]"
       << " DetId: " << ids->detId() << " size: " << ids->size() << " clusters: " << nd
       << " index/first-strip/width: " << sss.str() << std::endl;
    edm::LogError("SiStripClusterValidator") << ss.str();
  }
  return ss;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterValidator);
