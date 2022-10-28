// system includes
#include <vector>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// ROOT includes
#include "TRandom.h"

/**
    @file EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialClusterSource.cc
    @class SiStripTrivialClusterSource
    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi/RawToCluster converters.  
 */

class SiStripTrivialClusterSource : public edm::stream::EDProducer<> {
public:
  SiStripTrivialClusterSource(const edm::ParameterSet&);
  ~SiStripTrivialClusterSource() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /** Check for space in module */
  bool available(const edm::DetSet<SiStripDigi>&, const uint16_t, const uint32_t);

  /** Add cluster to module */
  void addcluster(edm::DetSet<SiStripDigi>&, const uint16_t, const uint16_t);

  /** token */
  const edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> esTokenCabling_;

  /** Configurables */
  double minocc_;
  double maxocc_;
  double mincluster_;
  double maxcluster_;
  uint16_t separation_;

  /** Setup */
  edm::ESHandle<SiStripDetCabling> cabling_;
  std::vector<uint32_t> detids_;
  uint32_t nstrips_;

  /** Random */
  TRandom random_;
};

SiStripTrivialClusterSource::SiStripTrivialClusterSource(const edm::ParameterSet& pset)
    : esTokenCabling_(esConsumes<edm::Transition::BeginRun>()),
      minocc_(pset.getUntrackedParameter<double>("MinOccupancy", 0.001)),
      maxocc_(pset.getUntrackedParameter<double>("MaxOccupancy", 0.03)),
      mincluster_(pset.getUntrackedParameter<unsigned int>("MinCluster", 4)),
      maxcluster_(pset.getUntrackedParameter<unsigned int>("MaxCluster", 4)),
      separation_(pset.getUntrackedParameter<unsigned int>("Separation", 2)),
      cabling_(),
      detids_(),
      nstrips_(0),
      random_() {
  produces<edm::DetSetVector<SiStripDigi>>();
}

SiStripTrivialClusterSource::~SiStripTrivialClusterSource() = default;

void SiStripTrivialClusterSource::beginRun(const edm::Run&, const edm::EventSetup& setup) {
  cabling_ = setup.getHandle(esTokenCabling_);
  cabling_->addAllDetectorsRawIds(detids_);
  for (unsigned int i = 0; i < detids_.size(); i++) {
    nstrips_ += cabling_->getConnections(detids_[i]).size() * 256;
  }
}

void SiStripTrivialClusterSource::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto clusters = std::make_unique<edm::DetSetVector<SiStripDigi>>();

  double occupancy = random_.Uniform(minocc_, maxocc_);
  double indexdigis = nstrips_ * occupancy;
  double indexcluster = random_.Uniform(mincluster_, maxcluster_);
  uint32_t ndigis = (indexdigis > 0.) ? static_cast<uint32_t>(indexdigis) : 0;
  uint16_t clustersize = (indexcluster > 0.) ? static_cast<uint16_t>(indexcluster) : 0;

  uint32_t counter = 0;
  while (counter < 10000) {
    if (clustersize && ndigis >= clustersize)
      ndigis -= clustersize;
    else
      break;

    while (counter < 10000) {
      double indexdet = random_.Uniform(0., detids_.size());
      uint32_t detid = detids_[static_cast<uint32_t>(indexdet)];
      uint32_t maxstrip = 256 * cabling_->getConnections(detid).size();
      double indexstrip = random_.Uniform(0., maxstrip - clustersize);
      uint16_t strip = static_cast<uint16_t>(indexstrip);

      edm::DetSet<SiStripDigi>& detset = clusters->find_or_insert(detid);
      detset.data.reserve(768);

      if (available(detset, strip, clustersize)) {
        addcluster(detset, strip, clustersize);
        counter = 0;
        break;
      }

      counter++;
    }
  }

  iEvent.put(std::move(clusters));
}

bool SiStripTrivialClusterSource::available(const edm::DetSet<SiStripDigi>& detset,
                                            const uint16_t firststrip,
                                            const uint32_t size) {
  for (edm::DetSet<SiStripDigi>::const_iterator idigi = detset.data.begin(); idigi != detset.data.end(); idigi++) {
    if (idigi->strip() >= (firststrip - separation_) &&
        idigi->strip() < static_cast<int>(firststrip + size + separation_) && idigi->adc()) {
      return false;
    }
  }
  return true;
}

void SiStripTrivialClusterSource::addcluster(edm::DetSet<SiStripDigi>& detset,
                                             const uint16_t firststrip,
                                             const uint16_t size) {
  for (unsigned int istrip = 0; istrip < size; ++istrip) {
    detset.data.push_back(SiStripDigi(firststrip + istrip, 0xFF));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripTrivialClusterSource);
