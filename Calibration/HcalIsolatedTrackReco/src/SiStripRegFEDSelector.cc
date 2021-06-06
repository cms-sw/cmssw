
#include "Calibration/HcalIsolatedTrackReco/interface/SiStripRegFEDSelector.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiStripRegFEDSelector::SiStripRegFEDSelector(const edm::ParameterSet& iConfig)
    : tok_seed_(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("regSeedLabel"))),
      tok_raw_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("rawInputLabel"))),
      tok_strip_(esConsumes<SiStripRegionCabling, SiStripRegionCablingRcd>()),
      delta_(iConfig.getParameter<double>("delta")) {
  produces<FEDRawDataCollection>();
}

SiStripRegFEDSelector::~SiStripRegFEDSelector() {}

void SiStripRegFEDSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto producedData = std::make_unique<FEDRawDataCollection>();

  edm::Handle<trigger::TriggerFilterObjectWithRefs> trigSeedTrks;
  iEvent.getByToken(tok_seed_, trigSeedTrks);

  std::vector<edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;
  trigSeedTrks->getObjects(trigger::TriggerTrack, isoPixTrackRefs);

  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByToken(tok_raw_, rawIn);

  const SiStripRegionCabling* strip_cabling = &iSetup.getData(tok_strip_);
  std::vector<int> stripFEDVec;

  //get vector of regions
  const SiStripRegionCabling::Cabling& ccab = strip_cabling->getRegionCabling();

  //size of region (eta,phi)
  const std::pair<double, double> regDim = strip_cabling->regionDimensions();

  const SiStripRegionCabling::ElementCabling elcabling;

  bool fedSaved[1000];
  for (int i = 0; i < 1000; i++)
    fedSaved[i] = false;

  //cycle on seeds
  for (uint32_t p = 0; p < isoPixTrackRefs.size(); p++) {
    double etaObj_ = isoPixTrackRefs[p]->track()->eta();
    double phiObj_ = isoPixTrackRefs[p]->track()->phi();

    //cycle on regions
    for (uint32_t i = 0; i < ccab.size(); i++) {
      SiStripRegionCabling::Position pos = strip_cabling->position(i);
      double dphi = fabs(pos.second - phiObj_);
      if (dphi > acos(-1))
        dphi = 2 * acos(-1) - dphi;
      double R = sqrt(pow(pos.first - etaObj_, 2) + dphi * dphi);
      if (R - sqrt(pow(regDim.first / 2, 2) + pow(regDim.second / 2, 2)) > delta_)
        continue;
      //get vector of subdets within region
      const SiStripRegionCabling::RegionCabling regSubdets = ccab[i];
      //cycle on subdets
      for (uint32_t idet = 0; idet < SiStripRegionCabling::ALLSUBDETS; idet++) {
        //get vector of layers within subdet of region
        const SiStripRegionCabling::WedgeCabling regSubdetLayers = regSubdets[idet];
        for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
          //get map of vectors of feds withing the layer of subdet of region
          const SiStripRegionCabling::ElementCabling fedVectorMap = regSubdetLayers[ilayer];
          SiStripRegionCabling::ElementCabling::const_iterator it = fedVectorMap.begin();
          for (; it != fedVectorMap.end(); it++) {
            for (uint32_t op = 0; op < (it->second).size(); op++) {
              //get fed id
              int fediid = (it->second)[op].fedId();
              if (!fedSaved[fediid]) {
                stripFEDVec.push_back(fediid);
              }
              fedSaved[fediid] = true;
            }
          }
        }
      }
    }
  }

  /////////////// Copying FEDs:

  const FEDRawDataCollection* rdc = rawIn.product();

  //   if ( ( rawData[i].provenance()->processName() != e.processHistory().rbegin()->processName() ) )
  //       continue ; // skip all raw collections not produced by the current process

  for (int j = 0; j < FEDNumbering::MAXFEDID; ++j) {
    bool rightFED = false;
    for (uint32_t k = 0; k < stripFEDVec.size(); k++) {
      if (j == stripFEDVec[k]) {
        rightFED = true;
      }
    }
    if (!rightFED)
      continue;
    const FEDRawData& fedData = rdc->FEDData(j);
    size_t size = fedData.size();

    if (size > 0) {
      // this fed has data -- lets copy it
      FEDRawData& fedDataProd = producedData->FEDData(j);
      if (fedDataProd.size() != 0) {
        edm::LogVerbatim("HcalIsoTrack") << " More than one FEDRawDataCollection with data in FED " << j
                                         << " Skipping the 2nd *****";
        continue;
      }
      fedDataProd.resize(size);
      unsigned char* dataProd = fedDataProd.data();
      const unsigned char* data = fedData.data();
      for (unsigned int k = 0; k < size; ++k) {
        dataProd[k] = data[k];
      }
    }
  }

  iEvent.put(std::move(producedData));
}

void SiStripRegFEDSelector::beginJob() {}

void SiStripRegFEDSelector::endJob() {}
