#include "IORawData/CaloPatterns/src/HcalPatternSource.h"
#include "IORawData/CaloPatterns/interface/HcalPatternXMLParser.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include <wordexp.h>
#include <cstdio>

HcalPatternSource::HcalPatternSource(const edm::ParameterSet& pset)
    : bunches_(pset.getUntrackedParameter<std::vector<int> >("Bunches", std::vector<int>())),
      presamples_(pset.getUntrackedParameter<int>("Presamples", 4)),
      samples_(pset.getUntrackedParameter<int>("Samples", 10)) {
  loadPatterns(pset.getUntrackedParameter<std::string>("Patterns"));
  produces<HBHEDigiCollection>();
  produces<HODigiCollection>();
  produces<HFDigiCollection>();
}

void HcalPatternSource::produce(edm::Event& e, const edm::EventSetup& es) {
  if (e.id().event() > bunches_.size())
    return;

  edm::ESHandle<HcalElectronicsMap> item;
  es.get<HcalElectronicsMapRcd>().get(item);
  const HcalElectronicsMap* elecmap = item.product();

  auto hbhe = std::make_unique<HBHEDigiCollection>();
  auto hf = std::make_unique<HFDigiCollection>();
  auto ho = std::make_unique<HODigiCollection>();

  int bc = bunches_[e.id().event() - 1];
  for (std::vector<HcalFiberPattern>::iterator i = patterns_.begin(); i != patterns_.end(); i++) {
    std::vector<HcalQIESample> samples;
    for (int fc = 0; fc < 3; fc++) {
      samples = i->getSamples(bc, presamples_, samples_, fc);
      HcalElectronicsId eid = i->getId(fc);

      HcalDetId did(elecmap->lookup(eid));

      if (did.null()) {
        edm::LogWarning("HCAL") << "No electronics map match for id " << eid;
        continue;
      }

      switch (did.subdet()) {
        case (HcalBarrel):
        case (HcalEndcap):
          hbhe->push_back(HBHEDataFrame(did));
          hbhe->back().setSize(samples_);
          hbhe->back().setPresamples(presamples_);
          for (int i = 0; i < samples_; i++)
            hbhe->back().setSample(i, samples[i]);
          hbhe->back().setReadoutIds(eid);
          break;
        case (HcalForward):
          hf->push_back(HFDataFrame(did));
          hf->back().setSize(samples_);
          hf->back().setPresamples(presamples_);
          for (int i = 0; i < samples_; i++)
            hf->back().setSample(i, samples[i]);
          hf->back().setReadoutIds(eid);
          break;
        case (HcalOuter):
          ho->push_back(HODataFrame(did));
          ho->back().setSize(samples_);
          ho->back().setPresamples(presamples_);
          for (int i = 0; i < samples_; i++)
            ho->back().setSample(i, samples[i]);
          ho->back().setReadoutIds(eid);
          break;
        default:
          continue;
      }
    }
  }
  hbhe->sort();
  ho->sort();
  hf->sort();

  e.put(std::move(hbhe));
  e.put(std::move(ho));
  e.put(std::move(hf));
}

void HcalPatternSource::loadPatterns(const std::string& patspec) {
  wordexp_t p;
  char** files;
  wordexp(patspec.c_str(), &p, WRDE_NOCMD);  // do not run shell commands!
  files = p.we_wordv;
  for (unsigned int i = 0; i < p.we_wordc; i++) {
    LogDebug("HCAL") << "Reading pattern file '" << files[i] << "'";
    loadPatternFile(files[i]);
    LogDebug("HCAL") << "Fibers so far " << patterns_.size();
  }
  wordfree(&p);
}

void HcalPatternSource::loadPatternFile(const std::string& filename) {
  HcalPatternXMLParser parser;
  std::string buffer, element;
  std::map<std::string, std::string> params;
  std::vector<uint32_t> data;
  FILE* f = fopen(filename.c_str(), "r");
  if (f == nullptr)
    return;
  else {
    char block[4096];
    while (!feof(f)) {
      int read = fread(block, 1, 4096, f);
      buffer.append(block, block + read);
    }
    fclose(f);
  }
  if (buffer.find("<?xml") != 0) {
    throw cms::Exception("InvalidFormat") << "Not a valid XML file: " << filename;
  }
  std::string::size_type i = 0, j;
  while (buffer.find("<CFGBrick>", i) != std::string::npos) {
    i = buffer.find("<CFGBrick>", i);
    j = buffer.find("</CFGBrick>", i);
    element = "<?xml version='1.0'?>\n";
    element.append(buffer, i, j - i);
    element.append("</CFGBrick>");
    //    LogDebug("HCAL") << element;
    params.clear();
    data.clear();
    parser.parse(element, params, data);
    patterns_.push_back(HcalFiberPattern(params, data));
    i = j + 5;
  }
}
