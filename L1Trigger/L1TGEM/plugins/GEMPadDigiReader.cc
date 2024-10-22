/** \class GEMDigiReader
 *  Dumps GEM trigger pad digis
 *
 *  \authors: Vadim Khotilovich
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

#include <map>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace std;

class GEMPadDigiReader : public edm::one::EDAnalyzer<> {
public:
  explicit GEMPadDigiReader(const edm::ParameterSet& pset);

  ~GEMPadDigiReader() override {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken_;
  edm::EDGetTokenT<GEMPadDigiCollection> gemPadToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
};

GEMPadDigiReader::GEMPadDigiReader(const edm::ParameterSet& pset)
    : gemDigiToken_(consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("gemDigiToken"))),
      gemPadToken_(consumes<GEMPadDigiCollection>(pset.getParameter<edm::InputTag>("gemPadToken"))),
      geomToken_(esConsumes<GEMGeometry, MuonGeometryRecord>()) {}

void GEMPadDigiReader::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  //cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::ESHandle<GEMGeometry> geometry = eventSetup.getHandle(geomToken_);

  edm::Handle<GEMDigiCollection> digis;
  event.getByToken(gemDigiToken_, digis);

  edm::Handle<GEMPadDigiCollection> pads;
  event.getByToken(gemPadToken_, pads);

  if (pads->begin() == pads->end())
    return;  // no pads in event

  for (auto pad_range_it = pads->begin(); pad_range_it != pads->end(); ++pad_range_it) {
    const auto& id = (*pad_range_it).first;
    const auto& roll = geometry->etaPartition(id);

    // GEMDetId print-out
    cout << "--------------" << endl;
    //cout<<"id: "<<id.rawId()<<" #strips "<<roll->nstrips()<<"  #pads "<<roll->npads()<<endl;

    // retrieve this DetUnit's digis
    std::map<std::pair<int, int>,  // #pad (starting from 1), BX
             std::vector<int>      // digi strip numbers (starting from 1)
             >
        digi_map;
    auto digis_in_det = digis->get(id);
    cout << "strip digis in detid: ";
    for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
      int pad_num = 1 + static_cast<int>(roll->padOfStrip(d->strip()));  // d->strip() is int
      digi_map[{pad_num, d->bx()}].push_back(d->strip());
      cout << "  (" << d->strip() << "," << d->bx() << ") -> " << pad_num;
    }
    cout << endl;

    // loop over pads of this DetUnit and print stuff
    auto pads_range = (*pad_range_it).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p) {
      int first_strip = roll->firstStripInPad(p->pad());  // p->pad() is int, firstStripInPad returns int
      int last_strip = roll->lastStripInPad(p->pad());

      if (p->pad() < 1 || p->pad() > roll->npads()) {
        cout << " XXXXXXXXXXXXX Problem! " << id << " has pad digi with too large pad# = " << p->pad() << endl;
      }

      auto& strips = digi_map[{p->pad(), p->bx()}];
      std::vector<int> pads_strips;
      remove_copy_if(
          strips.begin(), strips.end(), inserter(pads_strips, pads_strips.end()), [first_strip, last_strip](int s) {
            return s < first_strip || s > last_strip;
          });
      cout << id << " paddigi(pad,bx) " << *p << "   has " << pads_strips.size() << " strip digis strips in range ["
           << first_strip << "," << last_strip << "]: ";
      copy(pads_strips.begin(), pads_strips.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

  }  // for (detids with pads)
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMPadDigiReader);
