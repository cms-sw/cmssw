/** \class CruzetL1DTfilter
*
*  
*  This class is for use with Cosmic MC to simulate the L1 DT trigger in cruzet
*
*
*  \author Ivan Mikulec
*
*/
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class CruzetL1DTfilter : public edm::global::EDFilter<> {
public:
  explicit CruzetL1DTfilter(const edm::ParameterSet&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const;

private:
  // mode:
  // 1 - bottom only
  // 2 - top only
  // 3 - top or bottom
  // 4 - top and bottom
  int _mode;
  edm::InputTag _GMTInputTag;
};

CruzetL1DTfilter::CruzetL1DTfilter(const edm::ParameterSet& ps) {
  _mode = ps.getParameter<int>("mode");
  _GMTInputTag = ps.getParameter<edm::InputTag>("GMTInputTag");
}

bool CruzetL1DTfilter::filter(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const {
  edm::Handle<L1MuGMTReadoutCollection> gmtrc;
  e.getByLabel(_GMTInputTag, gmtrc);

  bool result = false;
  L1MuGMTReadoutRecord gmtrr = gmtrc->getRecord();

  std::vector<L1MuRegionalCand>::const_iterator iter1;
  std::vector<L1MuRegionalCand> rmc = gmtrr.getDTBXCands();
  bool top = false;
  bool bot = false;
  for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
    int phi = (*iter1).phi_packed();
    if (phi >= 18 && phi <= 53)
      top = true;
    if (phi >= 90 && phi <= 125)
      bot = true;
  }
  if (_mode == 1 && bot)
    result = true;
  if (_mode == 2 && top)
    result = true;
  if (_mode == 3 && (top || bot))
    result = true;
  if (_mode == 4 && (top && bot))
    result = true;

  return result;
}

DEFINE_FWK_MODULE(CruzetL1DTfilter);
