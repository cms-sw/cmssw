#include "L1Trigger/CSCTriggerPrimitives/src/CSCDigiSuppressor.h"
#include "DataFormats/Common/interface/Handle.h"
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


CSCDigiSuppressor::CSCDigiSuppressor(const edm::ParameterSet& ps)
: theLCTTag(ps.getParameter<edm::InputTag>("lctTag")),
  theStripDigiTag(ps.getParameter<edm::InputTag>("stripDigiTag"))
{
  produces<CSCStripDigiCollection>("MuonCSCSuppressedStripDigi");
}


void CSCDigiSuppressor::produce(edm::Event& e, const edm::EventSetup& eventSetup) 
{
  edm::Handle<CSCStripDigiCollection> oldStripDigis;
  e.getByLabel(theStripDigiTag, oldStripDigis);
  if (!oldStripDigis.isValid()) {
    edm::LogError("CSCDigiValidation") << "Cannot get strips by label "
                                       << theStripDigiTag.encode();
  }

  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  e.getByLabel(theLCTTag, lcts);

  std::auto_ptr<CSCStripDigiCollection> newStripDigis(new CSCStripDigiCollection());


  for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitItr = lcts->begin();
      detUnitItr != lcts->end(); ++detUnitItr)
  {
    const CSCDetId& id = (*detUnitItr).first;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitItr).second;
    std::list<int> keyStrips;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiItr = range.first;
         digiItr != range.second; digiItr++) 
    {
      // convert from 0..159 to 1..80
      keyStrips.push_back(digiItr->getStrip()/2+1);
    }

    fillDigis(id, keyStrips, *oldStripDigis, *newStripDigis);
  }

  e.put(newStripDigis, "MuonCSCSuppressedStripDigi");
}


void CSCDigiSuppressor::fillDigis(const CSCDetId & id, const std::list<int> & keyStrips, 
                                       const CSCStripDigiCollection & oldStripDigis,
                                       CSCStripDigiCollection & newStripDigis)
{
  std::list<int> cfebs = cfebsToRead(id, keyStrips);
  CSCStripDigiCollection::Range chamberDigis = oldStripDigis.get(id);
  // strips are sorted by layer
  for(int layer = 1; layer <= 6; ++layer)
  {
    CSCDetId layerId(id.rawId()+layer);
    for(CSCStripDigiCollection::const_iterator digiItr = chamberDigis.first;
        digiItr != chamberDigis.second; ++digiItr)
    {
      int cfeb = (digiItr->getStrip()-1)/16; 
      if(std::find(cfebs.begin(), cfebs.end(), cfeb) != cfebs.end())
      {
        newStripDigis.insertDigi(layerId, *digiItr);
      }
    }
  }
}


std::list<int>
CSCDigiSuppressor::cfebsToRead(const CSCDetId & id, const std::list<int> & keyStrips) const
{
  // always accept ME1A, because it's too much trouble looking
  // for LCTs in ME11
  if(id.station() == 1 && id.ring() == 4)
  {
    return std::list<int>(1, 0.);
  }

  int maxCFEBs = (id.station() == 1) ? 4 : 5;
  if(id.station() == 1 && id.ring() == 2) maxCFEBs = 5;

  //copied from CSCStripElectronicsSim
  std::list<int> cfebs;
  for(std::list<int>::const_iterator keyStripItr = keyStrips.begin(); 
      keyStripItr != keyStrips.end(); ++keyStripItr)
  {
    int cfeb = ((*keyStripItr)-1)/16;
    cfebs.push_back(cfeb);
    int remainder = ((*keyStripItr)-1)%16;
    // if we're within 3 strips of an edge, take neighboring CFEB, too
    if(remainder <= 2 && cfeb != 0)
    {
      cfebs.push_back(cfeb-1);
    }

    if(remainder >= 13 && cfeb < maxCFEBs) 
    {
      cfebs.push_back(cfeb+1);
    }
  }
  cfebs.sort();
  cfebs.unique();
  return cfebs;
}



