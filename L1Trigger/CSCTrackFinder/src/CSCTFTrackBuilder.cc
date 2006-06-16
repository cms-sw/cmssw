#include <L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerContainer.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCMuonPortCard.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>

CSCTFTrackBuilder::CSCTFTrackBuilder(const edm::ParameterSet& pset)
{
  m_muonportcard = new CSCMuonPortCard();

  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  my_SPs[e-1][s-1] = new CSCTFSectorProcessor(e, s, pset);
	}
    }
}

CSCTFTrackBuilder::~CSCTFTrackBuilder()
{
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  delete my_SPs[e-1][s-1];
	  my_SPs[e-1][s-1] = NULL;
	}
    }
}

void CSCTFTrackBuilder::buildTracks(const CSCCorrelatedLCTDigiCollection* lcts, L1CSCTrackCollection* trkcoll)
{
  std::vector<csc::L1Track> trks;
  std::vector<CSCTrackStub> stub_list;
  int m_minBX = my_SPs[0][0]->minBX(), m_maxBX = my_SPs[0][0]->maxBX();
  // we have to sort again due since we can't save the sorting across producer boundaries (FIX THIS!)

  m_muonportcard->loadDigis(*lcts);
  
  for(int bx = m_minBX; bx <= m_maxBX; ++bx)
    for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
      for(int st = CSCDetId::minStationId(); st <= CSCDetId::maxStationId(); ++st)
	for(int se = CSCTriggerNumbering::minTriggerSectorId(); 
	    se <= CSCTriggerNumbering::maxTriggerSectorId(); ++se)
	  {
	    if(st == 1)
	      {
		std::vector<CSCTrackStub> subs1, subs2;
		subs1 = m_muonportcard->sort(e, st, se, CSCTriggerNumbering::minTriggerSubSectorId(), bx);
		subs2 = m_muonportcard->sort(e, st, se, CSCTriggerNumbering::maxTriggerSubSectorId(), bx);
		stub_list.insert(stub_list.end(), subs1.begin(), subs1.end());
		stub_list.insert(stub_list.end(), subs2.begin(), subs2.end());
	      }
	    else
	      {
		std::vector<CSCTrackStub> sector;
		sector = m_muonportcard->sort(e, st, se, 0, bx);
		stub_list.insert(stub_list.end(), sector.begin(), sector.end());
	      }
	  }

  CSCTriggerContainer<CSCTrackStub> the_stubs(stub_list);
      
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  CSCTriggerContainer<CSCTrackStub> current_e_s = the_stubs.get(e, s);
	  if(my_SPs[e-1][s-1]->run(current_e_s))
	    {
	      std::vector<csc::L1Track> theTracks = my_SPs[e-1][s-1]->tracks().get();	      
	      trks.insert(trks.end(), theTracks.begin(), theTracks.end());
	    }
	}
    }

  // Now to combine tracks with their track stubs and send them off.
  trkcoll->resize(trks.size());
  std::vector<csc::L1Track>::const_iterator titr = trks.begin();
  L1CSCTrackCollection::iterator tcitr = trkcoll->begin();

  for(; titr != trks.end(); titr++)
    {      
      tcitr->first = (*titr);

      std::vector<CSCTrackStub> possible_stubs = the_stubs.get(titr->endcap(), titr->sector());
      std::vector<CSCTrackStub>::const_iterator tkstbs = possible_stubs.begin();

      unsigned me1ID = titr->me1ID();
      unsigned me2ID = titr->me2ID();
      unsigned me3ID = titr->me3ID();
      unsigned me4ID = titr->me4ID();

      for(; tkstbs != possible_stubs.end(); tkstbs++)
        {
          switch(tkstbs->station())
            {
            case 1:
              if((tkstbs->getMPCLink()
                  +(3*(CSCTriggerNumbering::triggerSubSectorFromLabels(tkstbs->getDetId()) - 1))) == me1ID && me1ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), tkstbs->getDigi());
                }
              break;
	    case 2:
              if(tkstbs->getMPCLink() == me2ID && me2ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), tkstbs->getDigi());
                }
              break;
            case 3:
              if(tkstbs->getMPCLink() == me3ID && me3ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), tkstbs->getDigi());
                }
              break;
            case 4:
              if(tkstbs->getMPCLink() == me4ID && me4ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), tkstbs->getDigi());
                }
              break;
	    default:
	      edm::LogWarning("CSCTFSectorProcessor::run()") << "SERIOUS ERROR: STATION" << tkstbs->station() << "NOT IN RANGE [1,4]\n";
            };
	}
      tcitr++; // increment to next track in the collection
    }
}

