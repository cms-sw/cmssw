#include <L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>

#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>

CSCTFTrackBuilder::CSCTFTrackBuilder(const edm::ParameterSet& pset)
{
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

void CSCTFTrackBuilder::buildTracks(const CSCCorrelatedLCTDigiCollection* lcts, const L1MuDTChambPhContainer* dttrig,
				    L1CSCTrackCollection* trkcoll, CSCTriggerContainer<CSCTrackStub>* stubs_to_dt)
{
  std::vector<csc::L1Track> trks;
  CSCTriggerContainer<CSCTrackStub> stub_list;

  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;

  for(Citer = lcts->begin(); Citer != lcts->end(); Citer++)
    {
      CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
      CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

      for(; Diter != Dend; Diter++)
	{
	  CSCTrackStub theStub((*Diter),(*Citer).first);
	  stub_list.push_back(theStub);	  
	}     
    }   

  // Now we append the track stubs the the DT Sector Collector
  // after processing from the DT Receiver.

  // stub_list.push_many(my_dtrc->process(dttrig));

  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  CSCTriggerContainer<CSCTrackStub> current_e_s = stub_list.get(e, s);
	  if(my_SPs[e-1][s-1]->run(current_e_s))
	    {
	      std::vector<csc::L1Track> theTracks = my_SPs[e-1][s-1]->tracks().get();	      
	      stubs_to_dt->push_many(my_SPs[e-1][s-1]->dtStubs());
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

      std::vector<CSCTrackStub> possible_stubs = stub_list.get(titr->endcap(), titr->sector());
      std::vector<CSCTrackStub>::const_iterator tkstbs = possible_stubs.begin();

      int me1ID = titr->me1ID();
      int me2ID = titr->me2ID();
      int me3ID = titr->me3ID();
      int me4ID = titr->me4ID();

      for(; tkstbs != possible_stubs.end(); tkstbs++)
        {
          switch(tkstbs->station())
            {
            case 1:
              if((tkstbs->getMPCLink()
                  +(3*(CSCTriggerNumbering::triggerSubSectorFromLabels(tkstbs->getDetId()) - 1))) == me1ID && me1ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), *(tkstbs->getDigi()));
                }
              break;
	    case 2:
              if(tkstbs->getMPCLink() == me2ID && me2ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), *(tkstbs->getDigi()));
                }
              break;
            case 3:
              if(tkstbs->getMPCLink() == me3ID && me3ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), *(tkstbs->getDigi()));
                }
              break;
            case 4:
              if(tkstbs->getMPCLink() == me4ID && me4ID != 0)
                {
                  tcitr->second.insertDigi(tkstbs->getDetId(), *(tkstbs->getDigi()));
                }
              break;
	    default:
	      edm::LogWarning("CSCTFSectorProcessor::run()") << "SERIOUS ERROR: STATION" << tkstbs->station() << "NOT IN RANGE [1,4]\n";
            };
	}
      tcitr++; // increment to next track in the collection
    }
}

