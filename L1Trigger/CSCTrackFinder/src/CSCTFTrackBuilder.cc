#include <L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>

#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>

void CSCTFTrackBuilder::initialize(const edm::EventSetup& c)
{
 //my_dtrc->initialize(c);
 	for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
	{
		for(int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
		{
			my_SPs[e-1][s-1]->initialize(c);
		}
	}
}

CSCTFTrackBuilder::CSCTFTrackBuilder(const edm::ParameterSet& pset)
{
  my_dtrc = new CSCTFDTReceiver();

  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId();
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  my_SPs[e-1][s-1] = new CSCTFSectorProcessor(e, s, pset);
	}
    }
  // All SPs work with the same configuration (impossible to make it more exclusive in this framework)
  run_core         = pset.getUntrackedParameter<bool>("run_core");
  trigger_on_ME1a  = pset.getUntrackedParameter<bool>("trigger_on_ME1a");
  trigger_on_ME1b  = pset.getUntrackedParameter<bool>("trigger_on_ME1b");
  trigger_on_ME2   = pset.getUntrackedParameter<bool>("trigger_on_ME2");
  trigger_on_ME3   = pset.getUntrackedParameter<bool>("trigger_on_ME3");
  trigger_on_ME4   = pset.getUntrackedParameter<bool>("trigger_on_ME4");
  trigger_on_MB1a  = pset.getUntrackedParameter<bool>("trigger_on_MB1a");
  trigger_on_MB1d  = pset.getUntrackedParameter<bool>("trigger_on_MB1d");
  singlesTrackRank = pset.getUntrackedParameter<unsigned int>("singlesTrackRank");
  singlesTrackOutput = pset.getUntrackedParameter<unsigned int>("singlesTrackOutput");
  lctMinBX         = pset.getUntrackedParameter<int>("lctMinBX",3);
  lctMaxBX         = pset.getUntrackedParameter<int>("lctMaxBX",9);
  trackMinBX       = pset.getUntrackedParameter<int>("trackMinBX",-3);
  trackMaxBX       = pset.getUntrackedParameter<int>("trackMaxBX",3);

}

CSCTFTrackBuilder::~CSCTFTrackBuilder()
{
  delete my_dtrc;
  my_dtrc = NULL;

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
				    L1CSCTrackCollection* trkcoll, CSCTriggerContainer<csctf::TrackStub>* stubs_to_dt)
{
  std::vector<csc::L1Track> trks;
  CSCTriggerContainer<csctf::TrackStub> stub_list;

  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;

  for(Citer = lcts->begin(); Citer != lcts->end(); Citer++)
    {
      CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
      CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

	  for(; Diter != Dend; Diter++)
	{
	  csctf::TrackStub theStub((*Diter),(*Citer).first);
	  stub_list.push_back(theStub);
	}
    }

  // Now we append the track stubs the the DT Sector Collector
  // after processing from the DT Receiver.
  stub_list.push_many(my_dtrc->process(dttrig));

  if(run_core){
    // run each sector processor in the TF
    for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
        for(int s = CSCTriggerNumbering::minTriggerSectorId();
          s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
        {
           CSCTriggerContainer<csctf::TrackStub> current_e_s = stub_list.get(e, s);
	       if(my_SPs[e-1][s-1]->run(current_e_s))
	       {
	         std::vector<csc::L1Track> theTracks = my_SPs[e-1][s-1]->tracks().get();
	         trks.insert(trks.end(), theTracks.begin(), theTracks.end());
	       }
	       stubs_to_dt->push_many(my_SPs[e-1][s-1]->dtStubs()); // send stubs whether or not we find a track!!!
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

      std::vector<csctf::TrackStub> possible_stubs = stub_list.get(titr->endcap(), titr->sector());
      std::vector<csctf::TrackStub>::const_iterator tkstbs = possible_stubs.begin();

      int me1ID = titr->me1ID();
      int me2ID = titr->me2ID();
      int me3ID = titr->me3ID();
      int me4ID = titr->me4ID();
      int mb1ID = titr->mb1ID();

      for(; tkstbs != possible_stubs.end(); tkstbs++)
        {
          switch(tkstbs->station())
            {
            case 1:
              if((tkstbs->getMPCLink()
                  +(3*(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(tkstbs->getDetId().rawId())) - 1))) == me1ID && me1ID != 0)
                {
                  tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
                }
              break;
	    case 2:
              if(tkstbs->getMPCLink() == me2ID && me2ID != 0)
                {
                  tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
                }
              break;
            case 3:
              if(tkstbs->getMPCLink() == me3ID && me3ID != 0)
                {
                  tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
                }
              break;
            case 4:
              if(tkstbs->getMPCLink() == me4ID && me4ID != 0)
                {
                  tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
                }
              break;
	    case 5:
	      if(tkstbs->getMPCLink() == mb1ID && mb1ID != 0)
	      {
		/// Hmmm how should I implement this??? Maybe change the L1Track to use stubs not LCTs?
	      }
	      break;
	    default:
	      edm::LogWarning("CSCTFTrackBuilder::buildTracks()") << "SERIOUS ERROR: STATION " << tkstbs->station() << " NOT IN RANGE [1,5]\n";
            };
	}
      tcitr++; // increment to next track in the collection
    }

    // Add-on for singles:
    CSCCorrelatedLCTDigiCollection myLCTcontainer[2][6][7/*lctMaxBX-lctMinBX*/]; //[endcap][sector][BX]

    for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=lcts->begin(); csc!=lcts->end(); csc++){
        CSCCorrelatedLCTDigiCollection::Range range = lcts->get((*csc).first);
        for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range.first; lct!=range.second; lct++){
           int endcap  = (*csc).first.endcap()-1;
           int sector  = (*csc).first.triggerSector()-1;
           int station = (*csc).first.station()-1;
           int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
           if( sector<0 || sector>6 || station<0 || station>3 || subSector<0 || subSector>2 || endcap<0 || endcap>1  ){
               edm::LogWarning("CSCTFTrackBuilder::buildTracks()")<<" CSC digi are out of range";
               continue;
           }
           int mpc = ( subSector ? subSector-1 : station+1 );
           if( (mpc==0&&trigger_on_ME1a) || (mpc==1&&trigger_on_ME1b) ||
               (mpc==2&&trigger_on_ME2)  || (mpc==3&&trigger_on_ME3)  ||
               (mpc==4&&trigger_on_ME4) ){
               int bx = lct->getBX() - lctMinBX;
               if( bx<0 || bx >= lctMaxBX-lctMinBX ) edm::LogWarning("CSCTFTrackBuilder::buildTracks()") << " LCT BX is out of ["<<lctMinBX<<","<<lctMaxBX<<") range: "<<lct->getBX();
               else
                 if( lct->isValid() ){
                     myLCTcontainer[endcap][sector][bx].put(range,(*csc).first);
                     break; //we break out of the loop because we put whole range if we encounter VP
                 }
           }
        }
    }

    // Now we put tracks from singles in a certain endcap/sector/bx only
    //   if there were no tracks from the core in this endcap/sector/bx
    L1CSCTrackCollection tracksFromSingles;
    for(unsigned int endcap=0; endcap<2; endcap++)
       for(unsigned int sector=0; sector<6; sector++)
          for(int bx=0; bx<lctMaxBX-lctMinBX; bx++)
             if( myLCTcontainer[endcap][sector][bx].begin() !=
                 myLCTcontainer[endcap][sector][bx].end() ){ // VP was detected in endcap/sector/bx
                bool coreTrackExists = false;
                // tracks are not ordered to be accessible by endcap/sector/bx => loop them all
                for(L1CSCTrackCollection::iterator trk=trkcoll->begin(); trk<trkcoll->end(); trk++)
                   if( trk->first.endcap()-1 == endcap &&
                       trk->first.sector()-1 == sector &&
                       trk->first.BX()-trackMinBX == bx && 
                       trk->first.outputLink() == singlesTrackOutput ){
                       coreTrackExists = true;
                       break;
                   }
                   if( coreTrackExists == false ){
                       csc::L1TrackId trackId(endcap+1,sector+1);
                       csc::L1Track   track(trackId);
                       track.setRank(singlesTrackRank);
                       track.setBx(bx+trackMinBX);
                       tracksFromSingles.push_back(L1CSCTrack(track,myLCTcontainer[endcap][sector][bx]));
                   } 
             }
    if( tracksFromSingles.size() )
       trkcoll->insert( trkcoll->end(), tracksFromSingles.begin(), tracksFromSingles.end() );
    // End of add-on for singles
}

