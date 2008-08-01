#include <L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>

#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>

#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <sstream>
#include <stdlib.h>

CSCTFTrackBuilder::CSCTFTrackBuilder(const edm::ParameterSet& pset, bool TMB07,
				     const L1MuTriggerScales* scales,
				     const L1MuTriggerPtScale* ptScale ){
  my_dtrc = new CSCTFDTReceiver();

  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId();
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
          // All SPs work with the same configuration (impossible to make it more exclusive in this framework)
	  my_SPs[e-1][s-1] = new CSCTFSectorProcessor(e, s, pset, TMB07,
						      scales, ptScale);
	}
    }
  // Uninitialize following parameters: 
  run_core = -1;
  trigger_on_ME1a = -1;
  trigger_on_ME1b = -1;
  trigger_on_ME2  = -1;
  trigger_on_ME3  = -1;
  trigger_on_ME4  = -1;
  trigger_on_MB1a = -1;
  trigger_on_MB1d = -1;
  singlesTrackPt  = -1;
  singlesTrackOutput = -1;

  try {
    run_core = pset.getParameter<bool>("run_core");
    LogDebug("CSCTFTrackBuilder") << "Using run_core configuration parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for run_core parameter in EventSetup";
  }
  
  try {
    trigger_on_ME1a = pset.getParameter<bool>("trigger_on_ME1a");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_ME1a parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_ME1a parameter in EventSetup";
  }
  
  try {
    trigger_on_ME1b = pset.getParameter<bool>("trigger_on_ME1b");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_ME1b parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_ME1b parameter in EventSetup";
  }
  
  try {
    trigger_on_ME2 = pset.getParameter<bool>("trigger_on_ME2");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_ME2 parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_ME2 parameter in EventSetup";
  }
  
  try {
    trigger_on_ME3 = pset.getParameter<bool>("trigger_on_ME3");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_ME3 parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_ME3 parameter in EventSetup";
  }
  
  try {
    trigger_on_ME4 = pset.getParameter<bool>("trigger_on_ME4");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_ME4 parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_ME4 parameter in EventSetup";
  }
  
  try {
    trigger_on_MB1a = pset.getParameter<bool>("trigger_on_MB1a");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_MB1a parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_MB1a parameter in EventSetup";
  }
  
  try {
    trigger_on_MB1d = pset.getParameter<bool>("trigger_on_MB1d");
    LogDebug("CSCTFTrackBuilder") << "Using trigger_on_MB1d parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for trigger_on_MB1d parameter in EventSetup";
  }
  
  try {
    singlesTrackPt = pset.getParameter<unsigned int>("singlesTrackPt");
    LogDebug("CSCTFTrackBuilder") << "Using singlesTrackPt parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for singlesTrackPt parameter in EventSetup";
  }
  
  try {
    singlesTrackOutput = pset.getParameter<unsigned int>("singlesTrackOutput");
    LogDebug("CSCTFTrackBuilder") << "Using singlesTrackOutput parameter from .cfi file";
  } catch(...) {
    LogDebug("CSCTFTrackBuilder") << "Looking for singlesTrackOutput parameter in EventSetup";
  }
  
  m_minBX = pset.getParameter<int>("MinBX");
  m_maxBX = pset.getParameter<int>("MaxBX");
  if( m_maxBX-m_minBX >= 7 ) edm::LogWarning("CSCTFTrackBuilder::ctor")<<" BX window width >= 7BX. Resetting m_maxBX="<<(m_maxBX=m_minBX+6);
}

void CSCTFTrackBuilder::initialize(const edm::EventSetup& c){
 //my_dtrc->initialize(c);
 	for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
	{
		for(int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
		{
			my_SPs[e-1][s-1]->initialize(c);
		}
	}

  edm::ESHandle<L1MuCSCTFConfiguration> config;
  c.get<L1MuCSCTFConfigurationRcd>().get(config);
  std::stringstream conf(config.product()->parameters());
  while( !conf.eof() ){
    char buff[1024];
    conf.getline(buff,1024);
    std::stringstream line(buff);

    std::string register_;     line>>register_;
    std::string chip_;         line>>chip_;
    std::string muon_;         line>>muon_;
    std::string writeValue_;   line>>writeValue_;
    std::string comments_;     std::getline(line,comments_);

    if( register_=="CSR_REQ" && chip_=="SP" ){
        unsigned int value = strtol(writeValue_.c_str(),'\0',16);
        // Initializeing in constructor from .cf? file always have priority over EventSetup
        if(run_core<0)        run_core         = value&0x8000;
        if(trigger_on_ME1a<0) trigger_on_ME1a  = value&0x0001;
        if(trigger_on_ME1b<0) trigger_on_ME1b  = value&0x0002;
        if(trigger_on_ME2 <0) trigger_on_ME2   = value&0x0004;
        if(trigger_on_ME3 <0) trigger_on_ME3   = value&0x0008;
        if(trigger_on_ME4 <0) trigger_on_ME4   = value&0x0010;
        if(trigger_on_MB1a<0) trigger_on_MB1a  = value&0x0100;
        if(trigger_on_MB1d<0) trigger_on_MB1d  = value&0x0200;
    }
    if( register_=="DAT_FTR" && chip_=="SP" ){
        unsigned int value = strtol(writeValue_.c_str(),'\0',16);
        // Initializeing in constructor from .cf? file always have priority over EventSetup
        if(singlesTrackPt<0) singlesTrackPt = value; // 0x1F - rank, 0x60 - Q1,Q0, 0x80 - charge
    }
    if( register_=="CSR_SFC" && chip_=="SP" ){
        unsigned int value = strtol(writeValue_.c_str(),'\0',16);
        // Initializeing in constructor from .cf? file always have priority over EventSetup
        if(singlesTrackOutput<0) singlesTrackOutput = (value&0x3000)>>12;
    }
  }
  // Check if parameters were not initialized in both: constuctor (from .cf? file) and initialize method (from EventSetup)
  if(run_core       <0) throw cms::Exception("CSCTFTrackBuilder")<<"run_core parameter left uninitialized";
  if(trigger_on_ME1a<0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_ME1a parameter left uninitialized";
  if(trigger_on_ME1b<0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_ME1b parameter left uninitialized";
  if(trigger_on_ME2 <0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_ME2 parameter left uninitialized";
  if(trigger_on_ME3 <0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_ME3 parameter left uninitialized";
  if(trigger_on_ME4 <0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_ME4 parameter left uninitialized";
  if(trigger_on_MB1a<0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_MB1a parameter left uninitialized";
  if(trigger_on_MB1d<0) throw cms::Exception("CSCTFTrackBuilder")<<"trigger_on_MB1d parameter left uninitialized";
  if( trigger_on_ME1a>0 || trigger_on_ME1b>0 ||trigger_on_ME2>0  ||
      trigger_on_ME3>0  || trigger_on_ME4>0  ||trigger_on_MB1a>0 ||trigger_on_MB1d>0 ){
      if(singlesTrackPt<0) throw cms::Exception("CSCTFTrackBuilder")<<"singlesTrackPt parameter left uninitialized";
      if(singlesTrackOutput<0) throw cms::Exception("CSCTFTrackBuilder")<<"singlesTrackOutput parameter left uninitialized";
  }
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

  CSCTriggerContainer<csctf::TrackStub> dtstubs = my_dtrc->process(dttrig);
  stub_list.push_many(dtstubs);

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
    CSCCorrelatedLCTDigiCollection myLCTcontainer[2][6][7]; //[endcap][sector][BX]
    // Loop over CSC LCTs if triggering on them:
    if( trigger_on_ME1a || trigger_on_ME1b || trigger_on_ME2 || trigger_on_ME3 || trigger_on_ME4 ){
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
///std::cout<<"Found LCT in endcap="<<endcap<<" sector="<<sector<<" station="<<station<<" subSector="<<subSector<<std::endl;
              int mpc = ( subSector ? subSector-1 : station+1 );
              if( (mpc==0&&trigger_on_ME1a) || (mpc==1&&trigger_on_ME1b) ||
                  (mpc==2&&trigger_on_ME2)  || (mpc==3&&trigger_on_ME3)  ||
                  (mpc==4&&trigger_on_ME4) ){
                  int bx = lct->getBX() - m_minBX;
                  if( bx<0 || bx>=7 ) edm::LogWarning("CSCTFTrackBuilder::buildTracks()") << " LCT BX is out of ["<<m_minBX<<","<<m_maxBX<<") range: "<<lct->getBX();
                  else
                     if( lct->isValid() ){
                         myLCTcontainer[endcap][sector][bx].put(range,(*csc).first);
                         break; //we break out of the loop because we put whole range if we encounter VP
                     }
              }
           }
       }
    }
    // Loop over DT stubs if triggering on them: 
    if( trigger_on_MB1a || trigger_on_MB1d ){
       std::vector<csctf::TrackStub> _dtstubs = dtstubs.get();
       for(std::vector<csctf::TrackStub>::const_iterator stub=_dtstubs.begin(); stub!=_dtstubs.end(); stub++){
          int endcap    = stub->endcap() -1;
          int sector    = stub->sector() -1;
          int station   = stub->station()-1;
          int subSector = stub->subsector();
///std::cout<<"Found DT  in endcap="<<endcap<<" sector="<<sector<<" station="<<station<<" subSector="<<subSector<<" mpc="<<stub->getMPCLink()<<std::endl;
          if( sector<0 || sector>6 || station<4 || station>8 || subSector<0 || subSector>12 || endcap<0 || endcap>1 ){
              edm::LogWarning("CSCTFTrackBuilder::buildTracks()")<<" DT digi are out of range";
              continue;
          }
          const CSCCorrelatedLCTDigi *lct = stub->getDigi(); //this is not a real LCT, but just a representation of DT information
          if( (trigger_on_MB1a && subSector%2==1) || //MB1a and MB1b may be swaped here!
              (trigger_on_MB1d && subSector%2==0) ){
             int bx = lct->getBX() - m_minBX;
             if( bx<0 || bx>=7 ) edm::LogWarning("CSCTFTrackBuilder::buildTracks()") << " DT stub BX is out of ["<<m_minBX<<","<<m_maxBX<<") range: "<<lct->getBX();
             else
                if( lct->isValid() )
                   //Construct fake CSCDetId (impossible to put DTDetId in CSC containers)
                   myLCTcontainer[endcap][sector][bx].insertDigi(CSCDetId(endcap,station-4+1,subSector%2+1,0),*lct);
          }
       }
    }

     // Core's input was loaded in a relative time window BX=[0-7)
     // To relate it to time window of tracks (centred at BX=0) we introduce a shift:
     int shift = (m_maxBX + m_minBX)/2 - m_minBX;

    // Now we put tracks from singles in a certain endcap/sector/bx only
    //   if there were no tracks from the core in this endcap/sector/bx
    L1CSCTrackCollection tracksFromSingles;
    for(unsigned int endcap=0; endcap<2; endcap++)
       for(unsigned int sector=0; sector<6; sector++)
          for(int bx=0; bx<7; bx++)
             if( myLCTcontainer[endcap][sector][bx].begin() !=
                 myLCTcontainer[endcap][sector][bx].end() ){ // VP was detected in endcap/sector/bx
                bool coreTrackExists = false;
                // tracks are not ordered to be accessible by endcap/sector/bx => loop them all
                for(L1CSCTrackCollection::iterator trk=trkcoll->begin(); trk<trkcoll->end(); trk++)
                   if( trk->first.endcap()-1 == endcap &&
                       trk->first.sector()-1 == sector &&
                       trk->first.BX()       == bx-shift && 
                       trk->first.outputLink() == singlesTrackOutput ){
                       coreTrackExists = true;
                       break;
                   }
                   if( coreTrackExists == false ){
                       csc::L1TrackId trackId(endcap+1,sector+1);
                       csc::L1Track   track(trackId);
                       track.setRank(singlesTrackPt&0x1F);
                       track.setBx(bx-shift);
                       //track.setPtPacked(singlesTrackPt);
                       //track.setQualityPacked(singlesTrackPt);
                       //track.setChargeValidPacked();
                       tracksFromSingles.push_back(L1CSCTrack(track,myLCTcontainer[endcap][sector][bx]));
                   } 
             }
    if( tracksFromSingles.size() )
       trkcoll->insert( trkcoll->end(), tracksFromSingles.begin(), tracksFromSingles.end() );
    // End of add-on for singles
}

