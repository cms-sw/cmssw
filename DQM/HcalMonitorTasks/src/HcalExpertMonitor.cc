#include "DQM/HcalMonitorTasks/interface/HcalExpertMonitor.h"
// define sizes of ieta arrays for each subdetector

#define PI        3.1415926535897932

using namespace std;
using namespace edm;

/*  

    v1.0
    24 September 2008
    by Jeff Temple

    Simple attempt at adding quick plots for expert-level HCAL debugging
*/


// constructor
HcalExpertMonitor::HcalExpertMonitor()
{}

// destructor
HcalExpertMonitor::~HcalExpertMonitor() {}

void HcalExpertMonitor::reset() {}

void HcalExpertMonitor::clearME()
{
  if (m_dbe) 
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    } // if (m_dbe)
  meEVT_=0;
} // void HcalExpertMonitor::clearME()


void HcalExpertMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);  // perform setups of base class

  ievt_=0; // event counter
  baseFolder_ = rootFolder_ + "ExpertMonitor"; // Will create an "ExpertMonitor" subfolder in .root output
  if (fVerbosity) cout <<"<HcalExpertMonitor::setup> Setup in progress"<<endl;

  
  if(fVerbosity) cout << "About to pushback fedUnpackList_" << endl;

  // Use this in CMSSW_2_2_X
  firstFED_ = FEDNumbering::getHcalFEDIds().first;   
  cout <<"FIRST FED = "<<firstFED_<<endl;
  for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++)

    // Use this in CMSSW_3_0_X and above
    //firstFED_ = FEDNumbering::MINHCALFEDID;
    //cout <<"FIRST FED = "<<firstFED_<<endl;

    //for (int i=FEDNumbering::MINHCALFEDID; 
    //   i<=FEDNumbering::MAXHCALFEDID; ++i) 
    {
      if(fVerbosity) cout << "<HcalExpertMonitor::setup>:Pushback for fedUnpackList_: " << i <<endl;
      fedUnpackList_.push_back(i);
    } // for (int i=FEDNumbering::MINHCALFEDID;...



  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      char* type;
      type = "ExpertMonitor Event Number";
      meEVT_ = m_dbe->bookInt(type); // store event number
    
      // Book Sample histogram
      SampleHist= m_dbe->book1D("sample1Dhist", "sample 1D histogram:  RecHit Energy vs. eta",90,-45,45);
      SampleHist2 = m_dbe->book2D("sample2Dhist","sample 2D histogram:  Digi occupancy",90,-45,45,74,-1,73);
      SampleHist2->setAxisTitle("i#eta",1);
      SampleHist2->setAxisTitle("i#phi",2);

    } // if (m_dbe)
  return;

} // void HcalExpertMonitor::setup()


void HcalExpertMonitor::processEvent(const HBHERecHitCollection& hbheHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits,
				     const HBHEDigiCollection& hbheDigis,
				     const HODigiCollection& hoDigis,
				     const HFDigiCollection& hfDigis,
				     const HcalTrigPrimDigiCollection& tpDigis,
				     const FEDRawDataCollection& rawraw,
				     const HcalUnpackerReport& report,
				     const HcalElectronicsMap& emap
				     )
  
{
  if (!m_dbe)
    {
      if (fVerbosity) cout <<"HcalExpertMonitor::processEvent   DQMStore not instantiated!!!"<<endl;
      return;
    }

  // Fill Event Number
  ievt_++;
  meEVT_->Fill(ievt_);

  processEvent_RecHit(hbheHits, hoHits, hfHits);
  processEvent_Digi(hbheDigis, hoDigis, hfDigis, tpDigis, emap);
  processEvent_RawData(rawraw, report, emap);
  return;
} // void HcalExpertMonitor::processEvent




////////////////////////////////////////////////////////////////////////////
//
//                         RECHIT DIAGNOSTICS
//
////////////////////////////////////////////////////////////////////////////



void HcalExpertMonitor::processEvent_RecHit(const HBHERecHitCollection& hbheHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits
				     )
  
{
  /*
    This processes RecHits
    Additional info on working with RecHits can be found in 
    HcalRecHitMonitor.cc
  */


  // Should never see this error message
  if (!m_dbe)
    {
      if (fVerbosity) cout <<"HcalExpertMonitor::processEvent_RecHit   DQMStore not instantiated!!!"<<endl;
      return;
    }

  // Set up iterators for looping over RecHits
  HBHERecHitCollection::const_iterator HBHEiter;
  HORecHitCollection::const_iterator HOiter;
  HFRecHitCollection::const_iterator HFiter;

  //////////////////////////////
  //  LOOP OVER HBHE REC HITS
  if (hbheHits.size()>0)
    {
      for (HBHEiter=hbheHits.begin(); 
	   HBHEiter!=hbheHits.end(); 
	   ++HBHEiter) 
	{ 
	  // rechit energy
	  double energy = HBHEiter->energy();
	  
	  // detector ID
	  HcalDetId id(HBHEiter->detid().rawId());
	  int ieta=id.ieta();
	  //int iphi=id.iphi();
	  SampleHist->Fill(ieta,energy);
	  if ((HcalSubdetector)(id.subdet())==HcalBarrel)
	    {
	      //cout <<"This is an HB hit"<<endl;
	    } // if id.subdet()==HcalBarrel

	  else
	    {
	      //cout <<"This is an HE hit"<<endl;
	    } // for (HBHEiter=hbheHits.begin()...

	} // for (HBHEiter=hbheHits.begin(),...)
    } // if (hbheHits.size()>0)

  else
    {
      if (fVerbosity) cout <<"HcalExpertMonitor::processEvent   No HBHE Rec Hits Found"<<endl;
    } 

  //////////////////////////////////
  //  LOOP OVER HO REC HITS
  if(hoHits.size()>0)
    {
      for (HOiter=hoHits.begin(); 
	   HOiter!=hoHits.end(); 
	   ++HOiter) 
	{ 
	  double energy = HOiter->energy();
	  HcalDetId id(HOiter->detid().rawId());
	  int ieta=id.ieta();
	  //int iphi=id.iphi();
	  SampleHist->Fill(ieta,energy);
	} // for (HOiter=hoHits.begin();...)
    } // if (hoHits.size()>0)  
  
  else
    {
      if (fVerbosity) cout <<"HcalExpertMonitor::processEvent   No HO RecHits found"<<endl;
    } // catch
  
  ///////////////////////////////////
  // LOOP OVER HF REC HITS

    if(hfHits.size()>0)
      {
	for (HFiter=hfHits.begin(); 
	     HFiter!=hfHits.end(); 
	     ++HFiter) 
	  { 
	    double energy =  HFiter->energy();
	    HcalDetId id(HFiter->detid().rawId());
	    int ieta=id.ieta();
	    //int iphi=id.iphi();
	    SampleHist->Fill(ieta,energy);
	  } // for (HFiter=hfHits.begin();...)
      } // if (hfHits.size()>0)	  
    else
      {
	if (fVerbosity) cout <<"HcalExpertMonitor::processEvent   No HF Rec Hits Found"<<endl;
      } // else
    
    return;
} // void HcalExpertMonitor::processEvent_RecHit(const HBHERecHit Collection&hbheHits; ...)



////////////////////////////////////////////////////////////////////////////
//
//                         DIGI DIAGNOSTICS
//
////////////////////////////////////////////////////////////////////////////

void HcalExpertMonitor::processEvent_Digi(const HBHEDigiCollection& hbheDigis,
					  const HODigiCollection& hoDigis,
					  const HFDigiCollection& hfDigis,
					  const HcalTrigPrimDigiCollection& tpDigis,
					  const HcalElectronicsMap& emap
					  )
{
  /*
    This processes Digis
    Additional info on working with Digis can be found in 
    HcalDigiMonitor.cc
  */


  // Should never see this error message
  if (!m_dbe)
    {
      if (fVerbosity) cout <<"HcalExpertMonitor::processEvent_Digi   DQMStore not instantiated!!!"<<endl;
      return;
    }


  // /////////////////////////////////////////
  // 
  //  LOOP OVER HBHE DIGIS
  //

  for (HBHEDigiCollection::const_iterator j=hbheDigis.begin(); j!=hbheDigis.end(); ++j)
    {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      int iEta = digi.id().ieta();
      int iPhi = digi.id().iphi();
      //int iDepth = digi.id().depth();
      SampleHist2->Fill(iEta,iPhi);
      if((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	{
	  //cout <<"This is an HB Digi"<<endl;
	}
      else
	{
	  //cout <<"This is an HE Digi"<<endl;
	}
    } // for (HBHEDigiCollection::const_iterator j=hbheDigis.begin()...


  // /////////////////////////////////////////
  // 
  //  LOOP OVER HO DIGIS
  //

  for (HODigiCollection::const_iterator j=hoDigis.begin(); j!=hoDigis.end(); ++j)
    {
      const HODataFrame digi = (const HODataFrame)(*j);
      int iEta = digi.id().ieta();
      int iPhi = digi.id().iphi();
      //int iDepth = digi.id().depth();
      SampleHist2->Fill(iEta,iPhi);
    } // for (HODigiCollection::const_iterator j=hoDigis.begin()...


  // /////////////////////////////////////////
  // 
  //  LOOP OVER HF DIGIS
  //

  for (HFDigiCollection::const_iterator j=hfDigis.begin(); j!=hfDigis.end(); ++j)
    {
      const HFDataFrame digi = (const HFDataFrame)(*j);
      int iEta = digi.id().ieta();
      int iPhi = digi.id().iphi();
      //int iDepth = digi.id().depth();
      SampleHist2->Fill(iEta,iPhi);
    } // for (HFDigiCollection::const_iterator j=hfDigis.begin()...

  return;
} // void HcalExpertMonitor::processEvent_Digi(const HBHEDigiCollection& hbheDigis...)


void HcalExpertMonitor::processEvent_RawData(const FEDRawDataCollection& rawraw,
					     const HcalUnpackerReport& report,
					     const HcalElectronicsMap& emap)
{
  /*
    This processes Raw Data.
    Additional info on working with Raw Data can be found in 
    HcalDataFormatMonitor.cc
  */
  

  // Should not see this error
  if(!m_dbe) 
    {
      cout <<"HcalExpertMonitor::processEvent_RawData:  DQMStore not instantiated!!!\n"<<endl;
      return;
    }

  // Loop over all FEDs reporting the event, unpacking if good.
  for (vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) 
    {
      const FEDRawData& fed = rawraw.FEDData(*i);
      if (fed.size()<12) continue; // Was 16.
      unpack(fed,emap);
    } // for (vector<int>::const_iterator i=fedUnpackList_.begin();...

  return;

} // void HcalExpertMonitor::processEvent_RawData(const FEDRawDataCollection& rawraw,








// Process one FED's worth (one DCC's worth) of the event data.
void HcalExpertMonitor::unpack(const FEDRawData& raw, 
			       const HcalElectronicsMap& emap)
{
  /* 
     This unpacks raw data info.  Additional info on working with the raw data can be found in the unpack method of HcalDataFormatMonitor.cc
  */


  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;

  // get the DCC trailer 
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  //DCC Event Fragment sizes distribution, in bytes.
  int rawsize=raw.size();

  int dccid=dccHeader->getSourceId();
  unsigned long dccEvtNum = dccHeader->getDCCEventNumber();
  int dccBCN = dccHeader->getBunchId();

  uint64_t* lastDataWord = (uint64_t*) ( raw.data()+raw.size()-(2*sizeof(uint64_t)) );
  int EvFragLength = ((*lastDataWord>>32)*8);
  EvFragLength = raw.size();

  // Dump out some raw data info
  cout <<"RAWSIZE = "<<rawsize<<endl;
  cout <<"dcc id = "<<dccid<<endl;
  cout <<"dccBCN = "<<dccBCN<<endl;
  cout <<"dccEvtNum = "<<dccEvtNum<<endl;
    cout <<"EvFragLength = "<<EvFragLength<<endl;
  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()) 
    {
    cout <<"No second CDF header found!"<<endl;
    }

  // walk through the HTR data...
  HcalHTRData htr;  
  bool isCM=false;
  unsigned char Chan24Thresh, Chan01Thresh;
  const short unsigned int* daq_first, *daq_last, *tp_first, *tp_last;
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
    if (!dccHeader->getSpigotPresent(spigot)) continue;

    // Load the given decoder with the pointer and length from this spigot.
    dccHeader->getSpigotData(spigot,htr, rawsize); 
    // Compact Mode? 
    isCM = ( ((htr.getExtHdr7()>>0x14) & 0x0001) == 1);
    if ( (htr.isUnsuppressed()) && !(isCM) ) {
      // get pointers
      htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);

      const unsigned short* HTRraw = htr.getRawData();
      unsigned short twoThresh = HTRraw[htr.getRawLength() -9];
      Chan24Thresh = (twoThresh>>8)&0x00FF;  //The high byte
      Chan01Thresh = (twoThresh   )&0x00FF;  //The low byte
      cout << "Spigot: " << dec << spigot << ": "<<hex << (int) Chan01Thresh << "to" << (int) Chan24Thresh << endl;
    }
  }

  return;
} // void HcalExpertMonitor::unpack(...)
