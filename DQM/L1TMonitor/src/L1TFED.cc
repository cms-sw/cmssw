/*
 * \file L1TFED.cc
 *
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TFED.h"

using namespace std;
using namespace edm;

L1TFED::L1TFED(const ParameterSet& ps)
{
  // verbosity switch
  verbose_        = ps.getUntrackedParameter<bool>("verbose", false);
  l1feds_         = ps.getParameter<std::vector<int> >("L1FEDS");
  directory_      = ps.getUntrackedParameter<std::string>("FEDDirName","L1T/FEDIntegrity");
  stableROConfig_ = ps.getUntrackedParameter<bool>("stableROConfig", true);
  rawl_           = consumes<FEDRawDataCollection>(ps.getParameter< InputTag >("rawTag"));

  if(verbose_) cout << "L1TFED: constructor...." << endl;
  nev_ = 0;
}

L1TFED::~L1TFED()
{
}

void L1TFED::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & iSetup) 
{
  ibooker.setCurrentFolder(directory_);
  
  fedentries = ibooker.book1D("FEDEntries", "Fed ID occupancy", l1feds_.size(), 0.,l1feds_.size() );	  
  fedfatal = ibooker.book1D("FEDFatal", "Fed ID non present ", l1feds_.size(), 0., l1feds_.size());	  
  fednonfatal = ibooker.book1D("FEDNonFatal", "Fed corrupted data ", l1feds_.size(), 0.,l1feds_.size() );
  hfedprof = ibooker.bookProfile("fedprofile","FED Size by ID", l1feds_.size(), 0., l1feds_.size(),0,0.,10000.);
  for(unsigned int i=0;i<l1feds_.size();i++){
    ostringstream sfed;
    sfed << l1feds_[i];
    fedentries->setBinLabel(i+1,"FED "+ sfed.str());
    fedfatal->setBinLabel(i+1,"FED "+ sfed.str());
    fednonfatal->setBinLabel(i+1,"FED "+ sfed.str());
    hfedprof->setBinLabel(i+1,"FED "+ sfed.str());
  }
  
  hfedsize = ibooker.book1D("fedsize","FED Size Distribution",100,0.,10000.);
}


void L1TFED::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1T FED Integrity: analyze...." << endl;

  edm::Handle<FEDRawDataCollection> rawdata;
  bool t = e.getByToken(rawl_,rawdata);
  
  if ( ! t ) {
    if(verbose_) cout << "can't find FEDRawDataCollection "<< endl;
  }
  
  else {

     if(verbose_) cout << "fedlist size = " << l1feds_.size() << endl;

     for (unsigned int i = 0; i<l1feds_.size(); i++){
        int fedId = l1feds_[i];
        if(verbose_) cout << "fedId = " << fedId << endl;
       
        const FEDRawData & data = rawdata->FEDData(fedId);
        
	if(size_t size=data.size()){
	  
	  fedentries->Fill(i);
	  hfedsize->Fill(float(size));
	  hfedprof->Fill(float(i),float(size));
	  if(verbose_) cout << "header check = " << FEDHeader(data.data()).check() << endl;
	  if(verbose_) cout << "trailer check = " << FEDTrailer(data.data()).check() << endl;
	  
	  if(!FEDHeader(data.data()).check()) fedfatal->Fill(i);
	  
        } else {
	  
	  if(verbose_) cout << "empty fed " << i << endl;
	  if(stableROConfig_) fedfatal->Fill(i);
	  
	}
     }  
  } 
}


