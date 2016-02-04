/*
 * \file L1TFED.cc
 *
 * $Date: 2010/04/06 01:14:46 $
 * $Revision: 1.15 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TFED.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

using namespace std;
using namespace edm;

L1TFED::L1TFED(const ParameterSet& ps)
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  rawl_  = ps.getParameter< InputTag >("rawTag");
  if(verbose_) cout << "L1TFED: constructor...." << endl;


  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    dbe = Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }
  
  l1feds_ = ps.getParameter<std::vector<int> >("L1FEDS");

  directory_ = ps.getUntrackedParameter<std::string>("FEDDirName","L1T/FEDIntegrity");


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder(directory_);
  }

  stableROConfig_ = ps.getUntrackedParameter<bool>("stableROConfig", true);
}

L1TFED::~L1TFED()
{
}

void L1TFED::beginJob(void)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder(directory_);
    dbe->rmdir(directory_);
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder(directory_);
    
    fedentries = dbe->book1D("FEDEntries", "Fed ID occupancy", l1feds_.size(), 0.,l1feds_.size() );	  
    fedfatal = dbe->book1D("FEDFatal", "Fed ID non present ", l1feds_.size(), 0., l1feds_.size());	  
    fednonfatal = dbe->book1D("FEDNonFatal", "Fed corrupted data ", l1feds_.size(), 0.,l1feds_.size() );
    hfedprof = dbe->bookProfile("fedprofile","FED Size by ID", l1feds_.size(), 0., l1feds_.size(),0,0.,5000.);
    for(unsigned int i=0;i<l1feds_.size();i++){
       ostringstream sfed;
       sfed << l1feds_[i];
       fedentries->setBinLabel(i+1,"FED "+ sfed.str());
       fedfatal->setBinLabel(i+1,"FED "+ sfed.str());
       fednonfatal->setBinLabel(i+1,"FED "+ sfed.str());
//       hfedprof->getTProfile()->GetXaxis()->SetBinLabel(i+1,"FED "+ sfed.str());

    }
    	  
    hfedsize = dbe->book1D("fedsize","FED Size Distribution",100,0.,10000.);

   }
}


void L1TFED::endJob(void)
{
 
  if(verbose_) std::cout << "L1T FED Integrity: end job...." << std::endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TFED::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1T FED Integrity: analyze...." << endl;

  edm::Handle<FEDRawDataCollection> rawdata;
  bool t = e.getByLabel(rawl_,rawdata);
  
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

//            if(!FEDHeader(data.data()).check() || !FEDTrailer(data.data()).check()) fedfatal->Fill(i);
// fedtrailer check seems to be always 0.

//          for fedId dedicated integrity checks.
/*          switch(fedId){
	 
	       case 813:
	       std::cout << "do something for GT 813 data corruption..." << std::endl; continue;
	       fednonfatal->Fill(fedId);
	    
	       case 814:
	       std::cout << "do something for GT 814 data corruption..." << std::endl; continue;
	       fednonfatal->Fill(fedId);
	    }
*/	 
        } else {
        
         if(verbose_) cout << "empty fed " << i << endl;
	 if(stableROConfig_) fedfatal->Fill(i);
        
	}
   }
  
 }

}


