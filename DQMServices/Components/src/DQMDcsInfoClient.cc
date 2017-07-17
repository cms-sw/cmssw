
/*
 * \file DQMDcsInfoClient.cc
 * \author Andreas Meyer
 * Last Update:
 *
*/

#include "DQMServices/Components/src/DQMDcsInfoClient.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

DQMDcsInfoClient::DQMDcsInfoClient( const edm::ParameterSet& ps ) {

  parameters_ = ps;

  dbe_ = edm::Service<DQMStore>().operator->();

  subsystemname_ = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "Info") ;
  dcsinfofolder_ = parameters_.getUntrackedParameter<std::string>("dcsInfoFolder", "DcsInfo") ;

}


DQMDcsInfoClient::~DQMDcsInfoClient() {

}

void 
DQMDcsInfoClient::beginRun(const edm::Run& r, const edm::EventSetup& c) 
{
  // Fetch GlobalTag information and fill the string/ME.
  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ +"/CMSSWInfo/");
  const edm::ParameterSet &globalTagPSet =
    edm::getProcessParameterSetContainingModule(moduleDescription())
    .getParameterSet("PoolDBESSource@GlobalTag");

  dbe_->bookString("globalTag_Harvesting", globalTagPSet.getParameter<std::string>("globaltag"));

  DCS.clear();
  DCS.resize(10);  // start with 10 LS, resize later
  processedLS_.clear();
}

void 
DQMDcsInfoClient::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  return;
}

void 
DQMDcsInfoClient::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c)
{
  if (!dbe_) return;

  unsigned int nlumi = l.id().luminosityBlock() ;
  processedLS_.insert(nlumi);
  // cout << " in lumi section " << nlumi << endl;

  if (nlumi+1 > DCS.size()) 
     DCS.resize(nlumi+1);
  // cout << "DCS size: " << DCS.size() << endl;     

  MonitorElement* DCSbyLS_ = 
            dbe_->get(subsystemname_ + "/" + dcsinfofolder_ + "/DCSbyLS" ); 

  if ( DCSbyLS_ ) 
  {
    if ( TH1F * h1 = DCSbyLS_->getTH1F()) 
    {
      int word = 0;
      for (int i = 0; i < 25 ; i++) 
      {

	if ( h1->GetBinContent(i+1) != 0 ) 
           word |= (0x0 << i); // set to 0 because HV was off (!)
	else 
           word |= (0x1 << i); // set to 1 because HV was on (!)
      }
      DCS[nlumi] = word;
    }
  }
  return; 
}

void 
DQMDcsInfoClient::endRun(const edm::Run& r, const edm::EventSetup& c) 
{

  // book 
  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ +"/EventInfo/");

  unsigned int nlsmax = DCS.size();
  reportSummary_=dbe_->bookFloat("reportSummary");
  reportSummary_->Fill(1.);
  
  reportSummaryMap_ = dbe_->get(subsystemname_ +"/EventInfo/reportSummaryMap");
  if (reportSummaryMap_) dbe_->removeElement(reportSummaryMap_->getName());

  reportSummaryMap_ = dbe_->book2D("reportSummaryMap",
                     "HV and GT vs Lumi", nlsmax, 1., nlsmax+1, 25, 0., 25.);
  unsigned int lastProcessedLS = *(--processedLS_.end());
  meProcessedLS_ = dbe_->book1D("ProcessedLS",
				"Processed Lumisections",
				lastProcessedLS+1,
				0.,lastProcessedLS+1);
  reportSummaryMap_->setBinLabel(1," CSC+",2);   
  reportSummaryMap_->setBinLabel(2," CSC-",2);   
  reportSummaryMap_->setBinLabel(3," DT0",2);    
  reportSummaryMap_->setBinLabel(4," DT+",2);    
  reportSummaryMap_->setBinLabel(5," DT-",2);    
  reportSummaryMap_->setBinLabel(6," EB+",2);    
  reportSummaryMap_->setBinLabel(7," EB-",2);    
  reportSummaryMap_->setBinLabel(8," EE+",2);    
  reportSummaryMap_->setBinLabel(9," EE-",2);    
  reportSummaryMap_->setBinLabel(10,"ES+",2);    
  reportSummaryMap_->setBinLabel(11,"ES-",2);   
  reportSummaryMap_->setBinLabel(12,"HBHEa",2); 
  reportSummaryMap_->setBinLabel(13,"HBHEb",2); 
  reportSummaryMap_->setBinLabel(14,"HBHEc",2); 
  reportSummaryMap_->setBinLabel(15,"HF",2);    
  reportSummaryMap_->setBinLabel(16,"HO",2);    
  reportSummaryMap_->setBinLabel(17,"BPIX",2);  
  reportSummaryMap_->setBinLabel(18,"FPIX",2);  
  reportSummaryMap_->setBinLabel(19,"RPC",2);   
  reportSummaryMap_->setBinLabel(20,"TIBTID",2);
  reportSummaryMap_->setBinLabel(21,"TOB",2);   
  reportSummaryMap_->setBinLabel(22,"TECp",2);  
  reportSummaryMap_->setBinLabel(23,"TECm",2);  
  reportSummaryMap_->setBinLabel(24,"CASTOR",2);
  reportSummaryMap_->setBinLabel(25,"PhysDecl",2);
  reportSummaryMap_->setAxisTitle("Luminosity Section");

  // fill
  for (unsigned int i = 0 ; i < DCS.size() ; i++ )
  {
    for ( int j = 0 ; j < 25 ; j++ ) 
    {
      if (DCS[i] & (0x1 << j))
        reportSummaryMap_->setBinContent(i,j+1,1.);
      else
        reportSummaryMap_->setBinContent(i,j+1,0.);
    }
  }

  std::set<unsigned int>::iterator it,ite;
  it  = processedLS_.begin();
  ite = processedLS_.end();
  unsigned int lastAccessed = 0;
  
  for (; it!=ite; it++)
  {
    while (lastAccessed < (*it))
    {
      //      std::cout << "Filling " << lastAccessed << " with -1" << std::endl; 
      meProcessedLS_->Fill(lastAccessed, -1.);
      lastAccessed++;
    }
    //    std::cout << "Filling " << *it << " with 1" << std::endl; 
    meProcessedLS_->Fill(*it);
    lastAccessed = (*it)+1;
  }
  
}
