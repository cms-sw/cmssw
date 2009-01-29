/*
 * \file L1TRPCTF.cc
 *
 * $Date: 2009/01/28 15:58:38 $
 * $Revision: 1.20 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTF.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"


using namespace std;
using namespace edm;

L1TRPCTF::L1TRPCTF(const ParameterSet& ps)
  : rpctfSource_( ps.getParameter< InputTag >("rpctfSource") ),
//    digiSource_( ps.getParameter< InputTag >("rpctfRPCDigiSource") ),
//   m_rpcDigiFine(false),
//    m_useRpcDigi(true),
   m_ntracks(0),
   m_rateUpdateTime( ps.getParameter< int >("rateUpdateTime") ),
   m_maxRateHistoSize( ps.getParameter< int >("maxRateHistoSize") ),
   output_dir_ (ps.getUntrackedParameter<string>("output_dir") )
//    m_rpcDigiWithBX0(0),
//    m_rpcDigiWithBXnon0(0)

 {

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TRPCTF: constructor...." << endl;


  m_dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    m_dbe = Service<DQMStore>().operator->();
    m_dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( m_dbe !=NULL ) {
    m_dbe->setCurrentFolder(output_dir_);
  }



}

L1TRPCTF::~L1TRPCTF()
{
}

void L1TRPCTF::beginJob(const EventSetup& c)
{

  nev_ = 0;
  nevRPC_ = 0;

  // get hold of back-end interface
  m_dbe = Service<DQMStore>().operator->();

  if ( m_dbe ) {
    m_dbe->setCurrentFolder(output_dir_);
    m_dbe->rmdir(output_dir_);
  }


  if ( m_dbe ) 
  {
    m_dbe->setCurrentFolder(output_dir_);
    
    rpctfetavalue[1] = m_dbe->book1D("RPCTF_eta_value", 
       "RPCTF eta value", 100, -2.5, 2.5 ) ;
    rpctfetavalue[2] = m_dbe->book1D("RPCTF_eta_value_+1", 
       "RPCTF eta value bx +1", 100, -2.5, 2.5 ) ;
    rpctfetavalue[0] = m_dbe->book1D("RPCTF_eta_value_-1", 
       "RPCTF eta value bx -1", 100, -2.5, 2.5 ) ;
    
    rpctfphivalue[1] = m_dbe->book1D("RPCTF_phi_value", 
       "RPCTF phi value", 144, 0.0, 6.2832 ) ;
    rpctfphivalue[2] = m_dbe->book1D("RPCTF_phi_value_+1", 
       "RPCTF phi value bx +1", 144, 0.0, 6.2832 ) ;
    rpctfphivalue[0] = m_dbe->book1D("RPCTF_phi_value_-1", 
       "RPCTF phi value bx -1", 144, 0.0, 6.2832 ) ;
        
    rpctfptvalue[1] = m_dbe->book1D("RPCTF_pt_value", 
       "RPCTF pt value", 160, -0.5, 159.5 ) ;
    rpctfptvalue[2] = m_dbe->book1D("RPCTF_pt_value_+1", 
       "RPCTF pt value bx +1", 160, -0.5, 159.5 ) ;
    rpctfptvalue[0] = m_dbe->book1D("RPCTF_pt_value_-1", 
       "RPCTF pt value bx -1", 160, -0.5, 159.5 ) ;
    
    rpctfchargevalue[1] = m_dbe->book1D("RPCTF_charge_value", 
       "RPCTF charge value", 3, -1.5, 1.5 ) ;
    rpctfchargevalue[2] = m_dbe->book1D("RPCTF_charge_value_+1", 
       "RPCTF charge value bx +1", 3, -1.5, 1.5 ) ;
    rpctfchargevalue[0] = m_dbe->book1D("RPCTF_charge_value_-1", 
       "RPCTF charge value bx -1", 3, -1.5, 1.5 ) ;

    rpctfquality[1] = m_dbe->book1D("RPCTF_quality", 
       "RPCTF quality", 6, -0.5, 5.5 ) ;
    rpctfquality[2] = m_dbe->book1D("RPCTF_quality_+1", 
       "RPCTF quality bx +1", 6, -0.5, 5.5 ) ;
    rpctfquality[0] = m_dbe->book1D("RPCTF_quality_-1", 
       "RPCTF quality bx -1", 6, -0.5, 5.5 ) ;

    rpctfntrack = m_dbe->book1D("RPCTF_ntrack", 
       "RPCTF number of tracks", 10, -0.5, 9.5 ) ;
    
    rpctfbx = m_dbe->book1D("RPCTF_bx", 
       "RPCTF bx distribiution", 5, -2.5, 2.5 ) ;

    m_qualVsEta = m_dbe->book2D("RPCTF_quality_vs_tower", 
                              "RPCTF quality vs eta", 
                              //100, -2.5, 2.5,
                               33, -16.5, 16.5,
                               6, -0.5, 5.5); // Currently only 0...3 quals are possible
    
    m_muonsEtaPhi = m_dbe->book2D("RPCTF_muons_tower_phipacked", 
                                "RPCTF muons(tower,phi)",  
                               // 100, -2.5, 2.5,
                                33, -16.5, 16.5,
                                144,  -0.5, 143.5);

   
    
    m_phipacked = m_dbe->book1D("RPCTF_phi_valuepacked", 
                           "RPCTF phi valuepacked", 144, -0.5, 143.5 ) ;

    
    m_rate = m_dbe->book1D("RPCTF_rate",
                           "RPCTrigger rate - arbitrary units", 3600, 0, 3600); // range will be extended if needed
    
        
  }  
}

void L1TRPCTF::endRun(const edm::Run & r, const edm::EventSetup & c){
  
  std::pair<int,int> p = m_rateHelper.removeAndGetRateForEarliestTime();
  while (p.first != -1 )
  {
     
    if (p.first > -1){
      fillRateHisto(p);
    }
    p = m_rateHelper.removeAndGetRateForEarliestTime();
  }


}


void L1TRPCTF::endJob(void)
{
  
  if(verbose_) cout << "L1TRPCTF: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

  if ( outputFile_.size() != 0  && m_dbe ) m_dbe->save(outputFile_);
    
  return;

}

void L1TRPCTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TRPCTF: analyze...." << endl;

  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByLabel(rpctfSource_,pCollection);
  
  if (!pCollection.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
			       << rpctfSource_.label() ;
    return;
  }

  
  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

  static int nrpctftrack;
  nrpctftrack = 0;
 
  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ ) 
  {
    
   if (verbose_) cout << "Readout Record " << RRItr->getBxInEvent() << endl;
   
   vector<vector<L1MuRegionalCand> > brlAndFwdCands;
   brlAndFwdCands.push_back(RRItr->getBrlRPCCands());
   brlAndFwdCands.push_back(RRItr->getFwdRPCCands());
  
   vector<vector<L1MuRegionalCand> >::iterator RPCTFCands = brlAndFwdCands.begin();
   for(; RPCTFCands!= brlAndFwdCands.end(); ++RPCTFCands)
   {
      for( vector<L1MuRegionalCand>::const_iterator 
          ECItr = RPCTFCands->begin() ;
          ECItr != RPCTFCands->end() ;
          ++ECItr ) 
      {
  
        int bxindex = ECItr->bx() + 1;
        
        if (!ECItr->empty()) {
          
          nrpctftrack++;
    
          if (verbose_) cout << "RPCTFCand bx " << ECItr->bx() << endl;
          
          rpctfbx->Fill(ECItr->bx());
    
          rpctfetavalue[bxindex]->Fill(ECItr->etaValue());
          if (verbose_) cout << "\tRPCTFCand eta value " << ECItr->etaValue() << endl;
  
          rpctfphivalue[bxindex]->Fill(ECItr->phiValue());
          if (verbose_) cout << "\tRPCTFCand phi value " << ECItr->phiValue() << endl;
    
          rpctfptvalue[bxindex]->Fill(ECItr->ptValue());
          if (verbose_) cout << "\tRPCTFCand pt value " << ECItr->ptValue()<< endl;
    
          rpctfchargevalue[bxindex]->Fill(ECItr->chargeValue());
          if (verbose_) cout << "\tRPCTFCand charge value " << ECItr->chargeValue() << endl;
    
          rpctfquality[bxindex]->Fill(ECItr->quality());
          if (verbose_) cout << "\tRPCTFCand quality " << ECItr->quality() << endl;
          
          int tower = ECItr->eta_packed();
          if (tower > 16) {
            tower = - ( (~tower & 63) + 1);
          }

          m_qualVsEta->Fill(tower, ECItr->quality());
          m_muonsEtaPhi->Fill(tower, ECItr->phi_packed());
          m_phipacked->Fill(ECItr->phi_packed());
          
        } // if !empty
      } // end candidates iteration
   } // end brl/endcap iteration
  } // end GMT records iteration

  rpctfntrack->Fill(nrpctftrack);
  
  if (nrpctftrack>0) {
    m_rateHelper.addOrbit(e.orbitNumber());
  }

  
  int et = m_rateHelper.getEarliestTime();
  
  if ( ( m_rateHelper.getTimeForOrbit(e.orbitNumber()) - et > m_rateUpdateTime) 
         && (et > -1) 
         && m_rateUpdateTime!=-1)
  {
  
    std::pair<int, int> p = m_rateHelper.removeAndGetRateForEarliestTime();
    
    if (p.first > -1){
      fillRateHisto(p);
      
    }
    
  }
  
  m_ntracks += nrpctftrack;


  if (verbose_) cout << "\tRPCTFCand ntrack " << nrpctftrack << endl;
	
}


/// Fills rate histo. Extends scale if needed
void L1TRPCTF::fillRateHisto(std::pair<int,int> & p)
{
  static bool resizePossible = true;
      
  if (!resizePossible) return; // we have run out of space allready
  
  //static int fills = 0;
  //++fills;
    
  // check if we are running out of storage space, if so extend the scale
  float occupancy = 1.*p.first/m_rate->getNbinsX();
  while( occupancy>0.95 && resizePossible ){
    
    //std::cout << " Trying to resize " << std::endl;
    m_dbe->setCurrentFolder(output_dir_);
    static float gd = 1.61;
    int curbins=m_rate->getNbinsX();
    int nbins=curbins*gd+1; // new size
    
    if (occupancy>gd) { 
      nbins=nbins*occupancy;
    } 
    
    if (nbins > m_maxRateHistoSize) // limit the number of bins
      nbins = m_maxRateHistoSize;
    
    if (curbins<nbins) {
    
      //std::cout << " Resizing " << std::endl;
      TH1F * histCopy= (TH1F *)m_rate->getTH1F()->Clone();
      
      std::string name = m_rate->getName(); 
      //std::cout << " Removing: " << name << std::endl;
      
      m_dbe->setCurrentFolder(output_dir_);
      m_dbe->removeElement(name);
      
      m_rate = m_dbe->book1D(name,
                          "RPCTrigger rate - arbitrary units", nbins, 0, nbins);
      
      for (int i = 1; i < histCopy->GetNbinsX(); ++i){
        m_rate->setBinContent(i,histCopy->GetBinContent(i));
      } 
      
      delete histCopy;
    } else {
      //std::cout << " Resize impossible " << std::endl;
      resizePossible = false;
    }
    
    occupancy = 1.*p.first/m_rate->getNbinsX();
  }
  
  if (resizePossible){
    m_rate->setBinContent(p.first,p.second);
    //std::cout << fills << " Filling"  << std::endl;
  }
  


}


void L1TRPCTF::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                    const edm::EventSetup& context)
{
   m_ntracks = 0;
//    m_rpcDigiWithBX0=0;
//    m_rpcDigiWithBXnon0=0;
//    m_bxs.clear();
//    m_useRpcDigi = true;

                          
}


void L1TRPCTF::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                        const edm::EventSetup& c)
{

}

