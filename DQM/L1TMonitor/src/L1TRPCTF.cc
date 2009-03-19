/*
 * \file L1TRPCTF.cc
 *
 * $Date: 2009/01/30 13:09:44 $
 * $Revision: 1.24 $
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
   m_rateBinSize( ps.getParameter< int >("rateBinSize") ),
   m_rateNoOfBins( ps.getParameter< int >("rateNoOfBins") ),
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

    
    m_rateMin = m_dbe->book1D("RPCTF_rate_min",
                              "RPCTrigger - minimal rate", m_rateNoOfBins, 0, m_rateNoOfBins); 
    
    m_rateMax = m_dbe->book1D("RPCTF_rate_max",
                              "RPCTrigger - peak rate", m_rateNoOfBins, 0, m_rateNoOfBins);

    m_rateAvg = m_dbe->book1D("RPCTF_rate_avg",
                              "RPCTrigger - average rate", m_rateNoOfBins, 0, m_rateNoOfBins); 
                              
    m_bxDiff = m_dbe->book1D("RPCTF_bx_diff",
			      "RPCTrigger - bx difference", 12000, -.5, 11999.5); 
    
  }  
}

void L1TRPCTF::endRun(const edm::Run & r, const edm::EventSetup & c){
  
      fillRateHistos(0,true);
      
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
        
    unsigned int globBx = e.orbitNumber()*3564+e.bunchCrossing();
    if (m_globBX.find(globBx)==m_globBX.end()) m_globBX.insert(globBx);
    if (m_globBX.size()>1020){
      static int lastUsedBx = 0;
      int diff = *m_globBX.begin()-lastUsedBx; // first entry will go to overflow bin, ignore
      m_bxDiff->Fill(diff);
      lastUsedBx = *m_globBX.begin();
      m_globBX.erase(m_globBX.begin());
    
    }
    
    
    
  }
  fillRateHistos(e.orbitNumber());
  
  
  m_ntracks += nrpctftrack;


  if (verbose_) cout << "\tRPCTFCand ntrack " << nrpctftrack << endl;
	
}


/** Fills rate histos. 
 */
void L1TRPCTF::fillRateHistos(int orbit, bool flush)
{
  
  static bool flushed = false;

  if (flushed) {
    LogWarning("L1TRPCTF") << "Rate histos allready flushed \n";
  }
    
  if (flush) flushed = true;
  

  int nbinsUsed = 0;
  do  {
    nbinsUsed = 0;
    int et = m_rateHelper.getEarliestTime();
    if (et==-1) break;
    
    if ( (( m_rateHelper.getTimeForOrbit(orbit) - et > m_rateUpdateTime+m_rateBinSize)  // 1 minute bins
            && m_rateUpdateTime!=-1) || flush  )
    {
      
      
      int startTimeInMinutes=et/m_rateBinSize; 
      int bin = 0;
      std::pair<int, int> p; 
      int max = 0, min = 0;
      float avg=0;
      int curTimeInMinutes=startTimeInMinutes;
      while (curTimeInMinutes==startTimeInMinutes){
        p = m_rateHelper.removeAndGetRateForEarliestTime(); 
        if (p.first < 0) break; // no more items to analize, go fill histos
        
        if (nbinsUsed==0) {
          bin = p.first/m_rateBinSize+1;
          max = p.second;
          min = p.second;
        } else {
          if (max < p.second) max = p.second;
          if (min > p.second) min = p.second;
        }
        
        ++nbinsUsed;
        avg+=p.second;
        curTimeInMinutes=m_rateHelper.getEarliestTime()/m_rateBinSize;
      }
      
      
      avg/=m_rateBinSize;
      
      if (nbinsUsed > 0){
        m_rateAvg->setBinContent(bin,avg); // smallest possible value in f.first is 1
        m_rateMin->setBinContent(bin,min);
        m_rateMax->setBinContent(bin,max);
      }
    
    }
  } while (flush);



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

