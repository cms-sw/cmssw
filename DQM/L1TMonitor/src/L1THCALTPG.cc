/*
 * \file L1THCALTPG.cc
 *
 * $Date: 2007/06/12 19:32:53 $
 * $Revision: 1.4 $
 * \author J. Berryhill
 *
 * $Log: L1THCALTPG.cc,v $
 * Revision 1.4  2007/06/12 19:32:53  berryhil
 *
 *
 * config files now include hcal tpg monitoring modules
 *
 * Revision 1.3  2007/02/23 22:00:16  wittich
 * add occ (weighted and unweighted) and rank histos
 *
 *
 */

#include "DQM/L1TMonitor/interface/L1THCALTPG.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

using namespace edm;


// Local definitions for the limits of the histograms
const unsigned int RTPBINS = 101;
const float RTPMIN = -0.5;
const float RTPMAX = 100.5;

const unsigned int TPPHIBINS = 72;
const float TPPHIMIN = 0.5;
const float TPPHIMAX = 72.5;

const unsigned int TPETABINS = 65;
const float TPETAMIN = -32.5;
const float TPETAMAX = 32.5;


L1THCALTPG::L1THCALTPG(const ParameterSet& ps)
  : hcaltpgSource_( ps.getParameter< InputTag >("hcaltpgSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) std::cout << "L1THCALTPG: constructor...." << std::endl;

  logFile_.open("L1THCALTPG.log");

  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DaqMonitorBEInterface", false) ) 
  {
    dbe = Service<DaqMonitorBEInterface>().operator->();
    dbe->setVerbose(0);
  }

  monitorDaemon_ = false;
  if ( ps.getUntrackedParameter<bool>("MonitorDaemon", false) ) {
    Service<MonitorDaemon> daemon;
    daemon.operator->();
    monitorDaemon_ = true;
  }

  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    std::cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << std::endl;
  }
  else{
    outputFile_ = "L1TDQM.root";
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1TMonitor/L1THCALTPG");
  }


}

L1THCALTPG::~L1THCALTPG()
{
}

void L1THCALTPG::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1THCALTPG");
    dbe->rmdir("L1TMonitor/L1THCALTPG");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1THCALTPG");
    hcalTpEtEtaPhi_ = 
      dbe->book2D("HcalTpEtEtaPhi", "HCAL TP E_{T}", TPPHIBINS, TPPHIMIN,
		  TPPHIMAX, TPETABINS, TPETAMIN, TPETAMAX);
    hcalTpOccEtaPhi_ =
	dbe->book2D("HcalTpOccEtaPhi", "HCAL TP OCCUPANCY", TPPHIBINS,
		    TPPHIMIN, TPPHIMAX, TPETABINS, TPETAMIN, TPETAMAX);
    hcalTpRank_ =
      dbe->book1D("HcalTpRank", "HCAL TP RANK", RTPBINS, RTPMIN, RTPMAX);
    
  }  
}


void L1THCALTPG::endJob(void)
{
  if(verbose_) std::cout << "L1THCALTPG: end job...." << std::endl;
  LogInfo("L1THCALTPG") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1THCALTPG::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) std::cout << "L1THCALTPG: analyze...." << std::endl;

  edm::Handle<HcalTrigPrimDigiCollection> hcalTpgs;

  try {
        e.getByLabel(hcaltpgSource_, hcalTpgs);
    //     e.getByType(hcalTpgs);
  }
  catch (...) {
    edm::LogInfo("L1THCALTPG") << "can't find HCAL TPG's with label "
			       << hcaltpgSource_.label() ;
    return;
  }
//
  std::cout << "--> event  " << hcalTpgs->size() << std::endl;
//   int j = 0;
  for ( HcalTrigPrimDigiCollection::const_iterator i = hcalTpgs->begin();
	i != hcalTpgs->end(); ++i ) {

    if (verbose_)
      {
  std::cout << "size  " <<  i->size() << std::endl;
  std::cout << "iphi  " <<  i->id().iphi() << std::endl;
  std::cout << "ieta  " <<  i->id().ieta() << std::endl;
  std::cout << "compressed Et  " <<  i->SOI_compressedEt() << std::endl;
  std::cout << "FG bit  " <<  i->SOI_fineGrain() << std::endl;
  std::cout << "raw  " <<  i->t0().raw() << std::endl;
  std::cout << "raw Et " <<  i->t0().compressedEt() << std::endl;
  std::cout << "raw FG " <<  i->t0().fineGrain() << std::endl;
  //  std::cout << "raw fiber " <<  i->t0().fiber() << std::endl;
  //  std::cout << "raw fiberChan " <<  i->t0().fiberChan() << std::endl;
  //  std::cout << "raw fiberAndChan " <<  i->t0().fiberAndChan() << std::endl;
      }

   int e = i->SOI_compressedEt();
    if ( e != 0 ) {
      // occupancy maps (weighted and unweighted
      hcalTpOccEtaPhi_->Fill(i->id().iphi(), i->id().ieta());
      hcalTpEtEtaPhi_ ->Fill(i->id().iphi(), i->id().ieta(),
			     i->SOI_compressedEt());
      // et
      hcalTpRank_->Fill(i->SOI_compressedEt());
      //std::cout << j++ << " : " << i->SOI_compressedEt() << std::endl;
    }
  }

}

