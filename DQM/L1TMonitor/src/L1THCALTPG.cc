/*
 * \file L1THCALTPG.cc
 *
 * $Date: 2008/03/14 20:35:46 $
 * $Revision: 1.11 $
 * \author J. Berryhill
 *
 * $Log: L1THCALTPG.cc,v $
 * Revision 1.11  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.10  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.9  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.8  2007/12/21 17:41:21  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.7  2007/12/05 16:48:53  berryhil
 *
 *
 * plug in L1THCALTPG, L1TECALTPG, L1TGCT, L1TRCT, GCTMonitor
 *
 * Revision 1.6  2007/11/19 15:08:22  lorenzo
 * changed top folder name
 *
 * Revision 1.5  2007/07/19 16:48:23  berryhil
 *
 *
 *
 * seal plugin re-migration
 * add status digis to L1TCSCTF
 * HCALTPG data format migration (fiberChan)
 * GT data format migration (PSB)
 *
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
#include "DQMServices/Core/interface/DQMStore.h"

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


  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    dbe = Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    std::cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << std::endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1T/L1THCALTPG");
  }


}

L1THCALTPG::~L1THCALTPG()
{
}

void L1THCALTPG::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1THCALTPG");
    dbe->rmdir("L1T/L1THCALTPG");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1THCALTPG");
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
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1THCALTPG::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) std::cout << "L1THCALTPG: analyze...." << std::endl;

  edm::Handle<HcalTrigPrimDigiCollection> hcalTpgs;
  e.getByLabel(hcaltpgSource_, hcalTpgs);
  
  if (!hcalTpgs.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find HCAL TPG's with label "
			       << hcaltpgSource_.label() ;
    return;
  }
//
//  std::cout << "--> event  " << hcalTpgs->size() << std::endl;
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

