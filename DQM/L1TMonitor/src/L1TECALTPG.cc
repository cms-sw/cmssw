/*
 * \file L1TECALTPG.cc
 *
 * $Date: 2007/02/19 19:24:09 $
 * $Revision: 1.1 $
 * \author J. Berryhill
 *
 * - initial version stolen from GCTMonnitor (thanks!) (wittich 02/07)
 *
 * $Log$
 */

#include "DQM/L1TMonitor/interface/L1TECALTPG.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

// end of header files 
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


L1TECALTPG::L1TECALTPG(const ParameterSet & ps)
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    std::cout << "L1TECALTPG: constructor...." << std::endl;

  logFile_.open("L1TECALTPG.log");

  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DaqMonitorBEInterface", false)) {
    dbe = Service < DaqMonitorBEInterface > ().operator->();
    dbe->setVerbose(0);
  }

  monitorDaemon_ = false;
  if (ps.getUntrackedParameter < bool > ("MonitorDaemon", false)) {
    Service < MonitorDaemon > daemon;
    daemon.operator->();
    monitorDaemon_ = true;
  }

  outputFile_ =
      ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    std::cout << "L1T Monitoring histograms will be saved to " 
	      << outputFile_ << std::endl;
  }
  else {
    outputFile_ = "L1TDQM.root";
  }

  bool disable =
      ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }


  if (dbe != NULL) {
    dbe->setCurrentFolder("L1TMonitor/L1TECALTPG");
  }


}

L1TECALTPG::~L1TECALTPG()
{
}

void L1TECALTPG::beginJob(const EventSetup & c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface *dbe = 0;
  dbe = Service < DaqMonitorBEInterface > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/L1TECALTPG");
    dbe->rmdir("L1TMonitor/L1TECALTPG");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1TMonitor/L1TECALTPG");
    ecalTpEtEtaPhi_ = 
      dbe->book2D("EcalTpEtEtaPhi", "ECAL TP E_{T}", TPPHIBINS, TPPHIMIN,
		  TPPHIMAX, TPETABINS, TPETAMIN, TPETAMAX);
    ecalTpOccEtaPhi_ =
	dbe->book2D("EcalTpOccEtaPhi", "ECAL TP OCCUPANCY", TPPHIBINS,
		    TPPHIMIN, TPPHIMAX, TPETABINS, TPETAMIN, TPETAMAX);
    ecalTpRank_ =
      dbe->book1D("EcalTpRank", "ECAL TP RANK", RTPBINS, RTPMIN, RTPMAX);

  }
}


void L1TECALTPG::endJob(void)
{
  if (verbose_)
    std::cout << "L1TECALTPG: end job...." << std::endl;
  LogInfo("L1TECALTPG") << "analyzed " << nev_ << " events";

  if (outputFile_.size() != 0 && dbe)
    dbe->save(outputFile_);

  return;
}

void L1TECALTPG::analyze(const Event & e, const EventSetup & c)
{
  nev_++;
  if (verbose_)
    std::cout << "L1TECALTPG: analyze...." << std::endl;

  // Get the ECAL TPGs
  edm::Handle < EcalTrigPrimDigiCollection > eTP;
  e.getByType(eTP);

  // Fill the ECAL TPG histograms
  for (EcalTrigPrimDigiCollection::const_iterator ieTP = eTP->begin();
       ieTP != eTP->end(); ieTP++) {
    ecalTpEtEtaPhi_->Fill(ieTP->id().iphi(), ieTP->id().ieta(),
			  ieTP->compressedEt());
    ecalTpOccEtaPhi_->Fill(ieTP->id().iphi(), ieTP->id().ieta());
    ecalTpRank_->Fill(ieTP->compressedEt());

  }


}
