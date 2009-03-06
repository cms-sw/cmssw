/*
 * \file L1TdeRCT.cc
 *
 * version 0.0 A.Savin 2008/04/26
 * version 1.0 A.Savin 2008/05/05
 * this version contains single channel histos and 1D efficiencies
 */

#include "DQM/L1TMonitor/interface/L1TdeRCT.h"

// GCT and RCT data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DQMServices/Core/interface/DQMStore.h"

// TPGs

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"


#include "TF2.h"

#include <iostream>

using namespace edm;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

const unsigned int TPGPHIBINS = 72;
const float TPGPHIMIN = -0.5;
const float TPGPHIMAX = 71.5;

const unsigned int TPGETABINS = 64;
const float TPGETAMIN = -32.;
const float TPGETAMAX = 32.;


const unsigned int DEBINS = 127;
const float DEMIN = -63.5;
const float DEMAX = 63.5;

const unsigned int PhiEtaMax = 396;
const unsigned int CHNLBINS = 396;
const float CHNLMIN = -0.5;
const float CHNLMAX = 395.5;

bool first = true ;


L1TdeRCT::L1TdeRCT(const ParameterSet & ps) :
   rctSourceEmul_( ps.getParameter< InputTag >("rctSourceEmul") ),
   rctSourceData_( ps.getParameter< InputTag >("rctSourceData") ),
   ecalTPGData_( ps.getParameter< InputTag >("ecalTPGData") ),
   hcalTPGData_( ps.getParameter< InputTag >("hcalTPGData") )

{


  singlechannelhistos_ = ps.getUntrackedParameter < bool > ("singlechannelhistos", false);
                                                                                                                        
  if (singlechannelhistos_)
    if(verbose_) std::cout << "L1TdeRCT: single channels histos ON" << std::endl;
                                                                                                                        
  // verbosity switch
  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

  if (verbose_)
    std::cout << "L1TdeRCT: constructor...." << std::endl;


  dbe = NULL;
  if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ =
      ps.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    if(verbose_) std::
  cout << "L1T Monitoring histograms will be saved to " <<
  outputFile_.c_str() << std::endl;
  }

  bool disable =
      ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }

  histFolder_ 
    = ps.getUntrackedParameter<std::string>("HistFolder", "L1TEMU/L1TdeRCT/");

  if (dbe != NULL) {
    dbe->setCurrentFolder(histFolder_);
  }



}

L1TdeRCT::~L1TdeRCT()
{
}

void L1TdeRCT::beginJob(const EventSetup & c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder(histFolder_);
    dbe->rmdir(histFolder_);
  }


  if (dbe) {

    dbe->setCurrentFolder(histFolder_);

    rctInputTPGEcalOcc_ =
  dbe->book2D("rctInputTPGEcalOcc", "rctInputTPGEcalOcc", TPGETABINS, TPGETAMIN,
        TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

    rctInputTPGHcalOcc_ =
  dbe->book2D("rctInputTPGHcalOcc", "rctInputTPGHcalOcc", TPGETABINS, TPGETAMIN,
        TPGETAMAX, TPGPHIBINS, TPGPHIMIN, TPGPHIMAX);

    rctInputTPGHcalSample_ =
  dbe->book1D("rctInputTPGHcalSample", "rctInputTPGHcalSample", 10, -0.5, 9.5) ;

    dbe->setCurrentFolder(histFolder_+"IsoEm");

    rctIsoEmEff1_ =
  dbe->book2D("rctIsoEmEff1", "rctIsoEmEff1", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff1oneD_ =
  dbe->book1D("rctIsoEmEff1oneD", "rctIsoEmEff1oneD", 
        CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmEff2_ =
  dbe->book2D("rctIsoEmEff2", "rctIsoEmEff2", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff2oneD_ =
  dbe->book1D("rctIsoEmEff2oneD", "rctIsoEmEff2oneD", 
        CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmIneff_ =
  dbe->book2D("rctIsoEmIneff", "rctIsoEmIneff", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmIneff1D_ =
  dbe->book1D("rctIsoEmIneff1D", "rctIsoEmIneff1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmOvereff_ =
  dbe->book2D("rctIsoEmOvereff", "rctIsoEmOvereff", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmOvereff1D_ =
  dbe->book1D("rctIsoEmOvereff1D", "rctIsoEmOvereff1D", 
                    CHNLBINS, CHNLMIN, CHNLMAX);

    dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData");

    rctIsoEmDataOcc_ =
  dbe->book2D("rctIsoEmDataOcc", "rctIsoEmDataOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmDataOcc1D_ =
  dbe->book1D("rctIsoEmDataOcc1D", "rctIsoEmDataOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmEmulOcc_ =
  dbe->book2D("rctIsoEmEmulOcc", "rctIsoEmEmulOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEmulOcc1D_ =
  dbe->book1D("rctIsoEmEmulOcc1D", "rctIsoEmEmulOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmEff1Occ_ =
  dbe->book2D("rctIsoEmEff1Occ", "rctIsoEmEff1Occ", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff1Occ1D_ =
  dbe->book1D("rctIsoEmEff1Occ1D", "rctIsoEmEff1Occ1D", 
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmEff2Occ_ =
  dbe->book2D("rctIsoEmEff2Occ", "rctIsoEmEff2Occ", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff2Occ1D_ =
  dbe->book1D("rctIsoEmEff2Occ1D", "rctIsoEmEff2Occ1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmIneffOcc_ =
  dbe->book2D("rctIsoEmIneffOcc", "rctIsoEmIneffOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmIneffOcc1D_ =
  dbe->book1D("rctIsoEmIneffOcc1D", "rctIsoEmIneffOcc1D", 
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctIsoEmOvereffOcc_ =
  dbe->book2D("rctIsoEmOvereffOcc", "rctIsoEmOvereffOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmOvereffOcc1D_ =
  dbe->book1D("rctIsoEmOvereffOcc1D", "rctIsoEmOvereffOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);


    dbe->setCurrentFolder(histFolder_+"NisoEm");
    rctNisoEmEff1_ =
  dbe->book2D("rctNisoEmEff1", "rctNisoEmEff1", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff1oneD_ =
  dbe->book1D("rctNisoEmEff1oneD", "rctNisoEmEff1oneD",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmEff2_ =
  dbe->book2D("rctNisoEmEff2", "rctNisoEmEff2", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff2oneD_ =
  dbe->book1D("rctNisoEmEff2oneD", "rctNisoEmEff2oneD",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmIneff_ =
  dbe->book2D("rctNisoEmIneff", "rctNisoEmIneff", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmIneff1D_ =
  dbe->book1D("rctNisoEmIneff1D", "rctNisoEmIneff1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmOvereff_ =
  dbe->book2D("rctNisoEmOvereff", "rctNisoEmOvereff", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmOvereff1D_ =
  dbe->book1D("rctNisoEmOvereff1D", "rctNisoEmOvereff1D", 
                    CHNLBINS, CHNLMIN, CHNLMAX);

    dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData");

    rctNisoEmDataOcc_ =
  dbe->book2D("rctNisoEmDataOcc", "rctNisoEmDataOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmDataOcc1D_ =
  dbe->book1D("rctNisoEmDataOcc1D", "rctNisoEmDataOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmEmulOcc_ =
  dbe->book2D("rctNisoEmEmulOcc", "rctNisoEmEmulOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEmulOcc1D_ =
  dbe->book1D("rctNisoEmEmulOcc1D", "rctNisoEmEmulOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmEff1Occ_ =
  dbe->book2D("rctNisoEmEff1Occ", "rctNisoEmEff1Occ", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff1Occ1D_ =
  dbe->book1D("rctNisoEmEff1Occ1D", "rctNisoEmEff1Occ1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmEff2Occ_ =
  dbe->book2D("rctNisoEmEff2Occ", "rctNisoEmEff2Occ", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff2Occ1D_ =
  dbe->book1D("rctNisoEmEff2Occ1D", "rctNisoEmEff2Occ1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmIneffOcc_ =
  dbe->book2D("rctNisoEmIneffOcc", "rctNisoEmIneffOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmIneffOcc1D_ =
  dbe->book1D("rctNisoEmIneffOcc1D", "rctNisoEmIneffOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    rctNisoEmOvereffOcc_ =
  dbe->book2D("rctNisoEmOvereffOcc", "rctNisoEmOvereffOcc", ETABINS, ETAMIN,
        ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmOvereffOcc1D_ =
  dbe->book1D("rctNisoEmOvereffOcc1D", "rctNisoEmOvereffOcc1D",
                    CHNLBINS, CHNLMIN, CHNLMAX);

    // region information
    dbe->setCurrentFolder(histFolder_+"RegionData");

    rctRegEff1D_ =
      dbe->book1D("rctRegEff1D", "1D region efficiency",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegIneff1D_ =
      dbe->book1D("rctRegIneff1D", "1D region inefficiency",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegOvereff1D_ =
      dbe->book1D("rctRegOvereff1D", "1D region overefficiency",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegEff2D_ =
      dbe->book2D("rctRegEff2D", "2D region efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegIneff2D_ =
      dbe->book2D("rctRegIneff2D", "2D region inefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegOvereff2D_ =
      dbe->book2D("rctRegOvereff2D", "2D region overefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegSpEff2D_ =
      dbe->book2D("rctRegSpEff2D", "2D region special efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    dbe->setCurrentFolder(histFolder_+"RegionData/ServiceData");

    rctRegEmulOcc1D_ =
      dbe->book1D("rctRegEmulOcc1D", "1D region occupancy from emulator",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegMatchedOcc1D_ =
      dbe->book1D("rctRegMatchedOcc1D", "1D region occupancy for matched hits",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegUnmatchedDataOcc1D_ =
      dbe->book1D("rctRegUnmatchedDataOcc1D", "1D region occupancy for unmatched hardware hits",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegUnmatchedEmulOcc1D_ =
      dbe->book1D("rctRegUnmatchedEmulOcc1D", "1D region occupancy for unmatched emulator hits",
      CHNLBINS, CHNLMIN, CHNLMAX);

    rctRegDataOcc2D_ =
      dbe->book2D("rctRegDataOcc2D", "2D region occupancy from hardware",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegEmulOcc2D_ =
      dbe->book2D("rctRegEmulOcc2D", "2D region occupancy from emulator",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegMatchedOcc2D_ =
      dbe->book2D("rctRegMatchedOcc2D", "2D region occupancy for matched hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegUnmatchedDataOcc2D_ =
      dbe->book2D("rctRegUnmatchedDataOcc2D", "2D region occupancy for unmatched hardware hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegUnmatchedEmulOcc2D_ =
      dbe->book2D("rctRegUnmatchedEmulOcc2D", "2D region occupancy for unmatched emulator hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctRegDeltaEt2D_ =
      dbe->book2D("rctRegDeltaEt2D", "2D region \\Delta E_{T}",
      CHNLBINS, CHNLMIN, CHNLMAX, 100, -50., 50.);

    rctRegDeltaEtOcc2D_ =
      dbe->book2D("rctRegDeltaEtOcc2D", "2D region occupancy for \\Delta E_{T}",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    // bit information
    dbe->setCurrentFolder(histFolder_+"BitData");

    rctBitOverFlowEff2D_ =
      dbe->book2D("rctBitOverFlowEff2D", "2D overflow bit efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitOverFlowIneff2D_ =
      dbe->book2D("rctBitOverFlowIneff2D", "2D overflow bit inefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitOverFlowOvereff2D_ =
      dbe->book2D("rctBitOverFlowOvereff2D", "2D overflow bit overefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitTauVetoEff2D_ =
      dbe->book2D("rctBitTauVetoEff2D", "2D tau veto bit efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitTauVetoIneff2D_ =
      dbe->book2D("rctBitTauVetoIneff2D", "2D tau veto bit inefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitTauVetoOvereff2D_ =
      dbe->book2D("rctBitTauVetoOvereff2D", "2D tau veto bit overefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMipEff2D_ =
      dbe->book2D("rctBitMipEff2D", "2D mip bit efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMipIneff2D_ =
      dbe->book2D("rctBitMipIneff2D", "2D mip bit inefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMipOvereff2D_ =
      dbe->book2D("rctBitMipOvereff2D", "2D mip bit overefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitQuietEff2D_ =
      dbe->book2D("rctBitQuietEff2D", "2D quiet bit efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitQuietIneff2D_ =
      dbe->book2D("rctBitQuietIneff2D", "2D quiet bit inefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitQuietOvereff2D_ =
      dbe->book2D("rctBitQuietOvereff2D", "2D quiet bit overefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitFineGrainEff2D_ =
      dbe->book2D("rctBitFineGrainEff2D", "2D fine grain bit efficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitFineGrainIneff2D_ =
      dbe->book2D("rctBitFineGrainIneff2D", "2D fine grain bit inefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitFineGrainOvereff2D_ =
      dbe->book2D("rctBitFineGrainOvereff2D", "2D fine grain bit overefficiency",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    dbe->setCurrentFolder(histFolder_+"BitData/ServiceData");

    rctBitEmulOverFlow2D_ =
      dbe->book2D("rctBitEmulOverFlow2D", "2D overflow bit from emulator",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitDataOverFlow2D_ =
      dbe->book2D("rctBitDataOverFlow2D", "2D overflow bit from hardware",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMatchedOverFlow2D_ =
      dbe->book2D("rctBitMatchedOverFlow2D", "2D overflow bit for matched hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedEmulOverFlow2D_ =
      dbe->book2D("rctBitUnmatchedEmulOverFlow2D", "2D overflow bit for unmatched emulator hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedDataOverFlow2D_ =
      dbe->book2D("rctBitUnmatchedDataOverFlow2D", "2D overflow bit for unmatched hardware hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitEmulTauVeto2D_ =
      dbe->book2D("rctBitEmulTauVeto2D", "2D tau veto bit from emulator",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitDataTauVeto2D_ =
      dbe->book2D("rctBitDataTauVeto2D", "2D tau veto bit from hardware",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMatchedTauVeto2D_ =
      dbe->book2D("rctBitMatchedTauVeto2D", "2D tau veto bit for matched hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedEmulTauVeto2D_ =
      dbe->book2D("rctBitUnmatchedEmulTauVeto2D", "2D tau veto bit for unmatched emulator hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedDataTauVeto2D_ =
      dbe->book2D("rctBitUnmatchedDataTauVeto2D", "2D tau veto bit for unmatched hardware hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitEmulMip2D_ =
      dbe->book2D("rctBitEmulMip2D", "2D mip bit from emulator",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitDataMip2D_ =
      dbe->book2D("rctBitDataMip2D", "2D mip bit from hardware",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMatchedMip2D_ =
      dbe->book2D("rctBitMatchedMip2D", "2D mip bit for matched hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedEmulMip2D_ =
      dbe->book2D("rctBitUnmatchedEmulMip2D", "2D mip bit for unmatched emulator hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedDataMip2D_ =
      dbe->book2D("rctBitUnmatchedDataMip2D", "2D mip bit for unmatched hardware hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);
      
    rctBitEmulQuiet2D_ =
      dbe->book2D("rctBitEmulQuiet2D", "2D quiet bit from emulator",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitDataQuiet2D_ =
      dbe->book2D("rctBitDataQuiet2D", "2D quiet bit from hardware",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMatchedQuiet2D_ =
      dbe->book2D("rctBitMatchedQuiet2D", "2D quiet bit for matched hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedEmulQuiet2D_ =
      dbe->book2D("rctBitUnmatchedEmulQuiet2D", "2D quiet bit for unmatched emulator hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedDataQuiet2D_ =
      dbe->book2D("rctBitUnmatchedDataQuiet2D", "2D quiet bit for unmatched hardware hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitEmulFineGrain2D_ =
      dbe->book2D("rctBitEmulFineGrain2D", "2D fine grain bit from emulator",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitDataFineGrain2D_ =
      dbe->book2D("rctBitDataFineGrain2D", "2D fine grain bit from hardware",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitMatchedFineGrain2D_ =
      dbe->book2D("rctBitMatchedFineGrain2D", "2D fine grain bit for matched hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedEmulFineGrain2D_ =
      dbe->book2D("rctBitUnmatchedEmulFineGrain2D", "2D fine grain bit for unmatched emulator hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctBitUnmatchedDataFineGrain2D_ =
      dbe->book2D("rctBitUnmatchedDataFineGrain2D", "2D fine grain bit for unmatched hardware hits",
      ETABINS, ETAMIN, ETAMAX, PHIBINS, PHIMIN, PHIMAX);

// for single channels 

    if(singlechannelhistos_)
   {
    for(int m=0; m<9; m++)
    {
    if(m==0) dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData/Eff1SnglChnls");
    if(m==1) dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData/Eff1SnglChnls");
    if(m==2) dbe->setCurrentFolder(histFolder_+"RegionData/ServiceData/EffSnglChnls");
    if(m==3) dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData/IneffSnglChnls");
    if(m==4) dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData/IneffSnglChnls");
    if(m==5) dbe->setCurrentFolder(histFolder_+"RegionData/ServiceData/IneffSnglChnls");
    if(m==6) dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData/OvereffSnglChnls");
    if(m==7) dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData/OvereffSnglChnls");
    if(m==8) dbe->setCurrentFolder(histFolder_+"RegionData/ServiceData/OvereffSnglChnls");

    for(int i=0; i<ETAMAX; i++)
    {
     for(int j=0; j<PHIMAX; j++)
     {
     char name[80], channel[80]={""} ;

     if(m==0) strcpy(name,"(Eemul-Edata)Chnl") ;
     if(m==1) strcpy(name,"(Eemul-Edata)Chnl") ;
     if(m==2) strcpy(name,"(Eemul-Edata)Chnl") ;
     if(m==3) strcpy(name,"EemulChnl") ;
     if(m==4) strcpy(name,"EemulChnl") ;
     if(m==5) strcpy(name,"EemulChnl") ;
     if(m==6) strcpy(name,"EdataChnl") ;
     if(m==7) strcpy(name,"EdataChnl") ;
     if(m==8) strcpy(name,"EdataChnl") ;

     if(i<10 && j<10) sprintf(channel,"_0%d0%d",i,j); 
     else if(i<10) sprintf(channel,"_0%d%d",i,j);
      else if(j<10) sprintf(channel,"_%d0%d",i,j);
       else sprintf(channel,"_%d%d",i,j);
     strcat(name,channel); 

     int chnl=PHIBINS*i+j;

     if(m==0) rctIsoEffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==1) rctNisoEffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==2) rctRegEffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==3) rctIsoIneffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==4) rctNisoIneffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==5) rctRegIneffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==6) rctIsoOvereffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==7) rctNisoOvereffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==8) rctRegOvereffChannel_[chnl] =
  dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     }
    }
    }
   }

//end of single channels


  }
}


void L1TdeRCT::endJob(void)
{
  if (verbose_)
    std::cout << "L1TdeRCT: end job...." << std::endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events";

  if (outputFile_.size() != 0 && dbe)
    dbe->save(outputFile_);

  return;
}

void L1TdeRCT::analyze(const Event & e, const EventSetup & c)
{
//    std::cout << "I am here!" << std::endl ;
  nev_++;
  if (verbose_) {
    std::cout << "L1TdeRCT: analyze...." << std::endl;
  }

  // get TPGs
  edm::Handle<EcalTrigPrimDigiCollection> ecalTpData;
  edm::Handle<HcalTrigPrimDigiCollection> hcalTpData;

  // Get the RCT digis
  edm::Handle < L1CaloEmCollection > emData;
  edm::Handle < L1CaloRegionCollection > rgnData;

  // Get the RCT digis
  edm::Handle < L1CaloEmCollection > emEmul;
  edm::Handle < L1CaloRegionCollection > rgnEmul;

  // need to change to getByLabel
  bool doEm = true; 
  bool doHd = true;
  bool doEcal = true;
  bool doHcal = true;

  // TPG, first try:  
  e.getByLabel(ecalTPGData_,ecalTpData);
  e.getByLabel(hcalTPGData_,hcalTpData);
   
  if (!ecalTpData.isValid()) {
    edm::LogInfo("TPG DataNotFound") << "can't find EcalTrigPrimDigiCollection with label "
             << ecalTPGData_.label() ;
    if (verbose_)std::cout << "Can not find ecalTpData!" << std::endl ;

    doEcal = false ;
  }

  if(doEcal)
  {
  for(EcalTrigPrimDigiCollection::const_iterator iEcalTp = ecalTpData->begin(); iEcalTp != ecalTpData->end(); iEcalTp++)
    if(iEcalTp->compressedEt() > 0)
    {

  if(iEcalTp->id().ieta() > 0)
  rctInputTPGEcalOcc_ -> Fill(1.*(iEcalTp->id().ieta())-0.5,iEcalTp->id().iphi()) ;
  else
  rctInputTPGEcalOcc_ -> Fill(1.*(iEcalTp->id().ieta())+0.5,iEcalTp->id().iphi()) ;

if(verbose_) std::cout << " ECAL data: Energy: " << iEcalTp->compressedEt() << " eta " << iEcalTp->id().ieta() << " phi " << iEcalTp->id().iphi() << std::endl ;
    }
   }

  if (!hcalTpData.isValid()) {
    edm::LogInfo("TPG DataNotFound") << "can't find HcalTrigPrimDigiCollection with label "
             << hcalTPGData_.label() ;
    if (verbose_)std::cout << "Can not find hcalTpData!" << std::endl ;
    
    doHcal = false ;
  }


  if(doHcal)
  {

  for(HcalTrigPrimDigiCollection::const_iterator iHcalTp = hcalTpData->begin(); iHcalTp != hcalTpData->end(); iHcalTp++)
  {
    int highSample=0;
    int highEt=0;

    for (int nSample = 0; nSample < 10; nSample++)
      {
  if (iHcalTp->sample(nSample).compressedEt() != 0)
    {
      if(verbose_) std::cout << "HCAL data: Et " 
          << iHcalTp->sample(nSample).compressedEt()
          << "  fg "
          << iHcalTp->sample(nSample).fineGrain()
          << "  ieta " << iHcalTp->id().ieta()
          << "  iphi " << iHcalTp->id().iphi()
          << "  sample " << nSample 
                      << std::endl ;
      if (iHcalTp->sample(nSample).compressedEt() > highEt)
        {
    highSample = nSample;
                highEt =  iHcalTp->sample(nSample).compressedEt() ;
        }
    }

       }

     if(highEt != 0)
      { 
                  if(iHcalTp->id().ieta() > 0)
                  rctInputTPGHcalOcc_ -> Fill(1.*(iHcalTp->id().ieta())-0.5,iHcalTp->id().iphi()) ;
                  else
                  rctInputTPGHcalOcc_ -> Fill(1.*(iHcalTp->id().ieta())+0.5,iHcalTp->id().iphi()) ;
                  rctInputTPGHcalSample_ -> Fill(highSample) ; 
       }

    }
  }

  
  e.getByLabel(rctSourceData_,rgnData);
  e.getByLabel(rctSourceEmul_,rgnEmul);
 
  if (!rgnData.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection with label "
             << rctSourceData_.label() ;
    if (verbose_)std::cout << "Can not find rgnData!" << std::endl ;
    doHd = false;
  }

//  if ( doHd ) {
  if (!rgnEmul.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection with label "
             << rctSourceEmul_.label() ;
    doHd = false;
    if (verbose_)std::cout << "Can not find rgnEmul!" << std::endl ;
  }
//  }

  
  e.getByLabel(rctSourceData_,emData);
  e.getByLabel(rctSourceEmul_,emEmul);
  
  if (!emData.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection with label "
             << rctSourceData_.label() ;
    if (verbose_)std::cout << "Can not find emData!" << std::endl ;
    doEm = false; 
  }

//  if ( doEm ) {

  if (!emEmul.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection with label "
             << rctSourceEmul_.label() ;
    if (verbose_)std::cout << "Can not find emEmul!" << std::endl ;
    doEm = false; return ;
  }

//  }


  // Isolated and non-isolated EM
  
  // StepI: Reset

  int nelectrIsoData = 0;
  int nelectrNisoData = 0;
  int nelectrIsoEmul = 0;
  int nelectrNisoEmul = 0;

  int electronDataRank[2][PhiEtaMax]={0};
  int electronDataEta[2][PhiEtaMax]={0};
  int electronDataPhi[2][PhiEtaMax]={0};
  int electronEmulRank[2][PhiEtaMax]={0};
  int electronEmulEta[2][PhiEtaMax]={0};
  int electronEmulPhi[2][PhiEtaMax]={0};

    // region/bit arrays
  int nRegionData = 0;
  int nRegionEmul = 0;
    
  int regionDataRank[PhiEtaMax] = {0};
  int regionDataEta [PhiEtaMax] = {0};
  int regionDataPhi [PhiEtaMax] = {0};

  bool regionDataOverFlow [PhiEtaMax] = {false};
  bool regionDataTauVeto  [PhiEtaMax] = {false};
  bool regionDataMip      [PhiEtaMax] = {false};
  bool regionDataQuiet    [PhiEtaMax] = {false};
  bool regionDataFineGrain[PhiEtaMax] = {false};

  int regionEmulRank[PhiEtaMax] = {0};
  int regionEmulEta [PhiEtaMax] = {0};
  int regionEmulPhi [PhiEtaMax] = {0};

  bool regionEmulOverFlow [PhiEtaMax] = {false};
  bool regionEmulTauVeto  [PhiEtaMax] = {false};
  bool regionEmulMip      [PhiEtaMax] = {false};
  bool regionEmulQuiet    [PhiEtaMax] = {false};
  bool regionEmulFineGrain[PhiEtaMax] = {false};
    
// just to fix a scale for the ratios //
if(first)
{
  rctIsoEmEmulOcc_->Fill(0.,0.) ;
  rctIsoEmDataOcc_->Fill(0.,0.) ;
  rctIsoEmEff1Occ_->Fill(0.,0.) ;
  rctIsoEmEff2Occ_->Fill(0.,0.) ;
  rctIsoEmIneffOcc_->Fill(0.,0.) ;
  rctIsoEmOvereffOcc_->Fill(0.,0.) ;
  rctNisoEmEmulOcc_->Fill(0.,0.) ;
  rctNisoEmDataOcc_->Fill(0.,0.) ;
  rctNisoEmEff1Occ_->Fill(0.,0.) ;
  rctNisoEmEff2Occ_->Fill(0.,0.) ;
  rctNisoEmIneffOcc_->Fill(0.,0.) ;
  rctNisoEmOvereffOcc_->Fill(0.,0.) ;

  rctRegDataOcc2D_->Fill(0.,0.) ;
  rctRegEmulOcc2D_->Fill(0.,0.) ;
  rctRegMatchedOcc2D_->Fill(0.,0.) ;
  rctRegUnmatchedDataOcc2D_->Fill(0.,0.) ;
  rctRegUnmatchedEmulOcc2D_->Fill(0.,0.) ;
  rctRegDeltaEtOcc2D_->Fill(0.,0.) ;
  
  rctBitDataOverFlow2D_->Fill(0.,0.) ;
  rctBitEmulOverFlow2D_->Fill(0.,0.) ;
  rctBitMatchedOverFlow2D_->Fill(0.,0.) ;
  rctBitUnmatchedDataOverFlow2D_->Fill(0.,0.) ;
  rctBitUnmatchedEmulOverFlow2D_->Fill(0.,0.) ;

  rctBitDataTauVeto2D_->Fill(0.,0.) ;
  rctBitEmulTauVeto2D_->Fill(0.,0.) ;
  rctBitMatchedTauVeto2D_->Fill(0.,0.) ;
  rctBitUnmatchedDataTauVeto2D_->Fill(0.,0.) ;
  rctBitUnmatchedEmulTauVeto2D_->Fill(0.,0.) ;

  rctBitDataMip2D_->Fill(0.,0.) ;
  rctBitEmulMip2D_->Fill(0.,0.) ;
  rctBitMatchedMip2D_->Fill(0.,0.) ;
  rctBitUnmatchedDataMip2D_->Fill(0.,0.) ;
  rctBitUnmatchedEmulMip2D_->Fill(0.,0.) ;

  rctBitDataQuiet2D_->Fill(0.,0.) ;
  rctBitEmulQuiet2D_->Fill(0.,0.) ;
  rctBitMatchedQuiet2D_->Fill(0.,0.) ;
  rctBitUnmatchedDataQuiet2D_->Fill(0.,0.) ;
  rctBitUnmatchedEmulQuiet2D_->Fill(0.,0.) ;

  rctBitDataFineGrain2D_->Fill(0.,0.) ;
  rctBitEmulFineGrain2D_->Fill(0.,0.) ;
  rctBitMatchedFineGrain2D_->Fill(0.,0.) ;
  rctBitUnmatchedDataFineGrain2D_->Fill(0.,0.) ;
  rctBitUnmatchedEmulFineGrain2D_->Fill(0.,0.) ;
  
  first = false ;
}


  // StepII: fill variables

  for (L1CaloEmCollection::const_iterator iem = emEmul->begin();
       iem != emEmul->end();
       iem++)
  {
    if(iem->rank() >= 1)
    {
      if(iem->isolated())
      {
        rctIsoEmEmulOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi());

        // to  show bad channles in the 2D ineff
        rctIsoEmIneffOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi(),0.00001);

        int channel;

        channel=PHIBINS*iem->regionId().ieta()+iem->regionId().iphi();
        rctIsoEmEmulOcc1D_->Fill(channel);
        electronEmulRank[0][nelectrIsoEmul]=iem->rank();
        electronEmulEta[0][nelectrIsoEmul]=iem->regionId().ieta();
        electronEmulPhi[0][nelectrIsoEmul]=iem->regionId().iphi();
        nelectrIsoEmul++ ;
      }
      
      else
      {
        rctNisoEmEmulOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi());

        // to  show bad channles in the 2D ineff
        rctNisoEmIneffOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi(),0.00001);

        int channel;

        channel=PHIBINS*iem->regionId().ieta()+iem->regionId().iphi();
        rctNisoEmEmulOcc1D_->Fill(channel);
        electronEmulRank[1][nelectrNisoEmul]=iem->rank();
        electronEmulEta[1][nelectrNisoEmul]=iem->regionId().ieta();
        electronEmulPhi[1][nelectrNisoEmul]=iem->regionId().iphi();
        nelectrNisoEmul++ ;
      }
    }
  }

  for (L1CaloEmCollection::const_iterator iem = emData->begin();
       iem != emData->end();
       iem++)
  {
    if(iem->rank() >= 1)
    {
      if (iem->isolated())
      {
        rctIsoEmDataOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi());

        // new stuff to avoid 0's in emulator 2D //
        rctIsoEmEmulOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi(),0.00001);

        int channel;

        channel=PHIBINS*iem->regionId().ieta()+iem->regionId().iphi();
        rctIsoEmDataOcc1D_->Fill(channel);

        // new stuff to avoid 0's 
        rctIsoEmEmulOcc1D_->Fill(channel,0.0001);
        
        electronDataRank[0][nelectrIsoData]=iem->rank();
        electronDataEta[0][nelectrIsoData]=iem->regionId().ieta();
        electronDataPhi[0][nelectrIsoData]=iem->regionId().iphi();
        nelectrIsoData++ ;
      }
      
      else
      {
        rctNisoEmDataOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi());

        // new stuff to avoid 0's in emulator 2D //
        rctNisoEmEmulOcc_->Fill(iem->regionId().ieta(), iem->regionId().iphi(),0.00001);

        int channel;

        channel=PHIBINS*iem->regionId().ieta()+iem->regionId().iphi();
        rctNisoEmDataOcc1D_->Fill(channel);

        // new stuff to avoid 0's
        rctNisoEmEmulOcc1D_->Fill(channel,0.0001);

        electronDataRank[1][nelectrNisoData]=iem->rank();
        electronDataEta[1][nelectrNisoData]=iem->regionId().ieta();
        electronDataPhi[1][nelectrNisoData]=iem->regionId().iphi();
        nelectrNisoData++ ;
      }
    }
  }

    // fill region/bit arrays for emulator
  for(L1CaloRegionCollection::const_iterator ireg = rgnEmul->begin();
      ireg != rgnEmul->end();
      ireg++)
  {
    if(ireg->overFlow())  rctBitEmulOverFlow2D_ ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->tauVeto())   rctBitEmulTauVeto2D_  ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->mip())       rctBitEmulMip2D_      ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->quiet())     rctBitEmulQuiet2D_    ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->fineGrain()) rctBitEmulFineGrain2D_->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->et() > 0)
    {
      rctRegEmulOcc1D_->Fill(PHIBINS*ireg->gctEta() + ireg->gctPhi());
      rctRegEmulOcc2D_->Fill(ireg->gctEta(), ireg->gctPhi());
    }

    // to show bad channels in 2D inefficiency:
    if(ireg->overFlow())  rctBitUnmatchedEmulOverFlow2D_ ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->tauVeto())   rctBitUnmatchedEmulTauVeto2D_  ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->mip())       rctBitUnmatchedEmulMip2D_      ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->quiet())     rctBitUnmatchedEmulQuiet2D_    ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->fineGrain()) rctBitUnmatchedEmulFineGrain2D_->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->et() > 0)    rctRegUnmatchedEmulOcc2D_      ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);

    regionEmulRank     [nRegionEmul] = ireg->et();
    regionEmulEta      [nRegionEmul] = ireg->gctEta();
    regionEmulPhi      [nRegionEmul] = ireg->gctPhi();
    regionEmulOverFlow [nRegionEmul] = ireg->overFlow();
    regionEmulTauVeto  [nRegionEmul] = ireg->tauVeto();
    regionEmulMip      [nRegionEmul] = ireg->mip();
    regionEmulQuiet    [nRegionEmul] = ireg->quiet();
    regionEmulFineGrain[nRegionEmul] = ireg->fineGrain();
    
    nRegionEmul++;
  }

      // fill region/bit arrays for hardware
  for(L1CaloRegionCollection::const_iterator ireg = rgnData->begin();
      ireg != rgnData->end();
      ireg++)
  {
    if(ireg->overFlow())  rctBitDataOverFlow2D_ ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->tauVeto())   rctBitDataTauVeto2D_  ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->mip())       rctBitDataMip2D_      ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->quiet())     rctBitDataQuiet2D_    ->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->fineGrain()) rctBitDataFineGrain2D_->Fill(ireg->gctEta(), ireg->gctPhi());
    if(ireg->et() > 0)    rctRegDataOcc2D_      ->Fill(ireg->gctEta(), ireg->gctPhi());

    // to show bad channels in 2D inefficiency:
    if(ireg->overFlow())  rctBitEmulOverFlow2D_ ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->tauVeto())   rctBitEmulTauVeto2D_  ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->mip())       rctBitEmulMip2D_      ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->quiet())     rctBitEmulQuiet2D_    ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->fineGrain()) rctBitEmulFineGrain2D_->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);
    if(ireg->et() > 0)    rctRegEmulOcc2D_      ->Fill(ireg->gctEta(), ireg->gctPhi(), 0.00001);

    regionDataRank     [nRegionData] = ireg->et();
    regionDataEta      [nRegionData] = ireg->gctEta();
    regionDataPhi      [nRegionData] = ireg->gctPhi();
    regionDataOverFlow [nRegionData] = ireg->overFlow();
    regionDataTauVeto  [nRegionData] = ireg->tauVeto();
    regionDataMip      [nRegionData] = ireg->mip();
    regionDataQuiet    [nRegionData] = ireg->quiet();
    regionDataFineGrain[nRegionData] = ireg->fineGrain();
    
    nRegionData++;
  }

 if(verbose_)
{
  std::cout << "I found Data! Iso: " << nelectrIsoData << " Niso: " << nelectrNisoData <<  std::endl ;
  for(int i=0; i<nelectrNisoData; i++) 
  std::cout << " Energy " << electronDataRank[1][i] << " eta " << electronDataEta[1][i] << " phi " << electronDataPhi[1][i] << std::endl ;

  std::cout << "I found Emul! Iso: " << nelectrIsoEmul << " Niso: " << nelectrNisoEmul <<  std::endl ;
  for(int i=0; i<nelectrNisoEmul; i++) 
  std::cout << " Energy " << electronEmulRank[1][i] << " eta " << electronEmulEta[1][i] << " phi " << electronEmulPhi[1][i] << std::endl ;

  std::cout << "I found Data! Regions: " << nRegionData <<  std::endl ;
  for(int i=0; i<nRegionData; i++) 
 if(regionDataRank[i] !=0 )  std::cout << " Energy " << regionDataRank[i] << " eta " << regionDataEta[i] << " phi " << regionDataPhi[i] << std::endl ;

  std::cout << "I found Emul! Regions: " << nRegionEmul <<  std::endl ;
  for(int i=0; i<nRegionEmul; i++) 
 if(regionEmulRank[i] !=0 )  std::cout << " Energy " << regionEmulRank[i] << " eta " << regionEmulEta[i] << " phi " << regionEmulPhi[i] << std::endl ;
}

  // StepIII: calculate and fill

  for(int k=0; k<2; k++)
  {
    int nelectrE, nelectrD;
    
    if(k==0)
    {
      nelectrE=nelectrIsoEmul;
      nelectrD=nelectrIsoData;
    }
    
    else 
    {
      nelectrE=nelectrNisoEmul;
      nelectrD=nelectrNisoData;
    }

    for(int i = 0; i < nelectrE; i++)
    {
      Bool_t found = kFALSE;

      for(int j = 0; j < nelectrD; j++)
      {
        if(electronEmulEta[k][i]==electronDataEta[k][j] &&
           electronEmulPhi[k][i]==electronDataPhi[k][j])
        {
          if(k==0)
          {
            rctIsoEmEff1Occ_->Fill(electronEmulEta[k][i], electronEmulPhi[k][i]);

            int chnl;

            chnl=PHIBINS*electronEmulEta[k][i]+electronEmulPhi[k][i];
            rctIsoEmEff1Occ1D_->Fill(chnl);
            if(singlechannelhistos_)
            {
              int energy_difference;
              
              energy_difference=(electronEmulRank[k][i] - electronDataRank[k][j]);
              rctIsoEffChannel_[chnl]->Fill(energy_difference);
            }

            if(electronEmulRank[k][i]==electronDataRank[k][j])
            {
              rctIsoEmEff2Occ1D_->Fill(chnl);
              rctIsoEmEff2Occ_->Fill(electronEmulEta[k][i], electronEmulPhi[k][i]);
            }
            
            DivideME1D(rctIsoEmEff1Occ1D_, rctIsoEmEmulOcc1D_, rctIsoEmEff1oneD_);
            DivideME2D(rctIsoEmEff1Occ_, rctIsoEmEmulOcc_, rctIsoEmEff1_) ;
            DivideME1D(rctIsoEmEff2Occ1D_, rctIsoEmEmulOcc1D_, rctIsoEmEff2oneD_);
            DivideME2D(rctIsoEmEff2Occ_, rctIsoEmEmulOcc_, rctIsoEmEff2_) ;
          }
          
          else
          {
            rctNisoEmEff1Occ_->Fill(electronEmulEta[k][i], electronEmulPhi[k][i]);

            int chnl;

            chnl=PHIBINS*electronEmulEta[k][i]+electronEmulPhi[k][i];
            rctNisoEmEff1Occ1D_->Fill(chnl);
            if(singlechannelhistos_)
            {
              int energy_difference;

              energy_difference=(electronEmulRank[k][i] - electronDataRank[k][j]) ;
              rctNisoEffChannel_[chnl]->Fill(energy_difference) ;
            }

            if(electronEmulRank[k][i]==electronDataRank[k][j])
            {
              rctNisoEmEff2Occ1D_->Fill(chnl);
              rctNisoEmEff2Occ_->Fill(electronEmulEta[k][i], electronEmulPhi[k][i]);
            }
          
            DivideME1D(rctNisoEmEff1Occ1D_, rctNisoEmEmulOcc1D_, rctNisoEmEff1oneD_);
            DivideME2D(rctNisoEmEff1Occ_, rctNisoEmEmulOcc_, rctNisoEmEff1_);
            DivideME1D(rctNisoEmEff2Occ1D_, rctNisoEmEmulOcc1D_, rctNisoEmEff2oneD_);
            DivideME2D(rctNisoEmEff2Occ_, rctNisoEmEmulOcc_, rctNisoEmEff2_);
          }
          
          found = kTRUE;
        }
      }

      if(found == kFALSE)
      {
        if(k==0)
        {
          rctIsoEmIneffOcc_->Fill(electronEmulEta[k][i], electronEmulPhi[k][i]);

          int chnl;

          chnl=PHIBINS*electronEmulEta[k][i]+electronEmulPhi[k][i];
          rctIsoEmIneffOcc1D_->Fill(chnl);
          DivideME1D(rctIsoEmIneffOcc1D_, rctIsoEmEmulOcc1D_, rctIsoEmIneff1D_);
          DivideME2D(rctIsoEmIneffOcc_, rctIsoEmEmulOcc_, rctIsoEmIneff_);
          if(singlechannelhistos_)
          {
            rctIsoIneffChannel_[chnl]->Fill(electronEmulRank[k][i]);
          }
        }

        else
        {
          rctNisoEmIneffOcc_->Fill(electronEmulEta[k][i], electronEmulPhi[k][i]);

          int chnl;

          chnl=PHIBINS*electronEmulEta[k][i]+electronEmulPhi[k][i];
          rctNisoEmIneffOcc1D_->Fill(chnl);
          DivideME1D(rctNisoEmIneffOcc1D_, rctNisoEmEmulOcc1D_, rctNisoEmIneff1D_);
          DivideME2D(rctNisoEmIneffOcc_, rctNisoEmEmulOcc_, rctNisoEmIneff_);
          if(singlechannelhistos_)
          {
            rctNisoIneffChannel_[chnl]->Fill(electronEmulRank[k][i]);
          }
        }
      }
    }

    for(int i = 0; i < nelectrD; i++)
    {
      Bool_t found = kFALSE;

      for(int j = 0; j < nelectrE; j++) 
      {
        if(electronEmulEta[k][j]==electronDataEta[k][i] &&
           electronEmulPhi[k][j]==electronDataPhi[k][i])
        {
          found = kTRUE;
        }
      }

      if(found == kFALSE)
      {
        if(k==0)
        {
          rctIsoEmOvereffOcc_->Fill(electronDataEta[k][i], electronDataPhi[k][i]);

          int chnl;
          
          chnl=PHIBINS*electronDataEta[k][i]+electronDataPhi[k][i];
          rctIsoEmOvereffOcc1D_->Fill(chnl);

          // we try new definition of overefficiency:
          DivideME1D(rctIsoEmOvereffOcc1D_, rctIsoEmDataOcc1D_, rctIsoEmOvereff1D_);
          DivideME2D(rctIsoEmOvereffOcc_, rctIsoEmDataOcc_, rctIsoEmOvereff_);
          if(singlechannelhistos_)
          {
            rctIsoOvereffChannel_[chnl]->Fill(electronDataRank[k][i]);
          }
        }

        else
        {
          rctNisoEmOvereffOcc_->Fill(electronDataEta[k][i], electronDataPhi[k][i]);

          int chnl;

          chnl=PHIBINS*electronDataEta[k][i]+electronDataPhi[k][i];
          rctNisoEmOvereffOcc1D_->Fill(chnl) ;

          // we try new defintiion of overefficiency
          DivideME1D(rctNisoEmOvereffOcc1D_, rctNisoEmDataOcc1D_, rctNisoEmOvereff1D_);
          DivideME2D(rctNisoEmOvereffOcc_, rctNisoEmDataOcc_, rctNisoEmOvereff_);
          if(singlechannelhistos_)
          {
            rctNisoOvereffChannel_[chnl]->Fill(electronDataRank[k][i]);
          }
        }
      }
    }
  }

    // calculate region/bit information
  for(int i = 0; i < nRegionEmul; i++)
    if(regionEmulRank[i] >= 1)
    {
      Bool_t regFound      = kFALSE;
//       Bool_t overFlowFound = kFALSE;
//       Bool_t tauVetoFound  = kFALSE;

      for(int j = 0; j < nRegionData; j++)
        if(regionEmulEta[i] == regionDataEta[j] &&
           regionEmulPhi[i] == regionDataPhi[j])
        {
          if(regionDataRank[j] >= 1)
          {
            int chnl;

            chnl = PHIBINS*regionEmulEta[i] + regionEmulPhi[i];
            rctRegMatchedOcc1D_->Fill(chnl);
            rctRegMatchedOcc2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);

            if(singlechannelhistos_) rctRegEffChannel_[chnl]->Fill(regionEmulRank[i] - regionDataRank[j]);

            if(regionEmulRank[i] == regionDataRank[j]) rctRegDeltaEtOcc2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);

            DivideME1D(rctRegMatchedOcc1D_, rctRegEmulOcc1D_, rctRegEff1D_);
            DivideME2D(rctRegMatchedOcc2D_, rctRegEmulOcc2D_, rctRegEff2D_);
            DivideME2D(rctRegDeltaEtOcc2D_, rctRegEmulOcc2D_, rctRegSpEff2D_);

            regFound = kTRUE;
          }

//           if(regionEmulOverFlow[i] == true &&
//              regionDataOverFlow[j] == true)
//           {
//             rctBitMatchedOverFlow2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);
//             DivideME2D(rctBitMatchedOverFlow2D_, rctBitEmulOverFlow2D_, rctBitOverFlowEff2D_);
//             overFlowFound = kTRUE;
//           }
// 
//           if(regionEmulTauVeto[i] == true &&
//              regionDataTauVeto[j] == true)
//           {
//             rctBitMatchedTauVeto2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);
//             DivideME2D(rctBitMatchedTauVeto2D_, rctBitEmulTauVeto2D_, rctBitTauVetoEff2D_);
//             tauVetoFound = kTRUE;
//           }
        }

      if(regFound == kFALSE)
      {
        int chnl;

        chnl = PHIBINS*regionEmulEta[i] + regionEmulPhi[i];
        rctRegUnmatchedEmulOcc1D_->Fill(chnl);
        rctRegUnmatchedEmulOcc2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);

        DivideME1D(rctRegUnmatchedEmulOcc1D_, rctRegEmulOcc1D_, rctRegIneff1D_);
        DivideME2D(rctRegUnmatchedEmulOcc2D_, rctRegEmulOcc2D_, rctRegIneff2D_);

        if(singlechannelhistos_) rctRegIneffChannel_[chnl]->Fill(regionEmulRank[i]);
      }

//       if(overFlowFound == kFALSE)
//       {
//         rctBitUnmatchedEmulOverFlow2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);
//         DivideME2D(rctBitUnmatchedEmulOverFlow2D_, rctBitEmulOverFlow2D_, rctBitOverFlowIneff2D_);
//       }
// 
//       if(tauVetoFound == kFALSE)
//       {
//         rctBitUnmatchedEmulTauVeto2D_->Fill(regionEmulEta[i], regionEmulPhi[i]);
//         DivideME2D(rctBitUnmatchedEmulTauVeto2D_, rctBitEmulTauVeto2D_, rctBitTauVetoIneff2D_);
//       }
    }

  for(int i = 0; i < nRegionData; i++)
    if(regionDataRank[i] >= 1)
    {
      Bool_t regFound      = kFALSE;
//       Bool_t overFlowFound = kFALSE;
//       Bool_t tauVetoFound  = kFALSE;

      for(int j = 0; j < nRegionEmul; j++)
        if(regionEmulEta[j] == regionDataEta[i] &&
           regionEmulPhi[j] == regionDataPhi[i])
        {
          if(regionEmulRank[j] >= 1)
            regFound = kTRUE;

//           if(regionDataOverFlow[i] == true &&
//              regionEmulOverFlow[j] == true)
//             overFlowFound = kTRUE;
// 
//           if(regionDataTauVeto[i] == true &&
//              regionEmulTauVeto[j] == true)
//             tauVetoFound = kTRUE;
        }

      if(regFound == kFALSE)
      {
        int chnl;

        chnl = PHIBINS*regionDataEta[i] + regionDataPhi[i];
        rctRegUnmatchedDataOcc1D_->Fill(chnl);
        rctRegUnmatchedDataOcc2D_->Fill(regionDataEta[i], regionDataPhi[i]);

        // we try a new definition of overefficiency:
        DivideME2D(rctRegUnmatchedDataOcc2D_, rctRegDataOcc2D_, rctRegOvereff2D_);

        if(singlechannelhistos_) rctRegOvereffChannel_[chnl]->Fill(regionDataRank[i]);
      }

//       if(overFlowFound == kFALSE)
//       {
//         rctBitUnmatchedDataOverFlow2D_->Fill(regionDataEta[i], regionDataPhi[i]);
//         DivideME2D(rctBitUnmatchedDataOverFlow2D_, rctBitDataOverFlow2D_, rctBitOverFlowOvereff2D_);
//       }
// 
//       if(tauVetoFound == kFALSE)
//       {
//         rctBitUnmatchedDataTauVeto2D_->Fill(regionDataEta[i], regionDataPhi[i]);
//         DivideME2D(rctBitUnmatchedDataTauVeto2D_, rctBitDataTauVeto2D_, rctBitTauVetoOvereff2D_);
//       }
    }
}

void L1TdeRCT::DivideME2D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result){

   TH2F* num = numerator->getTH2F();
   TH2F* den = denominator->getTH2F();
   TH2F* res = result->getTH2F();

   res->Divide(num,den,1,1,"");

}

void L1TdeRCT::DivideME1D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result){

   TH1F* num = numerator->getTH1F();
   TH1F* den = denominator->getTH1F();
   TH1F* res = result->getTH1F();

   res->Divide(num,den,1,1,"");
   
}

