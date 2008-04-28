/*
 * \file L1TdeRCT.cc
 *
 * $Date: 2008/03/20 19:38:25 $
 * $Revision: 1.13 $
 * \author P. Wittich
 * 
 * version 0.0 A.Savin 2008/04/26
 *
 */

#include "DQM/L1TMonitor/interface/L1TdeRCT.h"

// GCT and RCT data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TF2.h"

#include <iostream>

using namespace edm;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

const unsigned int PhiEtaMax = 396;


L1TdeRCT::L1TdeRCT(const ParameterSet & ps) :
   rctSourceData_( ps.getParameter< InputTag >("rctSourceData") ),
   rctSourceEmul_( ps.getParameter< InputTag >("rctSourceEmul") )

{

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
    std::
	cout << "L1T Monitoring histograms will be saved to " <<
	outputFile_.c_str() << std::endl;
  }

  bool disable =
      ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }


  if (dbe != NULL) {
    dbe->setCurrentFolder("L1T/L1TdeRCT");
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
    dbe->setCurrentFolder("L1T/L1TdeRCT");
    dbe->rmdir("L1T/L1TdeRCT");
  }


  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TdeRCT");

    rctIsoEmEmulOcc_ =
	dbe->book2D("rctIsoEmEmulOcc", "rctIsoEmEmulOcc", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff1Occ_ =
	dbe->book2D("rctIsoEmEff1Occ", "rctIsoEmEff1Occ", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff2Occ_ =
	dbe->book2D("rctIsoEmEff2Occ", "rctIsoEmEff2Occ", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmIneffOcc_ =
	dbe->book2D("rctIsoEmIneffOcc", "rctIsoEmIneffOcc", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmOvereffOcc_ =
	dbe->book2D("rctIsoEmOvereffOcc", "rctIsoEmOvereffOcc", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff1_ =
	dbe->book2D("rctIsoEmEff1", "rctIsoEmEff1", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmEff2_ =
	dbe->book2D("rctIsoEmEff2", "rctIsoEmEff2", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctIsoEmIneff_ =
	dbe->book2D("rctIsoEmIneff", "rctIsoEmIneff", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);
    rctIsoEmOvereff_ =
	dbe->book2D("rctIsoEmOvereff", "rctIsoEmOvereff", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);


    rctNisoEmEmulOcc_ =
	dbe->book2D("rctNisoEmEmulOcc", "rctNisoEmEmulOcc", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff1Occ_ =
	dbe->book2D("rctNisoEmEff1Occ", "rctNisoEmEff1Occ", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff2Occ_ =
	dbe->book2D("rctNisoEmEff2Occ", "rctNisoEmEff2Occ", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmIneffOcc_ =
	dbe->book2D("rctNisoEmIneffOcc", "rctNisoEmIneffOcc", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmOvereffOcc_ =
	dbe->book2D("rctNisoEmOvereffOcc", "rctNisoEmOvereffOcc", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff1_ =
	dbe->book2D("rctNisoEmEff1", "rctNisoEmEff1", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmEff2_ =
	dbe->book2D("rctNisoEmEff2", "rctNisoEmEff2", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmIneff_ =
	dbe->book2D("rctNisoEmIneff", "rctNisoEmIneff", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);

    rctNisoEmOvereff_ =
	dbe->book2D("rctNisoEmOvereff", "rctNisoEmOvereff", ETABINS, ETAMIN,
		    ETAMAX, PHIBINS, PHIMIN, PHIMAX);



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
    std::cout << "I am here!" << std::endl ;
  nev_++;
  if (verbose_) {
    std::cout << "L1TdeRCT: analyze...." << std::endl;
  }

  // Get the RCT digis
  edm::Handle < L1CaloEmCollection > emData;
  edm::Handle < L1CaloRegionCollection > rgnData;

  // Get the RCT digis
  edm::Handle < L1CaloEmCollection > emEmul;
  edm::Handle < L1CaloRegionCollection > rgnEmul;

  // need to change to getByLabel
  bool doEm = true; 
  bool doHd = true;

  
  e.getByLabel(rctSourceData_,rgnData);
  e.getByLabel(rctSourceEmul_,rgnEmul);
 
  if (!rgnData.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection with label "
			       << rctSourceData_.label() ;
    std::cout << "Can not find rgnData!" << std::endl ;
    doHd = false;
  }

//  if ( doHd ) {
  if (!rgnEmul.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloRegionCollection with label "
			       << rctSourceEmul_.label() ;
    doHd = false;
    std::cout << "Can not find rgnEmul!" << std::endl ;
  }
//  }

  
  e.getByLabel(rctSourceData_,emData);
  e.getByLabel(rctSourceEmul_,emEmul);
  
  if (!emData.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection with label "
			       << rctSourceData_.label() ;
    std::cout << "Can not find emData!" << std::endl ;
    doEm = false; 
  }

//  if ( doEm ) {

  if (!emEmul.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1CaloEmCollection with label "
			       << rctSourceEmul_.label() ;
    std::cout << "Can not find emEmul!" << std::endl ;
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


  // StepII: fill variables

  for (L1CaloEmCollection::const_iterator iem = emEmul->begin();
       iem != emEmul->end(); iem++) {
   if(iem->rank() >= 1){
    if (iem->isolated()) {
      rctIsoEmEmulOcc_->Fill(iem->regionId().ieta(),
			       iem->regionId().iphi());
      electronEmulRank[0][nelectrIsoEmul]=iem->rank();
      electronEmulEta[0][nelectrIsoEmul]=iem->regionId().ieta();
      electronEmulPhi[0][nelectrIsoEmul]=iem->regionId().iphi();
      nelectrIsoEmul++ ;
    }
    else {
      rctNisoEmEmulOcc_->Fill(iem->regionId().ieta(),
			       iem->regionId().iphi());
      electronEmulRank[1][nelectrNisoEmul]=iem->rank();
      electronEmulEta[1][nelectrNisoEmul]=iem->regionId().ieta();
      electronEmulPhi[1][nelectrNisoEmul]=iem->regionId().iphi();
      nelectrNisoEmul++ ;
    }
   }
  }

  for (L1CaloEmCollection::const_iterator iem = emData->begin();
       iem != emData->end(); iem++) {
   if(iem->rank() >= 1){
    if (iem->isolated()) {
      electronDataRank[0][nelectrIsoData]=iem->rank();
      electronDataEta[0][nelectrIsoData]=iem->regionId().ieta();
      electronDataPhi[0][nelectrIsoData]=iem->regionId().iphi();
      nelectrIsoData++ ;
    }
    else {
      electronDataRank[1][nelectrNisoData]=iem->rank();
      electronDataEta[1][nelectrNisoData]=iem->regionId().ieta();
      electronDataPhi[1][nelectrNisoData]=iem->regionId().iphi();
      nelectrNisoData++ ;
    }
   }
  }

std::cout << "I found something! Iso: " << nelectrIsoEmul << " Niso: " << nelectrNisoEmul <<  std::endl ;

  // StepIII: calculate and fill

  for(int k=0; k<2; k++)
  {
  int nelectrE, nelectrD  ;
  if(k==0)
   { nelectrE=nelectrIsoEmul ; nelectrD=nelectrIsoData; }
  else 
   { nelectrE=nelectrNisoEmul ;  nelectrD=nelectrNisoData; }

  for(int i = 0; i < nelectrE; i++)
  {
    Bool_t found = kFALSE;

    for(int j = 0; j < nelectrD; j++)
   {
      if(electronEmulEta[k][i]==electronDataEta[k][j] && electronEmulPhi[k][i]==electronDataPhi[k][j])
      {
       if(k==0){
        rctIsoEmEff1Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
        if(electronEmulRank[k][i]==electronDataRank[k][j]) {
        rctIsoEmEff2Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;}
//        rctIsoEmEff1_->Divide(rctIsoEmEff1Occ_, rctIsoEmEmulOcc_, 1., 1.);
          DivideME(rctIsoEmEff1Occ_, rctIsoEmEmulOcc_,rctIsoEmEff1_) ;
//        rctIsoEmEff2_->Divide(rctIsoEmEff2Occ_, rctIsoEmEmulOcc_, 1., 1.);  
          DivideME(rctIsoEmEff2Occ_, rctIsoEmEmulOcc_,rctIsoEmEff2_) ;
         }
       else {
        rctNisoEmEff1Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
        if(electronEmulRank[k][i]==electronDataRank[k][j]) {
        rctNisoEmEff2Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;}
//        rctNisoEmEff1_->Divide(rctNisoEmEff1Occ_, rctNisoEmEmulOcc_, 1., 1.);
          DivideME(rctNisoEmEff1Occ_, rctNisoEmEmulOcc_,rctNisoEmEff1_) ;
//        rctNisoEmEff2_->Divide(rctNisoEmEff2Occ_, rctNisoEmEmulOcc_, 1., 1.);
          DivideME(rctNisoEmEff2Occ_, rctNisoEmEmulOcc_,rctNisoEmEff2_) ;
          }
        found = kTRUE;
      }
   }

    if(found == kFALSE)
      {
       if(k==0){
        rctIsoEmIneffOcc_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
//        rctIsoEmIneff_->Divide(rctIsoEmIneffOcc_, rctIsoEmEmulOcc_, 1., 1.);
          DivideME(rctIsoEmIneffOcc_, rctIsoEmEmulOcc_,rctIsoEmIneff_) ;
          }
       else {
        rctNisoEmIneffOcc_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
//        rctNisoEmIneff_->Divide(rctNisoEmIneffOcc_, rctNisoEmEmulOcc_, 1., 1.);  
          DivideME(rctNisoEmIneffOcc_, rctNisoEmEmulOcc_,rctNisoEmIneff_) ;
          }
      }
 }

  for(int i = 0; i < nelectrD; i++)
  {
    Bool_t found = kFALSE;

    for(int j = 0; j < nelectrE; j++) 
   {
      if(electronEmulEta[k][j]==electronDataEta[k][i] && electronEmulPhi[k][j]==electronDataPhi[k][i])
      {
        found = kTRUE;
      }
   }

    if(found == kFALSE)
      {
       if(k==0){
        rctIsoEmOvereffOcc_->Fill(electronDataEta[k][i],electronDataPhi[k][i]) ;
//        rctIsoEmOvereff_->Divide(rctIsoEmOvereffOcc_, rctIsoEmEmulOcc_, 1., 1.);
        DivideME(rctIsoEmOvereffOcc_, rctIsoEmEmulOcc_,rctIsoEmOvereff_) ;
         }
       else {
        rctNisoEmOvereffOcc_->Fill(electronDataEta[k][i],electronDataPhi[k][i]) ;
//        rctNisoEmOvereff_->Divide(rctNisoEmOvereffOcc_, rctNisoEmEmulOcc_, 1., 1.);  
        DivideME(rctNisoEmOvereffOcc_, rctNisoEmEmulOcc_,rctNisoEmOvereff_) ;
         }
      }
  }
 }


}

void L1TdeRCT::DivideME(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result){

   TH2F* num = numerator->getTH2F();
   TH2F* den = denominator->getTH2F();
   TH2F* res = result->getTH2F();

   res->Divide(num,den,1,1,"");
   
}
