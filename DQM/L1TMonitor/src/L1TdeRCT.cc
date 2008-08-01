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

#include "TF2.h"

#include <iostream>

using namespace edm;

const unsigned int PHIBINS = 18;
const float PHIMIN = -0.5;
const float PHIMAX = 17.5;

const unsigned int ETABINS = 22;
const float ETAMIN = -0.5;
const float ETAMAX = 21.5;

const unsigned int DEBINS = 101;
const float DEMIN = -50.5;
const float DEMAX = 50.5;

const unsigned int PhiEtaMax = 396;
const unsigned int CHNLBINS = 396;
const float CHNLMIN = -0.5;
const float CHNLMAX = 395.5;


L1TdeRCT::L1TdeRCT(const ParameterSet & ps) :
   rctSourceData_( ps.getParameter< InputTag >("rctSourceData") ),
   rctSourceEmul_( ps.getParameter< InputTag >("rctSourceEmul") )

{


  singlechannelhistos_ = ps.getUntrackedParameter < bool > ("singlechannelhistos", false);
                                                                                                                        
  if (singlechannelhistos_)
    std::cout << "L1TdeRCT: single channels histos ON" << std::endl;
                                                                                                                        
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

// for single channels 

    if(singlechannelhistos_)
   {
    for(int m=0; m<6; m++)
    {
    if(m==0) dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData/Eff1SnglChnls");
    if(m==1) dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData/Eff1SnglChnls");
    if(m==2) dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData/IneffSnglChnls");
    if(m==3) dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData/IneffSnglChnls");
    if(m==4) dbe->setCurrentFolder(histFolder_+"IsoEm/ServiceData/OvereffSnglChnls");
    if(m==5) dbe->setCurrentFolder(histFolder_+"NisoEm/ServiceData/OvereffSnglChnls");

    for(int i=0; i<ETAMAX; i++)
    {
     for(int j=0; j<PHIMAX; j++)
     {
     char name[80], channel[80]={""} ;

     if(m==0) strcpy(name,"(Eemul-Edata)Chnl") ;
     if(m==1) strcpy(name,"(Eemul-Edata)Chnl") ;
     if(m==2) strcpy(name,"EemulChnl") ;
     if(m==3) strcpy(name,"EemulChnl") ;
     if(m==4) strcpy(name,"EdataChnl") ;
     if(m==5) strcpy(name,"EdataChnl") ;

     if(i<10 && j<10) sprintf(channel,"_0%d0%d",i,j); 
     else if(i<10) sprintf(channel,"_0%d%d",i,j);
      else if(j<10) sprintf(channel,"_%d0%d",i,j);
       else sprintf(channel,"_%d%d",i,j);
     strcat(name,channel); 

     int chnl=18*i+j;

     if(m==0) rctIsoEffChannel_[chnl] =
	dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==1) rctNisoEffChannel_[chnl] =
	dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==2) rctIsoIneffChannel_[chnl] =
	dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==3) rctNisoIneffChannel_[chnl] =
	dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==4) rctIsoOvereffChannel_[chnl] =
	dbe->book1D(name, name, DEBINS, DEMIN, DEMAX);
     if(m==5) rctNisoOvereffChannel_[chnl] =
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


  // StepII: fill variables

  for (L1CaloEmCollection::const_iterator iem = emEmul->begin();
       iem != emEmul->end(); iem++) {
   if(iem->rank() >= 1){
    if (iem->isolated()) {
      rctIsoEmEmulOcc_->Fill(iem->regionId().ieta(),
			       iem->regionId().iphi());
      int channel; channel=18*iem->regionId().ieta()+iem->regionId().iphi();
      rctIsoEmEmulOcc1D_->Fill(channel);
      electronEmulRank[0][nelectrIsoEmul]=iem->rank();
      electronEmulEta[0][nelectrIsoEmul]=iem->regionId().ieta();
      electronEmulPhi[0][nelectrIsoEmul]=iem->regionId().iphi();
      nelectrIsoEmul++ ;
    }
    else {
      rctNisoEmEmulOcc_->Fill(iem->regionId().ieta(),
			       iem->regionId().iphi());
      int channel; channel=18*iem->regionId().ieta()+iem->regionId().iphi();
      rctNisoEmEmulOcc1D_->Fill(channel);
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
      rctIsoEmDataOcc_->Fill(iem->regionId().ieta(),
			       iem->regionId().iphi());
      int channel; channel=18*iem->regionId().ieta()+iem->regionId().iphi();
      rctIsoEmDataOcc1D_->Fill(channel);
      electronDataRank[0][nelectrIsoData]=iem->rank();
      electronDataEta[0][nelectrIsoData]=iem->regionId().ieta();
      electronDataPhi[0][nelectrIsoData]=iem->regionId().iphi();
      nelectrIsoData++ ;
    }
    else {
      rctNisoEmDataOcc_->Fill(iem->regionId().ieta(),
			       iem->regionId().iphi());
      int channel; channel=18*iem->regionId().ieta()+iem->regionId().iphi();
      rctNisoEmDataOcc1D_->Fill(channel);
      electronDataRank[1][nelectrNisoData]=iem->rank();
      electronDataEta[1][nelectrNisoData]=iem->regionId().ieta();
      electronDataPhi[1][nelectrNisoData]=iem->regionId().iphi();
      nelectrNisoData++ ;
    }
   }
  }

// std::cout << "I found something! Iso: " << nelectrIsoEmul << " Niso: " << nelectrNisoEmul <<  std::endl ;

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
        int chnl; chnl=18*electronEmulEta[k][i]+electronEmulPhi[k][i]; 
        rctIsoEmEff1Occ1D_->Fill(chnl);
        if(singlechannelhistos_)
        {
        int energy_difference; 
          energy_difference=(electronEmulRank[k][i]-electronDataRank[k][j]) ;
        rctIsoEffChannel_[chnl]->Fill(energy_difference) ;
        }
        if(electronEmulRank[k][i]==electronDataRank[k][j]) {
        rctIsoEmEff2Occ1D_->Fill(chnl);
        rctIsoEmEff2Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;}
//        rctIsoEmEff1_->Divide(rctIsoEmEff1Occ_, rctIsoEmEmulOcc_, 1., 1.);
          DivideME1D(rctIsoEmEff1Occ1D_, rctIsoEmEmulOcc1D_,rctIsoEmEff1oneD_) ;
          DivideME2D(rctIsoEmEff1Occ_, rctIsoEmEmulOcc_,rctIsoEmEff1_) ;
//        rctIsoEmEff2_->Divide(rctIsoEmEff2Occ_, rctIsoEmEmulOcc_, 1., 1.);  
          DivideME1D(rctIsoEmEff2Occ1D_, rctIsoEmEmulOcc1D_,rctIsoEmEff2oneD_) ;
          DivideME2D(rctIsoEmEff2Occ_, rctIsoEmEmulOcc_,rctIsoEmEff2_) ;
         }
       else {
        rctNisoEmEff1Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
        int chnl; chnl=18*electronEmulEta[k][i]+electronEmulPhi[k][i];
        rctNisoEmEff1Occ1D_->Fill(chnl);
        if(singlechannelhistos_)
        {
        int energy_difference; 
          energy_difference=(electronEmulRank[k][i]-electronDataRank[k][j]) ;
        rctNisoEffChannel_[chnl]->Fill(energy_difference) ;
        }
        if(electronEmulRank[k][i]==electronDataRank[k][j]) {
        rctNisoEmEff2Occ1D_->Fill(chnl);
        rctNisoEmEff2Occ_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;}
//        rctNisoEmEff1_->Divide(rctNisoEmEff1Occ_, rctNisoEmEmulOcc_, 1., 1.);
          DivideME1D(rctNisoEmEff1Occ1D_, rctNisoEmEmulOcc1D_,rctNisoEmEff1oneD_) ;
          DivideME2D(rctNisoEmEff1Occ_, rctNisoEmEmulOcc_,rctNisoEmEff1_) ;
//        rctNisoEmEff2_->Divide(rctNisoEmEff2Occ_, rctNisoEmEmulOcc_, 1., 1.);
          DivideME1D(rctNisoEmEff2Occ1D_, rctNisoEmEmulOcc1D_,rctNisoEmEff2oneD_) ;
          DivideME2D(rctNisoEmEff2Occ_, rctNisoEmEmulOcc_,rctNisoEmEff2_) ;
          }
        found = kTRUE;
      }
   }

    if(found == kFALSE)
      {
       if(k==0){
        rctIsoEmIneffOcc_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
        int chnl; chnl=18*electronEmulEta[k][i]+electronEmulPhi[k][i];
        rctIsoEmIneffOcc1D_->Fill(chnl) ;
//        rctIsoEmIneff_->Divide(rctIsoEmIneffOcc_, rctIsoEmEmulOcc_, 1., 1.);
          DivideME1D(rctIsoEmIneffOcc1D_, rctIsoEmEmulOcc1D_,rctIsoEmIneff1D_) ;
          DivideME2D(rctIsoEmIneffOcc_, rctIsoEmEmulOcc_,rctIsoEmIneff_) ;
        if(singlechannelhistos_)
        {
         rctIsoIneffChannel_[chnl]->Fill(electronEmulRank[k][i]) ;
        }
          }
       else {
        rctNisoEmIneffOcc_->Fill(electronEmulEta[k][i],electronEmulPhi[k][i]) ;
        int chnl; chnl=18*electronEmulEta[k][i]+electronEmulPhi[k][i];
        rctNisoEmIneffOcc1D_->Fill(chnl) ;
//        rctNisoEmIneff_->Divide(rctNisoEmIneffOcc_, rctNisoEmEmulOcc_, 1., 1.);  
          DivideME1D(rctNisoEmIneffOcc1D_, rctNisoEmEmulOcc1D_,rctNisoEmIneff1D_) ;
          DivideME2D(rctNisoEmIneffOcc_, rctNisoEmEmulOcc_,rctNisoEmIneff_) ;
        if(singlechannelhistos_)
        {
         rctNisoIneffChannel_[chnl]->Fill(electronEmulRank[k][i]) ;
        }
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
        int chnl; chnl=18*electronDataEta[k][i]+electronDataPhi[k][i];
        rctIsoEmOvereffOcc1D_->Fill(chnl) ;
//        rctIsoEmOvereff_->Divide(rctIsoEmOvereffOcc_, rctIsoEmEmulOcc_, 1., 1.);
        DivideME1D(rctIsoEmOvereffOcc1D_, rctIsoEmEmulOcc1D_,rctIsoEmOvereff1D_) ;
        DivideME2D(rctIsoEmOvereffOcc_, rctIsoEmEmulOcc_,rctIsoEmOvereff_) ;
        if(singlechannelhistos_)
        {
         rctIsoOvereffChannel_[chnl]->Fill(electronDataRank[k][i]) ;
        }
         }
       else {
        rctNisoEmOvereffOcc_->Fill(electronDataEta[k][i],electronDataPhi[k][i]) ;
        int chnl; chnl=18*electronDataEta[k][i]+electronDataPhi[k][i]; 
        rctNisoEmOvereffOcc1D_->Fill(chnl) ;
//        rctNisoEmOvereff_->Divide(rctNisoEmOvereffOcc_, rctNisoEmEmulOcc_, 1., 1.);  
        DivideME1D(rctNisoEmOvereffOcc1D_, rctNisoEmEmulOcc1D_,rctNisoEmOvereff1D_) ;
        DivideME2D(rctNisoEmOvereffOcc_, rctNisoEmEmulOcc_,rctNisoEmOvereff_) ;
        if(singlechannelhistos_)
        {
         rctNisoOvereffChannel_[chnl]->Fill(electronDataRank[k][i]) ;
        }
         }
      }
  }
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
