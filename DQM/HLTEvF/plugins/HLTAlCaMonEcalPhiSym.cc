/*
 * \file HLTAlCaMonEcalPhiSym.cc
 *
 * \author Andrea Gozzelino - Universita%Gï¿½%@ e INFN Torino
 * \author Stefano Argiro
 *        
 * $Date: 2010/08/07 14:55:56 $
 * $Revision: 1.5 $
 *
 *
 * Description: Monitoring of Phi Symmetry Calibration Stream  
*/


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM include files

#include "DQMServices/Core/interface/MonitorElement.h"

// work on collections
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/HLTEvF/interface/HLTAlCaMonEcalPhiSym.h"


using namespace std;
using namespace edm;


// ******************************************
// constructors
// *****************************************

HLTAlCaMonEcalPhiSym::HLTAlCaMonEcalPhiSym( const edm::ParameterSet& ps ) :
eventCounter_(0)
{
  dbe_ = Service<DQMStore>().operator->();
  folderName_ = ps.getUntrackedParameter<std::string>("FolderName","ALCAStreamEcalPhiSym");
  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor",1);
  productMonitoredEB_= ps.getUntrackedParameter<edm::InputTag>("AlCaStreamEBTag");
  productMonitoredEE_= ps.getUntrackedParameter<edm::InputTag>("AlCaStreamEETag");

  saveToFile_=ps.getUntrackedParameter<bool>("SaveToFile",false);
  fileName_=  ps.getUntrackedParameter<std::string>("FileName","MonitorAlCaEcalPhiSym.root");

  // histogram parameters

  // Distribution of rechits in iPhi
  hiPhiDistrEB_nbin_= ps.getUntrackedParameter<int>("hiPhiDistrEB_nbin",361);
  hiPhiDistrEB_min_=  ps.getUntrackedParameter<double>("hiPhiDistrEB_min",1.);
  hiPhiDistrEB_max_=  ps.getUntrackedParameter<double>("hiPhiDistrEB_max",361.);
    
  // Distribution of rechits in iEta
  hiEtaDistrEB_nbin_= ps.getUntrackedParameter<int>("hiEtaDistrEB_nbin",171); 
  hiEtaDistrEB_min_ = ps.getUntrackedParameter<double>("hiEtaDistrEB_min",-85);
  hiEtaDistrEB_max_ = ps.getUntrackedParameter<double>("hiEtaDistrEB_max",85);
  
  // Energy Distribution of rechits  
  hRechitEnergyEB_nbin_=ps.getUntrackedParameter<int>("hRechitEnergyEB_nbin",160);
  hRechitEnergyEB_min_=ps.getUntrackedParameter<double>("hRechitEnergyEB_min",0.);
  hRechitEnergyEB_max_= ps.getUntrackedParameter<double>("hRechitEnergyEB_max",2.);
  
  // Distribution of total event energy
  hEventEnergyEB_nbin_= ps.getUntrackedParameter<int>("hEventEnergyEB_nbin",100);
  hEventEnergyEB_min_ = ps.getUntrackedParameter<double>("hEventEnergyEB_min",0.);
  hEventEnergyEB_max_ = ps.getUntrackedParameter<double>("hEventEnergyEB_max",20.);
  
  // Distribution of number of RecHits
  hNRecHitsEB_nbin_= ps.getUntrackedParameter<int>("hNRecHitsEB_nbin",100);
  hNRecHitsEB_min_ = ps.getUntrackedParameter<double>("hNRecHitsEB_min",0);
  hNRecHitsEB_max_ = ps.getUntrackedParameter<double>("hNRecHitsEB_max",250);
  
  // Distribution of Mean energy per rechit
  hMeanRecHitEnergyEB_nbin_= ps.getUntrackedParameter<int>("hMeanRecHitEnergyEB_nbin",50);
  hMeanRecHitEnergyEB_min_ = ps.getUntrackedParameter<int>("hMeanRecHitEnergyEB_min",0); 
  hMeanRecHitEnergyEB_max_ = ps.getUntrackedParameter<int>("hMeanRecHitEnergyEB_max",2);
  
    // Energy Distribution of rechits  
  hRechitEnergyEE_nbin_=ps.getUntrackedParameter<int>("hRechitEnergyEE_nbin",160);
  hRechitEnergyEE_min_ =ps.getUntrackedParameter<double>("hRechitEnergyEE_min",0.);
  hRechitEnergyEE_max_ =ps.getUntrackedParameter<double>("hRechitEnergyEE_max",3.);
  
  // Distribution of total event energy
  hEventEnergyEE_nbin_= ps.getUntrackedParameter<int>("hEventEnergyEE_nbin",100);
  hEventEnergyEE_min_ = ps.getUntrackedParameter<double>("hEventEnergyEE_min",0.);
  hEventEnergyEE_max_ = ps.getUntrackedParameter<double>("hEventEnergyEE_max",20.);
  
  // Distribution of number of RecHits
  hNRecHitsEE_nbin_= ps.getUntrackedParameter<int>("hNRecHitsEE_nbin",100);
  hNRecHitsEE_min_ = ps.getUntrackedParameter<double>("hNRecHitsEE_min",0);
  hNRecHitsEE_max_ = ps.getUntrackedParameter<double>("hNRecHitsEE_max",250);
  
  // Distribution of Mean energy per rechit
  hMeanRecHitEnergyEE_nbin_= ps.getUntrackedParameter<int>("hMeanRecHitEnergyEE_nbin",50);
  hMeanRecHitEnergyEE_min_ = ps.getUntrackedParameter<double>("hMeanRecHitEnergyEE_min",0); 
  hMeanRecHitEnergyEE_max_ = ps.getUntrackedParameter<double>("hMeanRecHitEnergyEE_max",5);

 

}


HLTAlCaMonEcalPhiSym::~HLTAlCaMonEcalPhiSym()
{}


//--------------------------------------------------------
void HLTAlCaMonEcalPhiSym::beginJob(){


  // create and cd into new folder
  dbe_->setCurrentFolder(folderName_);

  // book some histograms 1D
  hiPhiDistrEB_ = 
    dbe_->book1D("iphiDistributionEB", "RechitEB iphi",
		 hiPhiDistrEB_nbin_, 
		 hiPhiDistrEB_min_,
		 hiPhiDistrEB_max_);

  hiPhiDistrEB_->setAxisTitle("i#phi ", 1);
  hiPhiDistrEB_->setAxisTitle("# rechits", 2);


  hiEtaDistrEB_ = dbe_->book1D("iEtaDistributionEB", "RechitEB ieta",  
			       hiEtaDistrEB_nbin_,
			       hiEtaDistrEB_min_, 
			       hiEtaDistrEB_max_ );

  hiEtaDistrEB_->setAxisTitle("eta", 1);
  hiEtaDistrEB_->setAxisTitle("#rechits", 2);


  hRechitEnergyEB_ = dbe_->book1D("rhEnergyEB","rechits energy EB",
				  hRechitEnergyEB_nbin_,
				  hRechitEnergyEB_min_,
				  hRechitEnergyEB_max_);

  hRechitEnergyEB_->setAxisTitle("energy (GeV) ",1);
  hRechitEnergyEB_->setAxisTitle("#rechits",2);

  hEventEnergyEB_ = dbe_->book1D("eventEnergyEB","event energy EB",
				 hEventEnergyEB_nbin_,
				 hEventEnergyEB_min_,
				 hEventEnergyEB_max_);

  hEventEnergyEB_->setAxisTitle("energy (GeV) ",1);

  hNRecHitsEB_ = dbe_->book1D("nRechitsEB","#rechits in event EB",
			      hNRecHitsEB_nbin_,
			      hNRecHitsEB_min_,
			      hNRecHitsEB_max_);

  hNRecHitsEB_->setAxisTitle("rechits ",1);
 
  hMeanRecHitEnergyEB_ = dbe_->book1D("meanEnergyEB","Mean rechit energy EB",
				      hMeanRecHitEnergyEB_nbin_,
				      hMeanRecHitEnergyEB_min_,
				      hMeanRecHitEnergyEB_max_);

  hMeanRecHitEnergyEB_-> setAxisTitle("Mean Energy [GeV] ",1);
 
  
  hRechitEnergyEE_ = dbe_->book1D("rhEnergyEE","rechits energy EE",
				  hRechitEnergyEE_nbin_ ,
				  hRechitEnergyEE_min_ ,
				  hRechitEnergyEE_max_ );

  hRechitEnergyEE_->setAxisTitle("energy (GeV) ",1);
  hRechitEnergyEE_->setAxisTitle("#rechits",2);

  hEventEnergyEE_ = dbe_->book1D("eventEnergyEE","event energy EE",
				 hEventEnergyEE_nbin_,
				 hEventEnergyEE_min_,
				 hEventEnergyEE_max_);

  hEventEnergyEE_->setAxisTitle("energy (GeV) ",1);

  hNRecHitsEE_ = dbe_->book1D("nRechitsEE","#rechits in event EE" ,
			      hNRecHitsEE_nbin_,
			      hNRecHitsEE_min_,
			      hNRecHitsEE_max_);

  hNRecHitsEE_->setAxisTitle("rechits ",1);
 
  hMeanRecHitEnergyEE_ = dbe_->book1D("meanEnergyEE","Mean rechit energy EE",
				      hMeanRecHitEnergyEE_nbin_ ,
				      hMeanRecHitEnergyEE_min_ ,
				      hMeanRecHitEnergyEE_max_ );

  hMeanRecHitEnergyEE_-> setAxisTitle("Mean Energy [GeV] ",1);

}

//--------------------------------------------------------
void HLTAlCaMonEcalPhiSym::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void HLTAlCaMonEcalPhiSym::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

//-------------------------------------------------------------

void HLTAlCaMonEcalPhiSym::analyze(const Event& iEvent, 
			       const EventSetup& iSetup ){  
 
  if (eventCounter_% prescaleFactor_ ) return; 
  eventCounter_++;
    
  edm::Handle<EcalRecHitCollection> rhEB;
  edm::Handle<EcalRecHitCollection> rhEE;
 
  iEvent.getByLabel(productMonitoredEB_, rhEB); 
  iEvent.getByLabel(productMonitoredEE_, rhEE);

  EcalRecHitCollection::const_iterator itb;

  // fill EB histos
  if (rhEB.isValid()){
    float etot =0;
    for(itb=rhEB->begin(); itb!=rhEB->end(); ++itb){

      EBDetId id(itb->id());
      
      hiPhiDistrEB_->Fill(id.iphi());
      hiEtaDistrEB_->Fill(id.ieta());
      hRechitEnergyEB_->Fill(itb->energy());

      etot+= itb->energy();	 
    } // Eb rechits
    
    hNRecHitsEB_->Fill(rhEB->size());
    hMeanRecHitEnergyEB_->Fill(etot/rhEB->size());
    hEventEnergyEB_->Fill(etot);

  } // if valid

  // fill EE histos

  EcalRecHitCollection::const_iterator ite;

  if (rhEE.isValid()){

    float etot =0;
    for(ite=rhEE->begin(); ite!=rhEE->end(); ++ite){

      EEDetId id(ite->id());
      hRechitEnergyEE_->Fill(ite->energy());
      etot+= ite->energy();	 
    } // EE rechits
    
    hNRecHitsEE_->Fill(rhEE->size());
    hMeanRecHitEnergyEE_->Fill(etot/rhEE->size());
    hEventEnergyEE_->Fill(etot);
  }


} //analyze




//--------------------------------------------------------
void HLTAlCaMonEcalPhiSym::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void HLTAlCaMonEcalPhiSym::endRun(const Run& r, const EventSetup& context){

}
//--------------------------------------------------------
void HLTAlCaMonEcalPhiSym::endJob(){
  
  if (saveToFile_) {
     dbe_->save(fileName_);
  }
  
}


