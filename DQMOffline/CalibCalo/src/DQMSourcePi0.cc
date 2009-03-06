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
#include "DQMOffline/CalibCalo/src/DQMSourcePi0.h"


using namespace std;
using namespace edm;


// ******************************************
// constructors
// *****************************************

DQMSourcePi0::DQMSourcePi0( const edm::ParameterSet& ps ) :
eventCounter_(0)
{
  dbe_ = Service<DQMStore>().operator->();
  folderName_ = ps.getUntrackedParameter<string>("FolderName","HLT/AlCaEcalPi0");
  prescaleFactor_ = ps.getUntrackedParameter<int>("prescaleFactor",1);
  productMonitoredEB_= ps.getUntrackedParameter<edm::InputTag>("AlCaStreamEBTag");
  productMonitoredEE_= ps.getUntrackedParameter<edm::InputTag>("AlCaStreamEETag");
  isMonEB_ = ps.getUntrackedParameter<bool>("isMonEB",false);
  isMonEE_ = ps.getUntrackedParameter<bool>("isMonEE",false);

  saveToFile_=ps.getUntrackedParameter<bool>("SaveToFile",false);
  fileName_=  ps.getUntrackedParameter<string>("FileName","MonitorAlCaEcalPi0.root");

}


DQMSourcePi0::~DQMSourcePi0()
{}


//--------------------------------------------------------
void DQMSourcePi0::beginJob(const EventSetup& context){


  // create and cd into new folder
  dbe_->setCurrentFolder(folderName_);

  // book some histograms 1D

  hiPhiDistrEB_ = 
    dbe_->book1D("iphiDistributionEB", "RechitEB iphi", 361, 1,361);

  hiPhiDistrEB_->setAxisTitle("i#phi ", 1);
  hiPhiDistrEB_->setAxisTitle("# rechits", 2);


  hiEtaDistrEB_ = dbe_->book1D("iEtaDistributionEB", "RechitEB ieta", 171, -85, 86);
  hiEtaDistrEB_->setAxisTitle("eta", 1);
  hiEtaDistrEB_->setAxisTitle("#rechits", 2);


  hRechitEnergyEB_ = dbe_->book1D("rhEnergyEB","rechits energy EB",160,0.,2.0);
  hRechitEnergyEB_->setAxisTitle("energy (GeV) ",1);
  hRechitEnergyEB_->setAxisTitle("#rechits",2);

  hEventEnergyEB_ = dbe_->book1D("eventEnergyEB","event energy EB",100,0.,20.0);
  hEventEnergyEB_->setAxisTitle("energy (GeV) ",1);

  hNRecHitsEB_ = dbe_->book1D("nRechitsEB","#rechits in event EB",100,0.,250.);
  hNRecHitsEB_->setAxisTitle("rechits ",1);
 
  hMeanRecHitEnergyEB_ = dbe_->book1D("meanEnergyEB","Mean rechit energy EB",50,0.,2.);
  hMeanRecHitEnergyEB_-> setAxisTitle("Mean Energy [GeV] ",1);

 

  hRechitEnergyEE_ = dbe_->book1D("rhEnergyEE","rechits energy EE",160,0.,3.0);
  hRechitEnergyEE_->setAxisTitle("energy (GeV) ",1);
  hRechitEnergyEE_->setAxisTitle("#rechits",2);

  hEventEnergyEE_ = dbe_->book1D("eventEnergyEE","event energy EE",100,0.,20.0);
  hEventEnergyEE_->setAxisTitle("energy (GeV) ",1);

  hNRecHitsEE_ = dbe_->book1D("nRechitsEE","#rechits in event EE" ,100,0.,250.);
  hNRecHitsEE_->setAxisTitle("rechits ",1);
 
  hMeanRecHitEnergyEE_ = dbe_->book1D("meanEnergyEE","Mean rechit energy EE",50,0.,5.);
  hMeanRecHitEnergyEE_-> setAxisTitle("Mean Energy [GeV] ",1);
  

}

//--------------------------------------------------------
void DQMSourcePi0::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void DQMSourcePi0::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

//-------------------------------------------------------------

void DQMSourcePi0::analyze(const Event& iEvent, 
			       const EventSetup& iSetup ){  
 
  if (eventCounter_% prescaleFactor_ ) return; 
  eventCounter_++;
    
  edm::Handle<EcalRecHitCollection> rhEB;
  edm::Handle<EcalRecHitCollection> rhEE;
 
  if(isMonEB_) iEvent.getByLabel(productMonitoredEB_, rhEB); 
  if(isMonEE_) iEvent.getByLabel(productMonitoredEE_, rhEE);

  EcalRecHitCollection::const_iterator itb;

  // fill EB histos
  if(isMonEB_){
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

  } // if isMonEB

  // fill EE histos

  if(isMonEE_){  
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
    
  }//isMonEE
} //analyze




//--------------------------------------------------------
void DQMSourcePi0::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void DQMSourcePi0::endRun(const Run& r, const EventSetup& context){

}
//--------------------------------------------------------
void DQMSourcePi0::endJob(){

  if(dbe_) {  
    if (saveToFile_) {
      dbe_->save(fileName_);
    }
  }
}


