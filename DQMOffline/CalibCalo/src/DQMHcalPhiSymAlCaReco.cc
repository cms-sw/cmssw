/*
 * \file DQMHcalPhiSymAlCaReco.cc
 *
 * \author Olga Kodolova
 *        
 * $Date: 2008/08/13 09:20:27 $
 * $Revision: 1.1 $
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
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/CalibCalo/src/DQMHcalPhiSymAlCaReco.h"

using namespace std;
using namespace edm;

// ******************************************
// constructors
// *****************************************

DQMHcalPhiSymAlCaReco::DQMHcalPhiSymAlCaReco( const edm::ParameterSet& ps ) :
eventCounter_(0)
{
  dbe_ = Service<DQMStore>().operator->();
//
// Input from configurator file 
//
  folderName_ = ps.getUntrackedParameter<string>("FolderName","ALCAStreamHcalPhiSym");
  
  hbherecoMB = ps.getParameter<edm::InputTag>("hbheInputMB");
  horecoMB   = ps.getParameter<edm::InputTag>("hoInputMB");
  hfrecoMB   = ps.getParameter<edm::InputTag>("hfInputMB");
  
  hbherecoNoise = ps.getParameter<edm::InputTag>("hbheInputNoise");
  horecoNoise   = ps.getParameter<edm::InputTag>("hoInputNoise");
  hfrecoNoise   = ps.getParameter<edm::InputTag>("hfInputNoise");

  saveToFile_=ps.getUntrackedParameter<bool>("SaveToFile",false);
  fileName_=  ps.getUntrackedParameter<string>("FileName","MonitorAlCaHcalPhiSym.root");

  // histogram parameters

  // Distribution of rechits in iPhi, iEta 
  hiDistr_y_nbin_= ps.getUntrackedParameter<int>("hiDistr_y_nbin",72);
  hiDistr_y_min_=  ps.getUntrackedParameter<double>("hiDistr_y_min",1.);
  hiDistr_y_max_=  ps.getUntrackedParameter<double>("hiDistr_y_max",72.);
  hiDistr_x_nbin_= ps.getUntrackedParameter<int>("hiDistr_x_nbin",41);
  hiDistr_x_min_=  ps.getUntrackedParameter<double>("hiDistr_x_min",1.);
  hiDistr_x_max_=  ps.getUntrackedParameter<double>("hiDistr_x_max",41.);
    
}

DQMHcalPhiSymAlCaReco::~DQMHcalPhiSymAlCaReco()
{}

//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::beginJob(const EventSetup& context){
   std::cout<<" DQMHcalPhiSymAlCaReco::beginJob::start "<<std::endl;

  // create and cd into new folder
  dbe_->setCurrentFolder(folderName_);

  // book some histograms 1D
  hiDistrMBPl2D_ = 
    dbe_->book2D("MBdepthPl1", "iphi- +ieta signal distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrMBPl2D_->setAxisTitle("i#phi ", 1);
  hiDistrMBPl2D_->setAxisTitle("# rechits", 2);


  hiDistrNoisePl2D_ = 
    dbe_->book2D("NoisedepthPl1", "iphi-ieta noise distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrNoisePl2D_->setAxisTitle("i#phi ", 1);
  hiDistrNoisePl2D_->setAxisTitle("# rechits", 2);

//==================================================================================

  hiDistrMBMin2D_ = 
    dbe_->book2D("MBdepthMin1", "iphi- +ieta signal distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrMBMin2D_->setAxisTitle("i#phi ", 1);
  hiDistrMBMin2D_->setAxisTitle("# rechits", 2);


  hiDistrNoiseMin2D_ = 
    dbe_->book2D("NoisedepthMin1", "iphi-ieta noise distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrNoiseMin2D_->setAxisTitle("i#phi ", 1);
  hiDistrNoiseMin2D_->setAxisTitle("# rechits", 2);

  std::cout<<" DQMHcalPhiSymAlCaReco::beginJob::end "<<std::endl;

}

//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

//-------------------------------------------------------------

void DQMHcalPhiSymAlCaReco::analyze(const Event& iEvent, 
			       const EventSetup& iSetup ){  
 
//  if (eventCounter_>400) return; 
   std::cout<<" Event number "<<eventCounter_<<std::endl;
   eventCounter_++;
  
   edm::Handle<HBHERecHitCollection> hbheNS;
   iEvent.getByLabel(hbherecoNoise, hbheNS);

   if(!hbheNS.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
     cout<<" No HBHE MS "<<endl;
     return ;
   }
   
  
   
  const HBHERecHitCollection HithbheNS = *(hbheNS.product());
  
  std::cout<<" DQMHcalPhiSymAlCaReco::analyze::hbheNS.size "<<HithbheNS.size()<<std::endl;
    
  for(HBHERecHitCollection::const_iterator hbheItr=HithbheNS.begin(); hbheItr!=HithbheNS.end(); hbheItr++)
        {
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrNoisePl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 } else {
		 hiDistrNoiseMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
		 }
		 }
        }

   edm::Handle<HBHERecHitCollection> hbheMB;
   iEvent.getByLabel(hbherecoMB, hbheMB);

   if(!hbheMB.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
     cout<<" No HBHE MB"<<endl;
     return ;
   }

  const HBHERecHitCollection HithbheMB = *(hbheMB.product());

  for(HBHERecHitCollection::const_iterator hbheItr=HithbheMB.begin(); hbheItr!=HithbheMB.end(); hbheItr++)
        {
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrMBPl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 } else {
		 hiDistrMBMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
		 }
		 }

        }
   edm::Handle<HFRecHitCollection> hfNS;
   iEvent.getByLabel(hfrecoNoise, hfNS);

   if(!hfNS.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
     cout<<" No HF NS "<<endl;
     return ;
   }
  const HFRecHitCollection HithfNS = *(hfNS.product());
  cout<<" HFE NS size of collection "<<HithfNS.size()<<endl;
  
  for(HFRecHitCollection::const_iterator hbheItr=HithfNS.begin(); hbheItr!=HithfNS.end(); hbheItr++)
        {
	
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrNoisePl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 } else {
		 hiDistrNoiseMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
		 }
		 }
	
        }
   edm::Handle<HFRecHitCollection> hfMB;
   iEvent.getByLabel(hfrecoMB, hfMB);

   if(!hfMB.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
     cout<<" No HBHE MB"<<endl;
     return ;
   }

  const HFRecHitCollection HithfMB = *(hfMB.product());
  cout<<" HF MB size of collection "<<HithfMB.size()<<endl;

  for(HFRecHitCollection::const_iterator hbheItr=HithfMB.begin(); hbheItr!=HithfMB.end(); hbheItr++)
        {
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrMBPl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 } else {
		 hiDistrMBMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
		 }
		 }
        }	
	
	
} //analyze




//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::endRun(const Run& r, const EventSetup& context){

}
//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::endJob(){
  
  if (saveToFile_) {
     dbe_->save(fileName_);
  }
  
}


