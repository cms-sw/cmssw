/*
 * \file DQMHcalPhiSymAlCaReco.cc
 *
 * \author Olga Kodolova
 *        
 * $Date: 2009/04/17 15:07:59 $
 * $Revision: 1.6 $
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
  hiDistr_y_min_=  ps.getUntrackedParameter<double>("hiDistr_y_min",0.5);
  hiDistr_y_max_=  ps.getUntrackedParameter<double>("hiDistr_y_max",72.5);
  hiDistr_x_nbin_= ps.getUntrackedParameter<int>("hiDistr_x_nbin",41);
  hiDistr_x_min_=  ps.getUntrackedParameter<double>("hiDistr_x_min",0.5);
  hiDistr_x_max_=  ps.getUntrackedParameter<double>("hiDistr_x_max",41.5);
    
}

DQMHcalPhiSymAlCaReco::~DQMHcalPhiSymAlCaReco()
{}

//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::beginJob(const EventSetup& context){
 
  // create and cd into new folder
  dbe_->setCurrentFolder(folderName_);

  // book some histograms 1D
  // First moment
  hiDistrMBPl2D_ = 
    dbe_->book2D("MBdepthPl1", "iphi- +ieta signal distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrMBPl2D_->setAxisTitle("i#phi ", 2);
  hiDistrMBPl2D_->setAxisTitle("i#eta ", 1);


  hiDistrNoisePl2D_ = 
    dbe_->book2D("NoisedepthPl1", "iphi-ieta noise distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrNoisePl2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoisePl2D_->setAxisTitle("i#eta ", 1);
// Second moment
  hiDistrMB2Pl2D_ =
    dbe_->book2D("MB2depthPl1", "iphi- +ieta signal distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrMB2Pl2D_->setAxisTitle("i#phi ", 2);
  hiDistrMB2Pl2D_->setAxisTitle("i#eta ", 1);


  hiDistrNoise2Pl2D_ =
    dbe_->book2D("Noise2depthPl1", "iphi-ieta noise distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrNoise2Pl2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoise2Pl2D_->setAxisTitle("i#eta ", 1);

// Variance
  hiDistrVarMBPl2D_ =
    dbe_->book2D("VarMBdepthPl1", "iphi- +ieta signal distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrVarMBPl2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarMBPl2D_->setAxisTitle("i#eta ", 1);


  hiDistrVarNoisePl2D_ =
    dbe_->book2D("VarNoisedepthPl1", "iphi-ieta noise distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrVarNoisePl2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarNoisePl2D_->setAxisTitle("i#eta ", 1);

//==================================================================================
// First moment
  hiDistrMBMin2D_ = 
    dbe_->book2D("MBdepthMin1", "iphi- +ieta signal distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrMBMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrMBMin2D_->setAxisTitle("i#eta ", 1);


  hiDistrNoiseMin2D_ = 
    dbe_->book2D("NoisedepthMin1", "iphi-ieta noise distribution at depth1",
		 hiDistr_x_nbin_, 
		 hiDistr_x_min_,
		 hiDistr_x_max_,
		 hiDistr_y_nbin_, 
		 hiDistr_y_min_,
		 hiDistr_y_max_
		 );

  hiDistrNoiseMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoiseMin2D_->setAxisTitle("i#eta ", 1);
// Second moment
  hiDistrMB2Min2D_ =
    dbe_->book2D("MB2depthMin1", "iphi- +ieta signal distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrMB2Min2D_->setAxisTitle("i#phi ", 2);
  hiDistrMB2Min2D_->setAxisTitle("i#eta ", 1);


  hiDistrNoise2Min2D_ =
    dbe_->book2D("Noise2depthMin1", "iphi-ieta noise distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrNoise2Min2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoise2Min2D_->setAxisTitle("i#eta ", 1);

// Variance
  hiDistrVarMBMin2D_ =
    dbe_->book2D("VarMBdepthMin1", "iphi- +ieta signal distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrVarMBMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarMBMin2D_->setAxisTitle("i#eta ", 1);


  hiDistrVarNoiseMin2D_ =
    dbe_->book2D("VarNoisedepthMin1", "iphi-ieta noise distribution at depth1",
                 hiDistr_x_nbin_,
                 hiDistr_x_min_,
                 hiDistr_x_max_,
                 hiDistr_y_nbin_,
                 hiDistr_y_min_,
                 hiDistr_y_max_
                 );

  hiDistrVarNoiseMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarNoiseMin2D_->setAxisTitle("i#eta ", 1);


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
 

   eventCounter_++;
  
   edm::Handle<HBHERecHitCollection> hbheNS;
   iEvent.getByLabel(hbherecoNoise, hbheNS);

   if(!hbheNS.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
      return ;
   }
   
  
   
  const HBHERecHitCollection HithbheNS = *(hbheNS.product());
  
     
  for(HBHERecHitCollection::const_iterator hbheItr=HithbheNS.begin(); hbheItr!=HithbheNS.end(); hbheItr++)
        {
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrNoisePl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 hiDistrNoise2Pl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy()*hbheItr->energy());
                 } else {
		 hiDistrNoiseMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
                 hiDistrNoise2Min2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy()*hbheItr->energy());
		 }
		 }
        }

   edm::Handle<HBHERecHitCollection> hbheMB;
   iEvent.getByLabel(hbherecoMB, hbheMB);

   if(!hbheMB.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
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
                 hiDistrMB2Pl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy()*hbheItr->energy());
                 } else {
		 hiDistrMBMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
                 hiDistrMB2Min2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy()*hbheItr->energy());
		 }
		 }

        }
   edm::Handle<HFRecHitCollection> hfNS;
   iEvent.getByLabel(hfrecoNoise, hfNS);

   if(!hfNS.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
     return ;
   }
  const HFRecHitCollection HithfNS = *(hfNS.product());
  
  for(HFRecHitCollection::const_iterator hbheItr=HithfNS.begin(); hbheItr!=HithfNS.end(); hbheItr++)
        {
	
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrNoisePl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 hiDistrNoise2Pl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy()*hbheItr->energy());
                 } else {
		 hiDistrNoiseMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
                 hiDistrNoise2Min2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy()*hbheItr->energy());
		 }
		 }
	
        }
   edm::Handle<HFRecHitCollection> hfMB;
   iEvent.getByLabel(hfrecoMB, hfMB);

   if(!hfMB.isValid()){
     LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
      return ;
   }

  const HFRecHitCollection HithfMB = *(hfMB.product());
 
  for(HFRecHitCollection::const_iterator hbheItr=HithfMB.begin(); hbheItr!=HithfMB.end(); hbheItr++)
        {
        	 DetId id = (*hbheItr).detid(); 
	         HcalDetId hid=HcalDetId(id);
                 
		 if(hid.depth() == 1) {
                 if( hid.ieta() > 0 ) {
		 hiDistrMBPl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy());
                 hiDistrMB2Pl2D_->Fill(hid.ieta(),hid.iphi(),hbheItr->energy()*hbheItr->energy());
                 } else {
		 hiDistrMBMin2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy());
                 hiDistrMB2Min2D_->Fill(fabs(hid.ieta()),hid.iphi(),hbheItr->energy()*hbheItr->energy());
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
  for(int k=0; k<=hiDistr_x_nbin_;k++)
  {
    for(int j=0; j<=hiDistr_y_nbin_;j++)
    {
// First moment
       float cc1=hiDistrMBPl2D_->getBinContent(k,j);
       cc1 = cc1 * 1./eventCounter_;
       hiDistrMBPl2D_->setBinContent(k,j,cc1); 
       float cc2=hiDistrNoisePl2D_->getBinContent(k,j);
       cc2 = cc2 * 1./eventCounter_;
       hiDistrNoisePl2D_->setBinContent(k,j,cc2);
       float cc3=hiDistrMBMin2D_->getBinContent(k,j);
       cc3 = cc3 * 1./eventCounter_;
       hiDistrMBMin2D_->setBinContent(k,j,cc3);
       float cc4=hiDistrNoiseMin2D_->getBinContent(k,j);
       cc4 = cc4 * 1./eventCounter_;
       hiDistrNoiseMin2D_->setBinContent(k,j,cc4);
// Second moment
       float cc11=hiDistrMB2Pl2D_->getBinContent(k,j);
       cc11 = cc11 * 1./eventCounter_;
       hiDistrMB2Pl2D_->setBinContent(k,j,cc11);
       hiDistrVarMBPl2D_->setBinContent(k,j,cc11-cc1*cc1);
       float cc22=hiDistrNoise2Pl2D_->getBinContent(k,j);
       cc22 = cc22 * 1./eventCounter_;
       hiDistrNoise2Pl2D_->setBinContent(k,j,cc22);
       hiDistrVarNoisePl2D_->setBinContent(k,j,cc22-cc2*cc2);
       float cc33=hiDistrMB2Min2D_->getBinContent(k,j);
       cc33 = cc33 * 1./eventCounter_;
       hiDistrMB2Min2D_->setBinContent(k,j,cc33);
       hiDistrVarMBMin2D_->setBinContent(k,j,cc33-cc3*cc3);
       float cc44=hiDistrNoise2Min2D_->getBinContent(k,j);
       cc44 = cc44 * 1./eventCounter_;
       hiDistrNoise2Min2D_->setBinContent(k,j,cc44);
       hiDistrVarNoiseMin2D_->setBinContent(k,j,cc44-cc4*cc4);
    }
  }

}
//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::endJob(){
  if (saveToFile_) {
     dbe_->save(fileName_);
  }
}


