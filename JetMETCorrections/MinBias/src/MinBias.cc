// user include files
#include "JetMETCorrections/MinBias/interface/MinBias.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

using namespace std;
namespace cms
{
MinBias::MinBias(const edm::ParameterSet& iConfig)
{
  // get names of modules, producing object collections
  hbheLabel_= iConfig.getParameter<std::string>("hbheInput");
  hoLabel_= iConfig.getParameter<std::string>("hoInput");
  hfLabel_= iConfig.getParameter<std::string>("hfInput");
  hbheToken_= mayConsume<HBHERecHitCollection>(edm::InputTag(hbheLabel_));
  hoToken_=mayConsume<HORecHitCollection>(edm::InputTag(hoLabel_));
  hfToken_=mayConsume<HFRecHitCollection>(edm::InputTag(hfLabel_));
  allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile");

}


MinBias::~MinBias()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void MinBias::beginJob()
{
   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   myTree = new TTree("RecJet","RecJet Tree");
   myTree->Branch("mydet",  &mydet, "mydet/I");
   myTree->Branch("mysubd",  &mysubd, "mysubd/I");
   myTree->Branch("depth",  &depth, "depth/I");
   myTree->Branch("ieta",  &ieta, "ieta/I");
   myTree->Branch("iphi",  &iphi, "iphi/I");
   myTree->Branch("eta",  &eta, "eta/F");
   myTree->Branch("phi",  &phi, "phi/F");
   myTree->Branch("mom1",  &mom1, "mom1/F");
   myTree->Branch("mom2",  &mom2, "mom2/F");
   myTree->Branch("mom3",  &mom3, "mom3/F");
   myTree->Branch("mom4",  &mom4, "mom4/F");

   ievent = 0;

}

void MinBias::endJob()
{

   const std::vector<DetId>& did =  geo->getSubdetectorGeometry( DetId::Hcal, 1 )->getValidDetIds() ;
   int i=0;
   for(std::vector<DetId>::const_iterator id=did.begin(); id != did.end(); id++)
   {
//      if( (*id).det() == DetId::Hcal ) {
      GlobalPoint pos = geo->getPosition(*id);
      mydet = ((*id).rawId()>>28)&0xF;
      mysubd = ((*id).rawId()>>25)&0x7;
      depth = HcalDetId(*id).depth();
      ieta = HcalDetId(*id).ieta();
      iphi = HcalDetId(*id).iphi();
      phi = pos.phi();
      eta = pos.eta();
      if ( theFillDetMap0[*id] > 0. )
      {
      mom1 = theFillDetMap1[*id]/theFillDetMap0[*id];
      mom2 = theFillDetMap2[*id]/theFillDetMap0[*id]-(mom1*mom1);
      mom3 = theFillDetMap3[*id]/theFillDetMap0[*id]-3.*mom1*theFillDetMap2[*id]/theFillDetMap0[*id]+
             2.*pow(mom2,3);
      mom4 = (theFillDetMap4[*id]-4.*mom1*theFillDetMap3[*id]+6.*pow(mom1,2)*theFillDetMap2[*id])/theFillDetMap0[*id]-3.*pow(mom1,4);

      }	else
      {
       mom1 = 0.; mom2 = 0.; mom3 = 0.; mom4 = 0.;
      }
      std::cout<<" Detector "<<(*id).rawId()<<" mydet "<<mydet<<" "<<mysubd<<" "<<depth<<" "<<
      HcalDetId(*id).subdet()<<" "<<ieta<<" "<<iphi<<" "<<pos.eta()<<" "<<pos.phi()<<std::endl;
      std::cout<<" Energy "<<mom1<<" "<<mom2<<std::endl;
      myTree->Fill();
      i++;
//      }
   }
   std::cout<<" The number of CaloDet records "<<did.size()<<std::endl;
   std::cout<<" The number of Hcal records "<<i<<std::endl;


   std::cout << "===== Start writing user histograms =====" << std::endl;
   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   std::cout << "===== End writing user histograms =======" << std::endl;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MinBias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
   if(ievent == 0 ){
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   geo = pG.product();
   std::vector<DetId> did =  geo->getValidDetIds();

   for(std::vector<DetId>::const_iterator id=did.begin(); id != did.end(); id++)
   {
      if( (*id).det() == DetId::Hcal ) {
      theFillDetMap0[*id] = 0.;
      theFillDetMap1[*id] = 0.;
      theFillDetMap2[*id] = 0.;
      theFillDetMap3[*id] = 0.;
      theFillDetMap4[*id] = 0.;
   }
   }
   }



  if (!hbheLabel_.empty()) {
    edm::Handle<HBHERecHitCollection> hbhe;
    iEvent.getByToken(hbheToken_,hbhe);
    if (!hbhe.isValid()) {
      // can't find it!
      if (!allowMissingInputs_) {
	*hbhe;  // will throw the proper exception
      }
    } else {
      for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();
	  hbheItr != (*hbhe).end(); ++hbheItr)
	{
	  DetId id = (hbheItr)->detid();
	  if( (*hbheItr).energy() > 0. ) std::cout<<" Energy = "<<(*hbheItr).energy()<<std::endl;
	  theFillDetMap0[id] = theFillDetMap0[id]+ 1.;
	  theFillDetMap1[id] = theFillDetMap1[id]+(*hbheItr).energy();
	  theFillDetMap2[id] = theFillDetMap2[id]+pow((*hbheItr).energy(),2);
	  theFillDetMap3[id] = theFillDetMap3[id]+pow((*hbheItr).energy(),3);
	  theFillDetMap4[id] = theFillDetMap4[id]+pow((*hbheItr).energy(),4);
	}
    }
  }

  if (!hoLabel_.empty()) {
    edm::Handle<HORecHitCollection> ho;
    iEvent.getByToken(hoToken_,ho);
    if (!ho.isValid()) {
      // can't find it!
      if (!allowMissingInputs_) {
	*ho;  // will throw the proper exception
      }
  } else {
      for(HORecHitCollection::const_iterator hoItr = (*ho).begin();
	  hoItr != (*ho).end(); ++hoItr)
	{
	  DetId id = (hoItr)->detid();
	  theFillDetMap0[id] = theFillDetMap0[id]+ 1.;
	  theFillDetMap1[id] = theFillDetMap1[id]+(*hoItr).energy();
	  theFillDetMap2[id] = theFillDetMap2[id]+pow((*hoItr).energy(),2);
	  theFillDetMap3[id] = theFillDetMap3[id]+pow((*hoItr).energy(),3);
	  theFillDetMap4[id] = theFillDetMap4[id]+pow((*hoItr).energy(),4);
	}
    }
  }

  if (!hfLabel_.empty()) {
    edm::Handle<HFRecHitCollection> hf;
    iEvent.getByToken(hfToken_,hf);
    if (!hf.isValid()) {
      // can't find it!
      if (!allowMissingInputs_) {
	*hf;  // will throw the proper exception
      }
  } else {
      for(HFRecHitCollection::const_iterator hfItr = (*hf).begin();
	  hfItr != (*hf).end(); ++hfItr)
	{
	  DetId id = (hfItr)->detid();
	  theFillDetMap0[id] = theFillDetMap0[id]+ 1.;
	  theFillDetMap1[id] = theFillDetMap1[id]+(*hfItr).energy();
	  theFillDetMap2[id] = theFillDetMap2[id]+pow((*hfItr).energy(),2);
	  theFillDetMap3[id] = theFillDetMap3[id]+pow((*hfItr).energy(),3);
	  theFillDetMap4[id] = theFillDetMap4[id]+pow((*hfItr).energy(),4);
	}
    }
  }

}
}
