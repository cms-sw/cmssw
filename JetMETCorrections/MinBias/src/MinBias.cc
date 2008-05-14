// user include files
#include "JetMETCorrections/MinBias/interface/MinBias.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

using namespace std;
namespace cms
{
MinBias::MinBias(const edm::ParameterSet& iConfig)
{
  // get names of modules, producing object collections
  hbheLabel_= iConfig.getParameter<string>("hbheInput");
  hoLabel_=iConfig.getParameter<string>("hoInput");
  hfLabel_=iConfig.getParameter<string>("hfInput");
  allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile"); 

}


MinBias::~MinBias()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void MinBias::beginJob( const edm::EventSetup& iSetup)
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

void MinBias::endJob()
{
   const 
   std::vector<DetId> did =  geo->getValidDetIds();
   int i=0;
   for(std::vector<DetId>::const_iterator id=did.begin(); id != did.end(); id++)
   {
      if( (*id).det() == DetId::Hcal ) {
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
      cout<<" Detector "<<(*id).rawId()<<" mydet "<<mydet<<" "<<mysubd<<" "<<depth<<" "<<
      HcalDetId(*id).subdet()<<" "<<ieta<<" "<<iphi<<" "<<pos.eta()<<" "<<pos.phi()<<endl;
      cout<<" Energy "<<mom1<<" "<<mom2<<endl;
      myTree->Fill();
      i++;
      }
   }
   cout<<" The number of CaloDet records "<<did.size()<<endl;
   cout<<" The number of Hcal records "<<i<<endl;


   cout << "===== Start writing user histograms =====" << endl;
   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   cout << "===== End writing user histograms =======" << endl; 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MinBias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
  
  if (!hbheLabel_.empty()) {
    edm::Handle<HBHERecHitCollection> hbhe;
    iEvent.getByLabel(hbheLabel_,hbhe);
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
	  if( (*hbheItr).energy() > 0. ) cout<<" Energy = "<<(*hbheItr).energy()<<endl;
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
    iEvent.getByLabel(hoLabel_,ho);
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
    iEvent.getByLabel(hfLabel_,hf);
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
