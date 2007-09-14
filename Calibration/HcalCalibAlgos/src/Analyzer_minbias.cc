// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
//#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
//#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLHcal.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Calibration/HcalCalibAlgos/interface/Analyzer_minbias.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include <fstream>
#include <sstream>

using namespace std;
using namespace reco;
//
// constructors and destructor
//
namespace cms{
Analyzer_minbias::Analyzer_minbias(const edm::ParameterSet& iConfig)
{
  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile"); 
  // get names of modules, producing object collections
  //datasetType = iConfig.getParameter<string>("setType");   
//  mapHcal_.prefillMap();
//  hcalfile_=iConfig.getUntrackedParameter<std::string> ("fileNameHcal","");
//  MiscalibReaderFromXMLHcal hcalreader_(mapHcal_);
//  if(!hcalfile_.empty()) hcalreader_.parseXMLMiscalibFile(hcalfile_);
//  mapHcal_.print();
  
}

Analyzer_minbias::~Analyzer_minbias()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void Analyzer_minbias::beginJob( const edm::EventSetup& iSetup)
{
   double phibound = 4.*atan(1.);
   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   mystart = 0;
   
   myTree = new TTree("RecJet","RecJet Tree");
   myTree->Branch("mydet",  &mydet, "mydet/I");
   myTree->Branch("mysubd",  &mysubd, "mysubd/I");
   myTree->Branch("depth",  &depth, "depth/I");
   myTree->Branch("ieta",  &ieta, "ieta/I");
   myTree->Branch("iphi",  &iphi, "iphi/I");
   myTree->Branch("eta",  &eta, "eta/F");
   myTree->Branch("phi",  &phi, "phi/F");
   
   myTree->Branch("mom0",  &mom0, "mom0/F");
   myTree->Branch("mom1",  &mom1, "mom1/F");
   myTree->Branch("mom2",  &mom2, "mom2/F");
   myTree->Branch("mom3",  &mom3, "mom3/F");
   myTree->Branch("mom4",  &mom4, "mom4/F");

   myTree->Branch("mom0_cut",  &mom0_cut, "mom0_cut/F");
   myTree->Branch("mom1_cut",  &mom1_cut, "mom1_cut/F");
   myTree->Branch("mom2_cut",  &mom2_cut, "mom2_cut/F");
   myTree->Branch("mom3_cut",  &mom3_cut, "mom3_cut/F");
   myTree->Branch("mom4_cut",  &mom4_cut, "mom4_cut/F");
   myTree->Branch("occup",  &occup, "occup/F");
   
   hHBHEEt    = new TH1D( "hHBHEEt", "HBHEEt", 100,  -1., 10. );
   hHBHEEt_eta_1    = new TH1D( "hHBHEEt_eta_1", "HBHEEt_eta_1", 100,  -1., 10. );
   hHBHEEt_eta_25    = new TH1D( "hHBHEEt_eta_25", "HBHEEt_eta_25", 100,  -1., 10. );
   
   hHBHEEta    = new TH1D( "hHBHEEta", "HBHEEta", 100,  -3., 3. );
   hHBHEPhi    = new TH1D( "hHBHEPhi", "HBHEPhi", 100,  -1.*phibound, phibound );

   hHFEt    = new TH1D( "hHFEt", "HFEt", 100,  -1., 10. );
   hHFEt_eta_33    = new TH1D( "hHFEt_eta_33", "HFEt_eta_33", 100,  -1., 10. );
   
   hHFEta    = new TH1D( "hHFEta", "HFEta", 100,  -3., 3. );
   hHFPhi    = new TH1D( "hHFPhi", "HFPhi", 100,  -1.*phibound, phibound );
  
   hHOEt    = new TH1D( "hHOEt", "HOEt", 100,  -1., 10. );
   hHOEt_eta_5    = new TH1D( "hHOEt_eta_5", "HOEt_eta_5", 100,  -1., 10. );

   hHOEta    = new TH1D( "hHOEta", "HOEta", 100,  -3., 3. );
   hHOPhi    = new TH1D( "hHOPhi", "HOPhi", 100,  -1.*phibound, phibound );
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();
   
   std::vector<DetId> did =  geo->getValidDetIds();

   for(std::vector<DetId>::const_iterator id=did.begin(); id != did.end(); id++)
   {
      if( (*id).det() == DetId::Hcal ) {
      HcalDetId hid = HcalDetId(*id);
      theFillDetMap0[hid] = 0.;
      theFillDetMap1[hid] = 0.;
      theFillDetMap2[hid] = 0.;
      theFillDetMap3[hid] = 0.;
      theFillDetMap4[hid] = 0.;
   }
   }
  std::string ccc = "hcal.dat";

  myout_hcal = new ofstream(ccc.c_str());
  if(!myout_hcal) cout << " Output file not open!!! "<<endl;
    
   return ;
}

void Analyzer_minbias::endJob()
{
   int i=0;
   std::vector<DetId> alldid =  geo->getValidDetIds();
   for(std::vector<DetId>::const_iterator did=alldid.begin(); did != alldid.end(); did++)
   {      
      if( (*did).det() == DetId::Hcal ) 
      {
       HcalDetId hid = HcalDetId(*did);

       GlobalPoint pos = geo->getPosition(hid);
       mydet = ((hid).rawId()>>28)&0xF;
       mysubd = ((hid).rawId()>>25)&0x7;
       depth =(hid).depth();
       ieta = (hid).ieta();
       iphi = (hid).iphi();
       phi = pos.phi();
       eta = pos.eta();
       
       mom0 = theFillDetMap0[hid];
       mom1 = theFillDetMap1[hid];
       mom2 = theFillDetMap2[hid];
       mom3 = theFillDetMap3[hid];
       mom4 = theFillDetMap4[hid];
       
       
//       mom1 = theFillDetMap1[hid]/theFillDetMap0[hid];
//       mom2 = theFillDetMap2[hid]/theFillDetMap0[hid]-(mom1*mom1);
//       mom3 = theFillDetMap3[hid]/theFillDetMap0[hid]-3.*mom1*theFillDetMap2[hid]/theFillDetMap0[hid]+
//             2.*pow(mom2,3);
//       mom4 = (theFillDetMap4[hid]-4.*mom1*theFillDetMap3[hid]+6.*pow(mom1,2)*theFillDetMap2[hid])/theFillDetMap0[hid]-3.*pow(mom1,4);

       if(theFillDetMap_cut0[hid]>0.){
       
       mom0_cut = theFillDetMap_cut0[hid];
       mom1_cut = theFillDetMap_cut1[hid];
       mom2_cut = theFillDetMap_cut2[hid];
       mom3_cut = theFillDetMap_cut3[hid];
       mom4_cut = theFillDetMap_cut4[hid];
       
       
//       mom1_cut = theFillDetMap_cut1[hid]/theFillDetMap_cut0[hid];
//       mom2_cut = theFillDetMap_cut2[hid]/theFillDetMap_cut0[hid]-(mom1*mom1);
//       mom3_cut = theFillDetMap_cut3[hid]/theFillDetMap_cut0[hid]-3.*mom1*theFillDetMap_cut2[hid]/theFillDetMap_cut0[hid]+
//             2.*pow(mom2,3);
//       mom4_cut = (theFillDetMap_cut4[hid]-4.*mom1*theFillDetMap_cut3[hid]+6.*pow(mom1,2)*theFillDetMap_cut2[hid])/theFillDetMap_cut0[hid]-3.*pow(mom1,4);

       occup = theFillDetMap_cut0[hid]/theFillDetMap0[hid];
       
       } else
       {  
          mom0_cut = -10000.;
          mom1_cut = -10000.;
	  mom2_cut = -10000.;
	  mom3_cut = -10000.;
	  mom4_cut = -10000.;
       }

       cout<<" Result= "<<mydet<<" "<<mysubd<<" "<<ieta<<" "<<iphi<<" "<<mom1<<" "<<mom2<<endl;
       myTree->Fill();
       i++;
      } 
   }
   cout<<" Number of cells "<<i<<endl;    
   hOutputFile->Write() ;   
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   
   return ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
Analyzer_minbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  std::vector<Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
   std::vector<DetId> did =  geo->getValidDetIds();
 
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel("hbhereco", hbhe);
   edm::Handle<HORecHitCollection> ho;
   iEvent.getByLabel("horeco", ho);
   edm::Handle<HFRecHitCollection> hf;
   iEvent.getByLabel("hfreco", hf);

  
  const HBHERecHitCollection Hithbhe = *(hbhe.product());
  for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
        {
// Recalibration of energy
          float icalconst=1.;	 
//	 float icalconst=(mapHcal_.get().find(hbheItr->id().rawId()))->second;
	 HBHERecHit aHit(hbheItr->id(),hbheItr->energy()*icalconst,hbheItr->time());
	 
//         hHBHEEt->Fill((*hbheItr).energy());
         double energyhit = aHit.energy();
	 
	 hHBHEEta->Fill(geo->getPosition((*hbheItr).detid()).eta());
	 hHBHEPhi->Fill(geo->getPosition((*hbheItr).detid()).phi());
	 DetId id = (*hbheItr).detid(); 
	 HcalDetId hid=HcalDetId(id);
//	 if(mystart == 0) theHcalId.push_back(hid);
	
         if(hid.ieta() == 1 ) (*myout_hcal)<<iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<hid.iphi()<<energyhit<<endl; 
 
	 theFillDetMap0[hid] = theFillDetMap0[hid]+ 1.;
         theFillDetMap1[hid] = theFillDetMap1[hid]+energyhit;
	 theFillDetMap2[hid] = theFillDetMap2[hid]+pow(energyhit,2);
	 theFillDetMap3[hid] = theFillDetMap3[hid]+pow(energyhit,3);
	 theFillDetMap4[hid] = theFillDetMap4[hid]+pow(energyhit,4);
	 
	 if((*hbheItr).energy()>0.5)
	 { 
	 theFillDetMap_cut0[hid] = theFillDetMap_cut0[hid]+ 1.;
         theFillDetMap_cut1[hid] = theFillDetMap_cut1[hid]+energyhit;
	 theFillDetMap_cut2[hid] = theFillDetMap_cut2[hid]+pow(energyhit,2);
	 theFillDetMap_cut3[hid] = theFillDetMap_cut3[hid]+pow(energyhit,3);
	 theFillDetMap_cut4[hid] = theFillDetMap_cut4[hid]+pow(energyhit,4);
	 }
	 
	 if( hid.ieta() == 1 ) hHBHEEt_eta_1->Fill(energyhit);
	 if( hid.ieta() == 25 ) hHBHEEt_eta_25->Fill(energyhit);
	 
//	 cout<<" "<<geo->getPosition((*hbheItr).detid()).eta()<<" Eta= "<<hid.ieta()<<" "<<hid.depth()<<
//	 "energy "<<(*hbheItr).energy()<<endl;
	 
        }
  const HORecHitCollection Hitho = *(ho.product());
  for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
        {
         float icalconst=1.;
	// float icalconst=(mapHcal_.get().find(hoItr->id().rawId()))->second;
	 HORecHit aHit(hoItr->id(),hoItr->energy()*icalconst,hoItr->time());
	 double energyhit = aHit.energy();
	
	
         hHOEt->Fill(energyhit);
	 hHOEta->Fill(geo->getPosition((*hoItr).detid()).eta());
	 hHOPhi->Fill(geo->getPosition((*hoItr).detid()).phi());
	 HcalDetId hid=HcalDetId((*hoItr).detid());
//	 if(mystart == 0) theHcalId.push_back(hid);
	 
	 theFillDetMap0[hid] = theFillDetMap0[hid]+ 1.;
         theFillDetMap1[hid] = theFillDetMap1[hid]+energyhit;
	 theFillDetMap2[hid] = theFillDetMap2[hid]+pow(energyhit,2);
	 theFillDetMap3[hid] = theFillDetMap3[hid]+pow(energyhit,3);
	 theFillDetMap4[hid] = theFillDetMap4[hid]+pow(energyhit,4);
	 
	 
         if( hid.ieta() == 5 ) hHOEt_eta_5->Fill((*hoItr).energy());
//	 cout<<" "<<geo->getPosition((*hoItr).detid()).eta()<<" Eta= "<<hid.ieta()<<" "<<hid.depth()<<endl;
	  
        }

  const HFRecHitCollection Hithf = *(hf.product());
  for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
      {	
          float icalconst=1.; 
//         float icalconst=(mapHcal_.get().find(hfItr->id().rawId()))->second;
	 HFRecHit aHit(hfItr->id(),hfItr->energy()*icalconst,hfItr->time());
	 double energyhit = aHit.energy();

         hHFEt->Fill(energyhit);
	 hHFEta->Fill(geo->getPosition((*hfItr).detid()).eta());
	 hHFPhi->Fill(geo->getPosition((*hfItr).detid()).phi());
	 HcalDetId hid=HcalDetId((*hfItr).detid());
//	 if(mystart == 0) theHcalId.push_back(hid);
	 
	 theFillDetMap0[hid] = theFillDetMap0[hid]+ 1.;
         theFillDetMap1[hid] = theFillDetMap1[hid]+energyhit;
	 theFillDetMap2[hid] = theFillDetMap2[hid]+pow(energyhit,2);
	 theFillDetMap3[hid] = theFillDetMap3[hid]+pow(energyhit,3);
	 theFillDetMap4[hid] = theFillDetMap4[hid]+pow(energyhit,4);
	 
	 
         if( hid.ieta() == 33 ) hHFEt_eta_33->Fill(energyhit);	 
//	 cout<<" "<<geo->getPosition((*hfItr).detid()).eta()<<" Eta= "<<hid.ieta()<<" "<<hid.depth()<<
//         " "<<energyhit<<endl;
	 
      }
}
}
//define this as a plug-in
//DEFINE_ANOTHER_FWK_MODULE(Analyzer_minbias)

