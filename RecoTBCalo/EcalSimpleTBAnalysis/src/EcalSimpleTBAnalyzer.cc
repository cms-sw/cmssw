/**\class EcalSimpleTBAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EcalSimpleTBAnalyzer.cc,v 1.10 2012/02/01 19:41:58 vskarupe Exp $
//
//

#include "RecoTBCalo/EcalSimpleTBAnalysis/interface/EcalSimpleTBAnalyzer.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"


//#include<fstream>

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"

#include <iostream>
#include <string>
#include <stdexcept>
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//


//========================================================================
EcalSimpleTBAnalyzer::EcalSimpleTBAnalyzer( const edm::ParameterSet& iConfig ) : xtalInBeam_(0)
//========================================================================
{
   //now do what ever initialization is needed
   rootfile_          = iConfig.getUntrackedParameter<std::string>("rootfile","ecalSimpleTBanalysis.root");
   digiCollection_     = iConfig.getParameter<std::string>("digiCollection");
   digiProducer_       = iConfig.getParameter<std::string>("digiProducer");
   hitCollection_     = iConfig.getParameter<std::string>("hitCollection");
   hitProducer_       = iConfig.getParameter<std::string>("hitProducer");
   hodoRecInfoCollection_     = iConfig.getParameter<std::string>("hodoRecInfoCollection");
   hodoRecInfoProducer_       = iConfig.getParameter<std::string>("hodoRecInfoProducer");
   tdcRecInfoCollection_     = iConfig.getParameter<std::string>("tdcRecInfoCollection");
   tdcRecInfoProducer_       = iConfig.getParameter<std::string>("tdcRecInfoProducer");
   eventHeaderCollection_     = iConfig.getParameter<std::string>("eventHeaderCollection");
   eventHeaderProducer_       = iConfig.getParameter<std::string>("eventHeaderProducer");


   std::cout << "EcalSimpleTBAnalyzer: fetching hitCollection: " << hitCollection_.c_str()
	<< " produced by " << hitProducer_.c_str() << std::endl;

}


//========================================================================
EcalSimpleTBAnalyzer::~EcalSimpleTBAnalyzer()
//========================================================================
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  // Amplitude vs TDC offset
//   if (h_ampltdc)
//   delete h_ampltdc;
  
//   // Reconstructed energies
//   delete h_e1x1;
//   delete h_e3x3; 
//   delete h_e5x5; 
  
//   delete h_bprofx; 
//   delete h_bprofy; 
  
//   delete h_qualx; 
//   delete h_qualy; 
  
//   delete h_slopex; 
//   delete h_slopey; 
  
//   delete h_mapx; 
//   delete h_mapy; 

}

//========================================================================
void
EcalSimpleTBAnalyzer::beginJob() {
//========================================================================

  // Amplitude vs TDC offset
  h_ampltdc = new TH2F("h_ampltdc","Max Amplitude vs TDC offset", 100,0.,1.,1000, 0., 4000.);

  // Reconstructed energies
  h_tableIsMoving = new TH1F("h_tableIsMoving","TableIsMoving", 100000, 0., 100000.);

  h_e1x1 = new TH1F("h_e1x1","E1x1 energy", 1000, 0., 4000.);
  h_e3x3 = new TH1F("h_e3x3","E3x3 energy", 1000, 0., 4000.);
  h_e5x5 = new TH1F("h_e5x5","E5x5 energy", 1000, 0., 4000.);

  h_e1x1_center = new TH1F("h_e1x1_center","E1x1 energy", 1000, 0., 4000.);
  h_e3x3_center = new TH1F("h_e3x3_center","E3x3 energy", 1000, 0., 4000.);
  h_e5x5_center = new TH1F("h_e5x5_center","E5x5 energy", 1000, 0., 4000.);

  h_e1e9 = new TH1F("h_e1e9","E1/E9 ratio", 600, 0., 1.2);
  h_e1e25 = new TH1F("h_e1e25","E1/E25 ratio", 600, 0., 1.2);
  h_e9e25 = new TH1F("h_e9e25","E9/E25 ratio", 600, 0., 1.2);

  h_bprofx = new TH1F("h_bprofx","Beam Profile X",100,-20.,20.);
  h_bprofy = new TH1F("h_bprofy","Beam Profile Y",100,-20.,20.);

  h_qualx = new TH1F("h_qualx","Beam Quality X",5000,0.,5.);
  h_qualy = new TH1F("h_qualy","Beam Quality X",5000,0.,5.);

  h_slopex = new TH1F("h_slopex","Beam Slope X",500, -5e-4 , 5e-4 );
  h_slopey = new TH1F("h_slopey","Beam Slope Y",500, -5e-4 , 5e-4 );

  char hname[50];
  char htitle[50];
  for (unsigned int icry=0;icry<25;icry++)
    {       
      sprintf(hname,"h_mapx_%d",icry);
      sprintf(htitle,"Max Amplitude vs X %d",icry);
      h_mapx[icry] = new TH2F(hname,htitle,80,-20,20,1000,0.,4000.);
      sprintf(hname,"h_mapy_%d",icry);
      sprintf(htitle,"Max Amplitude vs Y %d",icry);
      h_mapy[icry] = new TH2F(hname,htitle,80,-20,20,1000,0.,4000.);
    }
  
  h_e1e9_mapx = new TH2F("h_e1e9_mapx","E1/E9 vs X",80,-20,20,600,0.,1.2);
  h_e1e9_mapy = new TH2F("h_e1e9_mapy","E1/E9 vs Y",80,-20,20,600,0.,1.2);

  h_e1e25_mapx = new TH2F("h_e1e25_mapx","E1/E25 vs X",80,-20,20,600,0.,1.2);
  h_e1e25_mapy = new TH2F("h_e1e25_mapy","E1/E25 vs Y",80,-20,20,600,0.,1.2);

  h_e9e25_mapx = new TH2F("h_e9e25_mapx","E9/E25 vs X",80,-20,20,600,0.,1.2);
  h_e9e25_mapy = new TH2F("h_e9e25_mapy","E9/E25 vs Y",80,-20,20,600,0.,1.2);

  h_Shape_ = new TH2F("h_Shape_","Xtal in Beam Shape",250,0,10,350,0,3500);

}

//========================================================================
void
EcalSimpleTBAnalyzer::endJob() {
//========================================================================

  TFile f(rootfile_.c_str(),"RECREATE");

  // Amplitude vs TDC offset
  h_ampltdc->Write(); 

  // Reconstructed energies
  h_e1x1->Write(); 
  h_e3x3->Write(); 
  h_e5x5->Write(); 

  h_e1x1_center->Write(); 
  h_e3x3_center->Write(); 
  h_e5x5_center->Write(); 

  h_e1e9->Write(); 
  h_e1e25->Write(); 
  h_e9e25->Write(); 

  h_bprofx->Write(); 
  h_bprofy->Write(); 

  h_qualx->Write(); 
  h_qualy->Write(); 

  h_slopex->Write(); 
  h_slopey->Write(); 
  
  h_Shape_->Write();

  for (unsigned int icry=0;icry<25;icry++)
    {       
      h_mapx[icry]->Write(); 
      h_mapy[icry]->Write(); 
    }

  h_e1e9_mapx->Write(); 
  h_e1e9_mapy->Write(); 

  h_e1e25_mapx->Write(); 
  h_e1e25_mapy->Write(); 

  h_e9e25_mapx->Write(); 
  h_e9e25_mapy->Write(); 

  h_tableIsMoving->Write();

  f.Close();
}

//
// member functions
//

//========================================================================
void
EcalSimpleTBAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
//========================================================================

   using namespace edm;
   using namespace cms;



   Handle<EBDigiCollection> pdigis;
   const EBDigiCollection* digis=0;
   //std::cout << "EcalSimpleTBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( digiProducer_, digiCollection_,pdigis);
   if ( pdigis.isValid() ) {
     digis = pdigis.product(); // get a ptr to the product
     //iEvent.getByLabel( hitProducer_, phits);
   } else {
           edm::LogError("EcalSimpleTBAnalyzerError") << "Error! can't get the product " << digiCollection_;
   }

   // fetch the digis and compute signal amplitude
   Handle<EBUncalibratedRecHitCollection> phits;
   const EBUncalibratedRecHitCollection* hits=0;
   //std::cout << "EcalSimpleTBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( hitProducer_, hitCollection_,phits);
   if (phits.isValid()) {
     hits = phits.product(); // get a ptr to the product
     //iEvent.getByLabel( hitProducer_, phits);
   } else {
           edm::LogError("EcalSimpleTBAnalyzerError") << "Error! can't get the product " << hitCollection_;
   }

   Handle<EcalTBHodoscopeRecInfo> pHodo;
   const EcalTBHodoscopeRecInfo* recHodo=0;
   //std::cout << "EcalSimpleTBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( hodoRecInfoProducer_, hodoRecInfoCollection_, pHodo);
   if ( pHodo.isValid() ) {
     recHodo = pHodo.product(); // get a ptr to the product
   } else {
           edm::LogError("EcalSimpleTBAnalyzerError") << "Error! can't get the product " << hodoRecInfoCollection_;
   }

   Handle<EcalTBTDCRecInfo> pTDC;
   const EcalTBTDCRecInfo* recTDC=0;
   //std::cout << "EcalSimpleTBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( tdcRecInfoProducer_, tdcRecInfoCollection_, pTDC);
   if ( pTDC.isValid() ) {
     recTDC = pTDC.product(); // get a ptr to the product
   } else {
           edm::LogError("EcalSimpleTBAnalyzerError") << "Error! can't get the product " << tdcRecInfoCollection_;
   }

   Handle<EcalTBEventHeader> pEventHeader;
   const EcalTBEventHeader* evtHeader=0;
   //std::cout << "EcalSimpleTBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( eventHeaderProducer_ , pEventHeader );
   if ( pEventHeader.isValid() ) {
     evtHeader = pEventHeader.product(); // get a ptr to the product
   } else {
           edm::LogError("EcalSimpleTBAnalyzerError") << "Error! can't get the product " << eventHeaderProducer_;
   }
   
   if (!hits)
     return;

   if (!recTDC)
     return;

   if (!recHodo)
     return;

   if (!evtHeader)
     return;

   if (hits->size() == 0)
     return;

   if (evtHeader->tableIsMoving())
     h_tableIsMoving->Fill(evtHeader->eventNumber());

   // Crystal hit by beam
   if (xtalInBeam_.null())
     {
       xtalInBeam_ = EBDetId(1,evtHeader->crystalInBeam(),EBDetId::SMCRYSTALMODE);
       std::cout<< "Xtal In Beam is " << xtalInBeam_.ic() << std::endl;
     }
   else if (xtalInBeam_ != EBDetId(1,evtHeader->crystalInBeam(),EBDetId::SMCRYSTALMODE))
     return;

   if (evtHeader->tableIsMoving())
     return;
   
   
//    EBDetId maxHitId(0); 
//    float maxHit= -999999.;
   
//    for(EBUncalibratedRecHitCollection::const_iterator ithit = hits->begin(); ithit != hits->end(); ++ithit) 
//      {
//        if (ithit->amplitude()>=maxHit)
// 	 {
// 	   maxHit=ithit->amplitude();
// 	   maxHitId=ithit->id();
// 	 }
       
//      }   

//    if (maxHitId==EBDetId(0))
//      return;

//   EBDetId maxHitId(1,704,EBDetId::SMCRYSTALMODE); 

   //Find EBDetId in a 5x5 Matrix (to be substituted by the Selector code)
   // Something like 
   // EBFixedWindowSelector<EcalUncalibratedRecHit> Simple5x5Matrix(hits,maxHitId,5,5);
   // std::vector<EcalUncalibratedRecHit> Energies5x5 = Simple5x5Matrix.getHits();


   EBDetId Xtals5x5[25];
   for (unsigned int icry=0;icry<25;icry++)
     {
       unsigned int row = icry / 5;
       unsigned int column= icry %5;
       int ieta=xtalInBeam_.ieta()+column-2;
       int iphi=xtalInBeam_.iphi()+row-2;
       EBDetId tempId(ieta, iphi,EBDetId::ETAPHIMODE);
       if (tempId.ism()==1) 
               Xtals5x5[icry]=tempId;
       else
               Xtals5x5[icry]=EBDetId(0);
       ///       std::cout << "** Xtal in the matrix **** row " << row  << ", column " << column << ", xtal " << Xtals5x5[icry].ic() << std::endl;
     } 


   
   double samples_save[10]; for(int i=0; i < 10; ++i) samples_save[i]=0.0;
   
   // find the rechit corresponding digi and the max sample
   EBDigiCollection::const_iterator myDg = digis->find(xtalInBeam_);
   double eMax = 0.;
   if (myDg != digis->end())
     {
       EBDataFrame myDigi = (*myDg);
       for (int sample = 0; sample < myDigi.size(); ++sample)
	 {
	   double analogSample = myDigi.sample(sample).adc();
	   samples_save[sample] = analogSample;
	   //  std::cout << analogSample << " ";
	   if ( eMax < analogSample )
	     {
	       eMax = analogSample;
	     }
	 }
       // std::cout << std::endl;
     }

   for(int i =0; i < 10; ++i) {
     h_Shape_->Fill(double(i)+recTDC->offset(),samples_save[i]);
   }

   double amplitude[25];
   
   double amplitude3x3=0;  
   double amplitude5x5=0;  
   
   for (unsigned int icry=0;icry<25;icry++)
     {
       if (!Xtals5x5[icry].null())
	 {
	   amplitude[icry]=(hits->find(Xtals5x5[icry]))->amplitude();
	   amplitude5x5 += amplitude[icry];
	   // Is in 3x3?
	   if ( icry == 6  || icry == 7  || icry == 8 ||
		icry == 11 || icry == 12 || icry ==13 ||
		icry == 16 || icry == 17 || icry ==18   )
	     {
	       amplitude3x3+=amplitude[icry];
	     }
	 }
     }


   h_e1x1->Fill(amplitude[12]);
   h_e3x3->Fill(amplitude3x3);
   h_e5x5->Fill(amplitude5x5);

   h_e1e9->Fill(amplitude[12]/amplitude3x3);
   h_e1e25->Fill(amplitude[12]/amplitude5x5);
   h_e9e25->Fill(amplitude3x3/amplitude5x5);

   if (recTDC)
     h_ampltdc->Fill(recTDC->offset(),amplitude[12]);

   if (recHodo)
     {
       float x=recHodo->posX();
       float y=recHodo->posY();
       float xslope=recHodo->slopeX();
       float yslope=recHodo->slopeY();
       float xqual=recHodo->qualX();
       float yqual=recHodo->qualY();
       
       //Filling beam profiles
       h_bprofx->Fill(x);
       h_bprofy->Fill(y);
       h_qualx->Fill(xqual);
       h_qualy->Fill(yqual);
       h_slopex->Fill(xslope);
       h_slopey->Fill(yslope);

       //Fill central events

       
       if ( fabs(x + 2.5) < 2.5 && fabs(y + 0.5) < 2.5)
	 {
	   h_e1x1_center->Fill(amplitude[12]);
	   h_e3x3_center->Fill(amplitude3x3);
	   h_e5x5_center->Fill(amplitude5x5);
	   
	   h_e1e9->Fill(amplitude[12]/amplitude3x3);
	   h_e1e25->Fill(amplitude[12]/amplitude5x5);
	   h_e9e25->Fill(amplitude3x3/amplitude5x5);
	 }

       for (unsigned int icry=0;icry<25;icry++)
	 {       
	   h_mapx[icry]->Fill(x,amplitude[icry]);
	   h_mapy[icry]->Fill(y,amplitude[icry]);
	 }

       h_e1e9_mapx->Fill(x,amplitude[12]/amplitude3x3);
       h_e1e9_mapy->Fill(y,amplitude[12]/amplitude3x3);

       h_e1e25_mapx->Fill(x,amplitude[12]/amplitude5x5);
       h_e1e25_mapy->Fill(y,amplitude[12]/amplitude5x5);

       h_e9e25_mapx->Fill(x,amplitude3x3/amplitude5x5);
       h_e9e25_mapy->Fill(y,amplitude3x3/amplitude5x5);
     }

}


