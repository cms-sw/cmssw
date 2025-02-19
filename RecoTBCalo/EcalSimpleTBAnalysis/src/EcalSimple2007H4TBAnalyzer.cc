/**\class EcalSimple2007H4TBAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EcalSimple2007H4TBAnalyzer.cc,v 1.4 2012/02/01 19:41:58 vskarupe Exp $
//
//

#include "RecoTBCalo/EcalSimpleTBAnalysis/interface/EcalSimple2007H4TBAnalyzer.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

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
EcalSimple2007H4TBAnalyzer::EcalSimple2007H4TBAnalyzer( const edm::ParameterSet& iConfig ) : xtalInBeam_(0)
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


   std::cout << "EcalSimple2007H4TBAnalyzer: fetching hitCollection: " << hitCollection_.c_str()
	<< " produced by " << hitProducer_.c_str() << std::endl;

}


//========================================================================
EcalSimple2007H4TBAnalyzer::~EcalSimple2007H4TBAnalyzer()
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
EcalSimple2007H4TBAnalyzer::beginRun(edm::Run const &, edm::EventSetup const& iSetup) {
//========================================================================

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);   

  
  theTBGeometry_ =  &(*pG);
//  const std::vector<DetId>& validIds=theTBGeometry_->getValidDetIds(DetId::Ecal,EcalEndcap);
//   std::cout << "Found " << validIds.size() << " channels in the geometry" << std::endl;
//   for (unsigned int i=0;i<validIds.size();++i)
//     std::cout << EEDetId(validIds[i]) << std::endl;

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

  h_S6 = new TH1F("h_S6","Amplitude S6", 1000, 0., 4000.);

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
EcalSimple2007H4TBAnalyzer::endJob() {
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

  h_S6->Write(); 
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
EcalSimple2007H4TBAnalyzer::analyze( edm::Event const & iEvent, edm::EventSetup const & iSetup ) {
//========================================================================

   using namespace edm;
   using namespace cms;



   Handle<EEDigiCollection> pdigis;
   const EEDigiCollection* digis=0;
   //std::cout << "EcalSimple2007H4TBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( digiProducer_, digiCollection_,pdigis);
   if ( pdigis.isValid() ) {
     digis = pdigis.product(); // get a ptr to the product
     //iEvent.getByLabel( hitProducer_, phits);
   } else {
           edm::LogError("EcalSimple2007H4TBAnalyzerError") << "Error! can't get the product " << digiCollection_;
   }

   // fetch the digis and compute signal amplitude
   Handle<EEUncalibratedRecHitCollection> phits;
   const EEUncalibratedRecHitCollection* hits=0;
   //std::cout << "EcalSimple2007H4TBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( hitProducer_, hitCollection_,phits);
   if (phits.isValid()) {
     hits = phits.product(); // get a ptr to the product
     //iEvent.getByLabel( hitProducer_, phits);
   } else {
           edm::LogError("EcalSimple2007H4TBAnalyzerError") << "Error! can't get the product " << hitCollection_;
   }

   Handle<EcalTBHodoscopeRecInfo> pHodo;
   const EcalTBHodoscopeRecInfo* recHodo=0;
   //std::cout << "EcalSimple2007H4TBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( hodoRecInfoProducer_, hodoRecInfoCollection_, pHodo);
   if ( pHodo.isValid() ) {
     recHodo = pHodo.product(); // get a ptr to the product
   } else {
           edm::LogError("EcalSimple2007H4TBAnalyzerError") << "Error! can't get the product " << hodoRecInfoCollection_;
   }

   Handle<EcalTBTDCRecInfo> pTDC;
   const EcalTBTDCRecInfo* recTDC=0;
   //std::cout << "EcalSimple2007H4TBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( tdcRecInfoProducer_, tdcRecInfoCollection_, pTDC);
   if ( pTDC.isValid() ) {
     recTDC = pTDC.product(); // get a ptr to the product
   } else {
           edm::LogError("EcalSimple2007H4TBAnalyzerError") << "Error! can't get the product " << tdcRecInfoCollection_;
   }

   Handle<EcalTBEventHeader> pEventHeader;
   const EcalTBEventHeader* evtHeader=0;
   //std::cout << "EcalSimple2007H4TBAnalyzer::analyze getting product with label: " << digiProducer_.c_str()<< " prodname: " << digiCollection_.c_str() << endl;
   iEvent.getByLabel( eventHeaderProducer_ , pEventHeader );
   if ( pEventHeader.isValid() ) {
     evtHeader = pEventHeader.product(); // get a ptr to the product
   } else {
           edm::LogError("EcalSimple2007H4TBAnalyzerError") << "Error! can't get the product " << eventHeaderProducer_;
   }
   
   
   if (!hits)
     {
       //       std::cout << "1" << std::endl;
       return;
     }

   if (!recTDC)
     {
       //       std::cout << "2" << std::endl;
       return;
     }

   if (!recHodo)
     {
       //       std::cout << "3" << std::endl;
       return;
     }

   if (!evtHeader)
     {
       //       std::cout << "4" << std::endl;
       return;
     }

   if (hits->size() == 0)
     {
       //       std::cout << "5" << std::endl;
       return;
     }

   //Accessing various event information
   if (evtHeader->tableIsMoving())
     h_tableIsMoving->Fill(evtHeader->eventNumber());

//    std::cout << "event " << evtHeader->eventNumber() 
// 	     << "\trun number " << evtHeader->runNumber()   
// 	     << "\tburst number " << evtHeader->burstNumber()   
// 	     << "\tbeginLV1A " << evtHeader->begBurstLV1A()
// 	     << "\tendLV1A " << evtHeader->endBurstLV1A()
// 	     << "\ttime " << evtHeader->date()
// 	     << "\thas errors " << int(evtHeader->syncError())
// 	     << std::endl;

//    std::cout << "scaler";
//    for (int iscaler=0;iscaler < 36;iscaler++)
//      std::cout << "\t#" << iscaler << " " <<  evtHeader->scaler(iscaler);
//    std::cout<<std::endl;

   //S6 beam scintillator
   h_S6->Fill(evtHeader->S6ADC());

   if (xtalInBeamTmp.null())
     {
       xtalInBeamTmp = EBDetId(1,evtHeader->crystalInBeam(),EBDetId::SMCRYSTALMODE);
       xtalInBeam_ = EEDetId( 35 - ((xtalInBeamTmp.ic()-1)%20) ,int(int(xtalInBeamTmp.ic())/int(20))+51, -1);
       std::cout<< "Xtal In Beam is " << xtalInBeam_.ic() << xtalInBeam_ << std::endl;
       for (unsigned int icry=0;icry<25;icry++)
	 {
	   unsigned int row = icry / 5;
	   unsigned int column= icry %5;
	   int ix=xtalInBeam_.ix()+row-2;
	   int iy=xtalInBeam_.iy()+column-2;
	   EEDetId tempId(ix, iy, xtalInBeam_.zside());
	   //Selecting matrix of xtals used in 2007H4TB
	   if (tempId.ix()<16 || tempId.ix()>35 || tempId.iy()<51 || tempId.iy()>75)
	     Xtals5x5[icry]=EEDetId(0);
	   else
	     {
	       Xtals5x5[icry]=tempId;
	       const CaloCellGeometry* cell=theTBGeometry_->getGeometry(Xtals5x5[icry]);
	       if (!cell) 
		 continue;
	       const TruncatedPyramid* tp ( dynamic_cast<const TruncatedPyramid*>(cell) ) ;
	       std::cout << "** Xtal in the matrix **** row " << row  << ", column " << column << ", xtal " << Xtals5x5[icry] << " Position " << tp->getPosition(0.) << std::endl;
	     }
	 }
     }
   else 
     if (xtalInBeamTmp != EBDetId(1,evtHeader->crystalInBeam(),EBDetId::SMCRYSTALMODE)) //run analysis only on first xtal in beam
       return;

   //Avoid moving table events
   if (evtHeader->tableIsMoving())
     {
       std::cout << "Table is moving" << std::endl;
       return;
     }


   
   // Searching for max amplitude xtal alternative to use xtalInBeam_
   EEDetId maxHitId(0); 
   float maxHit= -999999.;
   for(EEUncalibratedRecHitCollection::const_iterator ithit = hits->begin(); ithit != hits->end(); ++ithit) 
     {
       if (ithit->amplitude()>=maxHit)
	 {
	   maxHit=ithit->amplitude();
	   maxHitId=ithit->id();
	 }
       
     }   
   if (maxHitId==EEDetId(0))
     {
       std::cout << "No maxHit found" << std::endl;
       return;
     }

    
   //Filling the digis shape for the xtalInBeam
   double samples_save[10]; for(int i=0; i < 10; ++i) samples_save[i]=0.0;
   
   double eMax = 0.;
   for ( EEDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	 ++digiItr ) 
     {		
       if ( EEDetId((*digiItr).id()) != xtalInBeam_ )
	 continue;
       
       EEDataFrame myDigi = (*digiItr);
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

   for(int i =0; i < 10; ++i) 
     h_Shape_->Fill(double(i)+recTDC->offset(),samples_save[i]);



   // Taking amplitudes in 5x5
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

   //Filling amplitudes
   h_e1x1->Fill(amplitude[12]);
   h_e3x3->Fill(amplitude3x3);
   h_e5x5->Fill(amplitude5x5);

   h_e1e9->Fill(amplitude[12]/amplitude3x3);
   h_e1e25->Fill(amplitude[12]/amplitude5x5);
   h_e9e25->Fill(amplitude3x3/amplitude5x5);

   //Checking stability of amplitude vs TDC
   if (recTDC)
     h_ampltdc->Fill(recTDC->offset(),amplitude[12]);

   //Various amplitudes as a function of hodoscope coordinates
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


