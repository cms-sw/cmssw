#include "Fireworks/Calo/interface/CaloProxyLegoBuilder.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "THStack.h"
#include "TH2F.h"
#include "TList.h"
#include "TEveElement.h"
#include "TEveManager.h"
CaloProxyLegoBuilder::CaloProxyLegoBuilder()
{
}

CaloProxyLegoBuilder::~CaloProxyLegoBuilder()
{
}

void CaloProxyLegoBuilder::build(const fwlite::Event* iEvent, TObject** product)
{
   
   // define a parameter to control resolution in phi.
   // 1,2,4 are natural values, don't use other numbers
   // no eta rebining is supported yet
   m_legoRebinFactor = 1;
   
   TList* list = dynamic_cast<TList*>(*product);
   if ( !list  && *product ) {
      std::cout << "incorrect type of the main lego list" << std::endl;
      return;
   }
   
   TH2F *ecal(0), *hcal(0), *jetsLego(0);
   THStack* stack(0);
   
   if ( ! list ) {
      // THStack* stack = dynamic_cast<THStack*>(*product);
      // if ( !stack  && *product ) {
      // std::cout << "incorrect type" << std::endl;
      //	 return;
      //}
   
      list = new TList();
      *product = list;

      // if( ! stack ) {
      stack =  new THStack("LegoStack","Calo tower lego plot");
      list->Add( stack );
      stack->SetMaximum(100); // 100 GeV
      // lets use index numbers for now.
      // http://ecal-od-software.web.cern.ch/ecal-od-software/documents/documents/cal_newedm_roadmap_v1_0.pdf
      // Eta mapping:
      //   ieta - [-41,-1]+[1,41] - total 82 bins 
      //   calo tower gives eta of the ceneter of each bin
      //   size:
      //      0.087 - [-20,-1]+[1,20]
      //      the rest have variable size from 0.09-0.30
      // Phi mapping:
      //   iphi - [1-72]
      //   calo tower gives phi of the center of each bin
      //   for |ieta|<=20 phi bins are all of the same size
      //      ieta 36-37 transition corresponds to 3.1 -> -3.1 transition
      //   for 20 < |ieta| < 40
      //      there are only 36 active bins corresponding to odd numbers
      //      ieta 35->37, corresponds to 3.05 -> -3.05 transition
      //   for |ieta| >= 40
      //      there are only 18 active bins 3,7,11,15 etc
      //      ieta 31 -> 35, corresponds to 2.79253 -> -3.14159 transition
      
      double xbins[79] = {
	   -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, 
	   -3.139, -2.964, -2.853, -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830, 
	   -1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, 
	   -0.870, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087,  
	   0.000,
	    0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.870,
	    0.957,  1.044,  1.131,  1.218,  1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  
	    1.830,  1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  2.853,  2.964,  3.139,  
	    3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716};

      
      TH2F* bkg = new TH2F("bkgLego","Background distribution",
			   // 120,-5.220,5.220,36,-3.1416,3.1416)
			   // 40,-20,20,72,0,72);
			   // 40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
			   78, xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      bkg->SetFillColor(Color_t(kWhite));
      stack->Add(bkg);
      
      jetsLego = new TH2F("jetsLego","Jets distribution",
			  // 120,-5.220,5.220,36,-3.1416,3.1416)
			  // 40,-20,20,72,0,72);
			  // 40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
			  78, xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      jetsLego->SetFillColor(Color_t(kYellow));
      stack->Add(jetsLego);

      ecal = new TH2F("ecalLego","CaloTower ECAL Et distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      // 40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
		      78, xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      ecal->SetFillColor(Color_t(kBlue));
      stack->Add(ecal);
      
      hcal = new TH2F("hcalLego","CaloTower HCAL Et distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      // 40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
		      78, xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      hcal->SetFillColor(Color_t(kRed));
      stack->Add(hcal);
      
      TH2F* overlay = new TH2F("overLego","Highlight distribution",
			       // 120,-5.220,5.220,36,-3.1416,3.1416)
			       // 40,-20,20,72,0,72);
			       // 40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
			       78, xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      overlay->SetFillColor(Color_t(kRed));
      stack->Add(overlay);
      
      TEveElementList* stackElements = new TEveElementList("Lego","Lego plot",true);
      TEveElementList* ecalLegoEve = new TEveElementList("ecalLegoEve","",true);
      stackElements->AddElement( ecalLegoEve );
      list->Add(ecalLegoEve);
      ecalLegoEve->SetMainColor(Color_t(kBlue));
      TEveElementList* hcalLegoEve = new TEveElementList("hcalLegoEve","",true);
      stackElements->AddElement( hcalLegoEve );
      hcalLegoEve->SetMainColor(Color_t(kRed));
      list->Add(hcalLegoEve);
//      stackElements->AddElement( new TEveElementObjectPtr(bkg,false) );
//      stackElements->AddElement( new TEveElementObjectPtr(jetsLego,false) );
//      stackElements->AddElement( new TEveElementObjectPtr(ecal,false) );
//      stackElements->AddElement( new TEveElementObjectPtr(hcal,false) );
//      stackElements->AddElement( new TEveElementObjectPtr(overlay,false) );
      list->Add( stackElements );
      gEve->AddElement( stackElements );
   } else {
      stack = dynamic_cast<THStack*>(list->FindObject("LegoStack"));
      if ( !stack ) {
	 std::cout << "incorrect type of the lego stack objectÑ" << std::endl;
	 return;
      }
      ecal = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("ecalLego") );
      hcal = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("hcalLego") );
      jetsLego = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("jetsLego") );
      
      if ( ecal && hcal ) {
	 if ( TEveElementList* ecalLegoEve = (TEveElementList*)list->FindObject("ecalLegoEve") ) {
	    ecal->SetFillColor( ecalLegoEve->GetMainColor() );
	    std::cout << "ecal color: " << ecalLegoEve->GetMainColor() << std::endl;
	 } else {
	    std::cout << "cannot get ecalLegoEve" << std::endl;
	 }
	    
	 if ( TEveElementList* hcalLegoEve = (TEveElementList*)list->FindObject("hcalLegoEve") ) {
	    hcal->SetFillColor( hcalLegoEve->GetMainColor() );
	    std::cout << "hcal color: " << hcalLegoEve->GetMainColor() << std::endl;
	 } else {
	    std::cout << "cannot get hcalLegoEve" << std::endl;
	 }
      } else {
	 std::cout << "ecal or hcal is not set" << std::endl;
      }
   }
   
   fwlite::Handle<CaloTowerCollection> towers;
   towers.getByLabel(*iEvent,"towerMaker");
   
   if(0 == towers.ptr() ) {
      std::cout <<"failed to get calo towers"<<std::endl;
      return;
   }
   ecal->Reset();
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
          ecal->Fill(tower->eta(), tower->phi(), tower->emEt());
   hcal->Reset();
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
          hcal->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());

   // remove phi gaps at high eta
   if ( m_legoRebinFactor == 1 && 1==2 ) {
      for ( int ix = 1; ix <= ecal->GetNbinsX(); ++ix )
	{
	   // this is slow, but simple and robust
	   if ( fabs(ecal->GetXaxis()->GetBinCenter(ix)) < 1.74 ) continue;
	   for ( int iy = 1; iy < jetsLego->GetNbinsY(); iy+=2 ) {
	      ecal->SetBinContent(ix,iy+1,ecal->GetBinContent(ix,iy));
	      hcal->SetBinContent(ix,iy+1,hcal->GetBinContent(ix,iy));
	   }
	}
   }
   
   fwlite::Handle<reco::CaloJetCollection> jets;
   jets.getByLabel(*iEvent,"iterativeCone5CaloJets");
   
   if(0 == jets.ptr() ) {
      std::cout <<"failed to get jets"<<std::endl;
      return;
   }
   jetsLego->Reset();
   // loop over all bins and mark those close to jets
   double minJetEt = 15; // GeV
   double coneSize = 0.5; // jet cone size
   for ( int ix = 1; ix <= jetsLego->GetNbinsX(); ++ix )
     for ( int iy = 1; iy <= jetsLego->GetNbinsY(); ++iy )
       for(reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet)
	 if ( jet->et() > minJetEt &&
	      deltaR( jet->eta(), jet->phi(), 
		      jetsLego->GetXaxis()->GetBinCenter(ix),
		      jetsLego->GetYaxis()->GetBinCenter(iy) ) < 
	      coneSize + sqrt( pow(jetsLego->GetXaxis()->GetBinWidth(ix),2) +
			       pow(jetsLego->GetYaxis()->GetBinWidth(iy),2) ) )
	   jetsLego->SetBinContent(ix, iy, 0.1);
}

double CaloProxyLegoBuilder::deltaR( double eta1, double phi1, double eta2, double phi2 )
{
   double dEta = eta2-eta1;
   double dPhi = fabs(phi2-phi1);
   if ( dPhi > 3.1416 ) dPhi = 2*3.1416 - dPhi;
   return sqrt(dPhi*dPhi+dEta*dEta);
}

   
