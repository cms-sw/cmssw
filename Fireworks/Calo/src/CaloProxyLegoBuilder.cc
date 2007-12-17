#include "Fireworks/Calo/interface/CaloProxyLegoBuilder.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "THStack.h"
#include "TH2F.h"
#include "TList.h"
CaloProxyLegoBuilder::CaloProxyLegoBuilder()
{
}

CaloProxyLegoBuilder::~CaloProxyLegoBuilder()
{
}

void CaloProxyLegoBuilder::build(const fwlite::Event* iEvent, TObject** product)
{
   THStack* stack = dynamic_cast<THStack*>(*product);
   if ( !stack  && *product ) {
      std::cout << "incorrect type" << std::endl;
      return;
   }
   TH2F *ecal(0), *hcal(0), *jetsLego(0);
   
   if( ! stack ) {
      stack =  new THStack("LegoStack","Calo tower lego plot");
      stack->SetMaximum(100); // 100 GeV
      *product = stack;
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

      
      TH2F* bkg = new TH2F("bkgLego","Background distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
      bkg->SetFillColor(Color_t(kWhite));
      stack->Add(bkg);

      
      jetsLego = new TH2F("jetsLego","Jets distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
      jetsLego->SetFillColor(Color_t(kYellow));
      stack->Add(jetsLego);

      ecal = new TH2F("ecalLego","CaloTower ECAL Et distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
      ecal->SetFillColor(Color_t(kBlue));
      stack->Add(ecal);
      
      hcal = new TH2F("hcalLego","CaloTower HCAL Et distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
      hcal->SetFillColor(Color_t(kRed));
      stack->Add(hcal);
      
   } else {
      ecal = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("ecalLego") );
      hcal = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("hcalLego") );
      jetsLego = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("jetsLego") );
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

   
