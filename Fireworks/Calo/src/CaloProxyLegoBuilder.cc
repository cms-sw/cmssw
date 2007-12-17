#include "Fireworks/Calo/interface/CaloProxyLegoBuilder.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
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
   TH2F *ecal(0), *hcal(0);
   
   if( ! stack ) {
      stack =  new THStack("LegoStack","Calo tower lego plot");
      stack->SetMaximum(1000); // 1 TeV
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

      
      ecal = new TH2F("ecalLego","CaloTower ECAL Et distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
      ecal->SetFillColor(Color_t(kBlue));
      hcal = new TH2F("hcalLego","CaloTower HCAL Et distribution",
		      // 120,-5.220,5.220,36,-3.1416,3.1416)
		      // 40,-20,20,72,0,72);
		      40, -1.74, 1.74, 72, -3.1416, 3.1416); // region with all tower of the same size: 0.087x0.087
      hcal->SetFillColor(Color_t(kRed));
      stack->Add(ecal);
      stack->Add(hcal);
   } else {
      ecal = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("ecalLego") );
      hcal = dynamic_cast<TH2F*>( stack->GetHists()->FindObject("hcalLego") );
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

}

