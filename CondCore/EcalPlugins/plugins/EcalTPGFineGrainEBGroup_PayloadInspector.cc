#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum {kEBTotalTowers = 2448, kEETotalTowers = 1584};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 17, MAX_IPHI = 72};   // barrel lower and upper bounds on eta and phi

  /***********************************************
     2d plot of EcalTPGFineGrainEBGroup of 1 IOV
  ************************************************/
  class EcalTPGFineGrainEBGroupPlot : public cond::payloadInspector::PlotImage<EcalTPGFineGrainEBGroup> {

  public:
    EcalTPGFineGrainEBGroupPlot() : cond::payloadInspector::PlotImage<EcalTPGFineGrainEBGroup>("EcalTPGFineGrainEBGroup - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* barrel = new TH2F("EB","Ecal TPGFineGrain EB Group", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      int EBcount = 0;
      double minEB=0, maxEB=1;


      auto iov = iovs.front();
      std::shared_ptr<EcalTPGFineGrainEBGroup> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){

        const EcalTPGFineGrainEBGroup::EcalTPGGroupsMap &towerMap = (*payload).getMap();
        
        EcalTPGFineGrainEBGroup::EcalTPGGroupsMapItr it;
        for(it = towerMap.begin(); it != towerMap.end(); ++it) {
            //EcalTrigTowerDetId ttId = EcalTrigTowerDetId::detIdFromDenseIndex((*it).first);
            EcalTrigTowerDetId ttId((*it).first);
            int ieta = ttId.ieta();
            //ieta--;
            if(ieta > 0) ieta--;   // -1 to -17
            int iphi = ttId.iphi() - 1;  // 0 to 71
            // std::cout << " sub det " << ttId.subDet() << " phi " << iphi << " eta " << ieta << std::endl;
            // ieta goes from -18 to -2 and 1 to 17. Change it to -17/-1 and 0/16
            
            //std::cout <<(*it).first<<std::endl;
            //std::cout << " ieta " << ieta << " phi " << iphi << " value " << (*it).second << std::endl;


            if(ttId.subDet() == 1) {   // barrel
              
              barrel->Fill(iphi, ieta,(*it).second);
            
              if(maxEB<(*it).second)
                maxEB=(*it).second;
              if(minEB>(*it).second)
                minEB=(*it).second;

              EBcount++;
            }
          
        }
      }  // payload

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      //      TCanvas canvas("CC map","CC map", 1600, 450);
      Double_t w = 1400;
      Double_t h = 1200;
      TCanvas canvas("c", "c", w, h);
      //      canvas.SetWindowSize(w + (w - canvas.GetWw()), h + (h - canvas.GetWh()));

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPGFine GrainEBGroup, IOV %i", run));
 
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 1; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), 0.0, 0.04, 1.0, 0.94);
        pad[obj]->Draw();
      }
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.2, 0.88, Form("%i towers", EBcount));
      //      canvas.cd();
      pad[0]->cd();
      //barrel->SetStats(0);
      barrel->SetMinimum(minEB);
      barrel->SetMaximum(maxEB);
      barrel->Draw("colz");
      TLine* l = new TLine(0., 0., 0., 0.);
      l->SetLineWidth(1);
      for(int i = 0; i < MAX_IETA; i++) {
        Double_t x = 4.+ (i * 4);
        l = new TLine(x, -MAX_IETA, x, MAX_IETA);
        l->Draw();
      }
      l = new TLine(0., 0., 72., 0.);
      l->Draw();

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

}


// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGFineGrainEBGroup){
  PAYLOAD_INSPECTOR_CLASS(EcalTPGFineGrainEBGroupPlot);
}