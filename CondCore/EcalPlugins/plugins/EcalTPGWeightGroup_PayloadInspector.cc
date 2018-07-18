#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"
// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {

  enum {kEBTotalTowers = 2448, kEETotalTowers = 1584};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 17, MAX_IPHI = 72};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100, EEhistXMax = 220};           // endcaps lower and upper bounds on x and y
/***********************************************
    2d plot of EcalTPGWeightGroup of 1 IOV
************************************************/
  class EcalTPGWeightGroupPlot : public cond::payloadInspector::PlotImage<EcalTPGWeightGroup> {

  public:
    EcalTPGWeightGroupPlot() : cond::payloadInspector::PlotImage<EcalTPGWeightGroup>("EcalTPGWeightGroup - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
      uint32_t minEB=0;
      uint32_t maxEB=2;
      uint32_t minEE=0;
      uint32_t maxEE=2;

      TH2F* barrel = new TH2F("EB","EB Tower Status", 72, 0, 72, 34, -17, 17);
      TH2F* endc_p = new TH2F("EE+","EE+ Tower Status",22, 0, 22, 22, 0, 22);
      TH2F* endc_m = new TH2F("EE-","EE- Tower Status",22, 0, 22, 22, 0, 22);

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGWeightGroup> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);

      if( payload.get() ){


        const EcalTPGWeightGroup::EcalTPGGroupsMap & map=(*payload).getMap();
        EcalTPGWeightGroup::EcalTPGGroupsMapItr it;

        for(it = map.begin() ; it != map.end() ; it++){
            EcalTrigTowerDetId rawid = EcalTrigTowerDetId::detIdFromDenseIndex((*it).first);
            EcalTrigTowerDetId ttId_eb(rawid);

            //std::cout<<(*it).second << std::endl;

            if(ttId_eb.subDet()==1){
              //barrel
              int ieta = ttId_eb.ieta();
              if(ieta < 0) ieta--;   // -1 to -17
              int iphi = ttId_eb.iphi() - 1;  // 0 to 71

              if(minEB > (*it).second)
                minEB = (*it).second;
              
              if(maxEB < (*it).second)
                maxEB = (*it).second;

              barrel->Fill(iphi, ieta,(*it).second);
            }


            if(EcalScDetId::validHashIndex((*it).first)){
              EcalScDetId rawid_ee = EcalScDetId::unhashIndex((*it).first);
              EcalScDetId ttId_ee(rawid_ee);

                //endcaps

              int ix = ttId_ee.ix();
              int iy = ttId_ee.iy();
              int zside = ttId_ee.zside();


              if(minEE > (*it).second)
                minEE = (*it).second;
                
              if(maxEE < (*it).second)
                maxEE = (*it).second;
              
              if(zside == 1){
                endc_p->Fill(ix, iy, (*it).second);
              }else{
                endc_m->Fill(ix, iy, (*it).second);
              }

            }

        }

       
      }    // payload


     TCanvas canvas("CC map","CC map",800,800);
     TLatex t1;
     t1.SetNDC();
     t1.SetTextAlign(26);
     t1.SetTextSize(0.05);
     t1.DrawLatex(0.5, 0.96, Form("Ecal TPG WeightGroup, IOV %i", run));

     //TPad* padb = new TPad("padb","padb", 0., 0.55, 1., 1.);
     TPad* padb = new TPad("padb","padb", 0., 0.45, 1., 0.9);
     padb->Draw();
     TPad* padem = new TPad("padem","padem", 0., 0., 0.45, 0.45);
     padem->Draw();
     TPad* padep = new TPad("padep","padep", 0.55, 0., 1., 0.45);
     padep->Draw();

     TLine* l = new TLine(0., 0., 0., 0.);
     l->SetLineWidth(1);
     padb->cd();
     barrel->SetStats(false);
     barrel->SetMaximum(maxEB);
     barrel->SetMinimum(minEB);
     barrel->Draw("colz");
     
     //barrel->Draw("col");
     for(int i = 0; i <17; i++) {
       Double_t x = 4.+ (i * 4);
       l = new TLine(x, -17., x, 17.);
       l->Draw();
     }

     l = new TLine(0., 0., 72., 0.);
     l->Draw();
 

     padem->cd();
     DrawEE_Tower(endc_m,l,minEE,maxEE);

 
     padep->cd();
     DrawEE_Tower(endc_p,l,minEE,maxEE);
       

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
  }// fill method

};


}


// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGWeightGroup){
  PAYLOAD_INSPECTOR_CLASS(EcalTPGWeightGroupPlot);
}