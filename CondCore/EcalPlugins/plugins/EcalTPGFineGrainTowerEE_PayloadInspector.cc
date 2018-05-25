#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {

/***********************************************
   2d plot of EcalTPGFineGrainTowerEE of 1 IOV
************************************************/
class EcalTPGFineGrainTowerEEPlot : public cond::payloadInspector::PlotImage<EcalTPGFineGrainTowerEE> {

  public:
    EcalTPGFineGrainTowerEEPlot() : cond::payloadInspector::PlotImage<EcalTPGFineGrainTowerEE>("EcalTPGFineGrainTowerEE - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
     TH2F* endc_p = new TH2F("EE+","EE+ Tower TPG FineGrain",22, 0, 22, 22, 0, 22);
     TH2F* endc_m = new TH2F("EE-","EE- Tower TPG FineGrain",22, 0, 22, 22, 0, 22);

     auto iov = iovs.front();
     std::shared_ptr<EcalTPGFineGrainTowerEE> payload = fetchPayload( std::get<1>(iov) );
     unsigned int run = std::get<0>(iov);
     double minEE = 0, maxEE = 1;


    if( payload.get() ){
        const EcalTPGFineGrainTowerEEMap &towerMap = (*payload).getMap();

        EcalTPGFineGrainTowerEEMapIterator it;
        for(it = towerMap.begin(); it != towerMap.end(); ++it) {
          if(EcalScDetId::validHashIndex((*it).first)) {
            EcalScDetId ttId = EcalScDetId::unhashIndex((*it).first); 

            int ix = ttId.ix();
            int iy = ttId.iy();
            int zside = ttId.zside();
            
            uint32_t weight = (uint32_t)((*it).second);

            if(zside == -1)
              endc_m->Fill(ix, iy, weight);
            else
              endc_p->Fill(ix, iy, weight);

            if(maxEE < weight)
              maxEE = weight;

            if(minEE>weight)
              minEE=weight;
          }
        }//tower map
    }//payload


    TCanvas canvas("CC map","CC map",800,800);
    TLatex t1;
    t1.SetNDC();
    t1.SetTextAlign(26);
    t1.SetTextSize(0.05);
    t1.DrawLatex(0.5, 0.96, Form("Ecal TPGFineGrain Tower EE, IOV %i", run));

    
    TPad* padem = new TPad("padem","padem", 0., 0.3, 0.45, 0.75);
    padem->Draw();
    TPad* padep = new TPad("padep","padep", 0.55, 0.3, 1., 0.75);
    padep->Draw();

    TLine* l = new TLine(0., 0., 72., 0.);
    l->Draw();
 
    int ixSectorsEE[136] = {
      8,14,14,17,17,18,18,19,19,20,20,21,21,20,20,19,19,18,18,17,
      17,14,14, 8, 8, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 4,
      4, 5, 5, 8, 8, 8, 9, 9,10,10,12,12,13,13,12,12,10,10, 9, 9,
      10,10, 0,11,11, 0,10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 0,12,13,
      13,14,14,15,15,16,16,17,17, 0, 9, 8, 8, 3, 3, 1, 0,13,14,14,
      19,19,21, 0, 9, 8, 8, 7, 7, 5, 5, 3, 3, 2, 0,13,14,14,15,15,
      17,17,19,19,20, 0,14,14,13,13,12,12,0};

    int iySectorsEE[136] = {
      1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 8,14,14,17,17,18,18,19,19,
      20,20,21,21,20,20,19,19,18,18,17,17,14,14, 8, 8, 5, 5, 4, 4,
      3, 3, 2, 2, 1, 4, 4, 7, 7, 9, 9,10,10,12,12,13,13,12,12,10,
      10, 9, 0,13,21, 0,13,13,14,14,15,15,16,16,18,18,19, 0,13,13,
      14,14,15,15,16,16,18,18,19, 0,11,11,12,12,13,13, 0,11,11,12,
      12,13,13, 0,10,10, 9, 9, 8, 8, 7, 7, 6, 6, 0,10,10, 9, 9, 8,
      8, 7, 7, 6, 6, 0, 2, 4, 4, 7, 7, 9, 0} ;

    padem->cd();
    endc_m->SetStats(false);
    endc_m->SetMinimum(minEE);
    endc_m->SetMaximum(maxEE);
    endc_m->Draw("colz");

    for ( int i = 0; i < 136; i=i+1) 
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) )
        l->DrawLine(ixSectorsEE[i], iySectorsEE[i], ixSectorsEE[i+1], iySectorsEE[i+1]);

 
    padep->cd();
    endc_p->SetStats(false);
    endc_p->SetMinimum(minEE);
    endc_p->SetMaximum(maxEE);
    endc_p->Draw("colz");
    
    for ( int i = 0; i < 136; i=i+1)
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) )
       l->DrawLine(ixSectorsEE[i], iySectorsEE[i], ixSectorsEE[i+1], iySectorsEE[i+1]);
       
     

    std::string ImageName(m_imageFileName);
    canvas.SaveAs(ImageName.c_str());

    return true;
  }// fill method

};


}


// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGFineGrainTowerEE){
  PAYLOAD_INSPECTOR_CLASS(EcalTPGFineGrainTowerEEPlot);
}