#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>
#include <cstring>

namespace {
  enum {TEMPLATESAMPLES=5};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 17, MAX_IPHI = 72};   // barrel lower and upper bounds on eta and phi

  /***********************************************
     2d plot of EcalTPGFineGrainEBIdMap of 1 IOV
  ************************************************/
  class EcalTPGFineGrainEBIdMapPlot : public cond::payloadInspector::PlotImage<EcalTPGFineGrainEBIdMap> {

  public:
    EcalTPGFineGrainEBIdMapPlot() : cond::payloadInspector::PlotImage<EcalTPGFineGrainEBIdMap>("Ecal TPGFineGrainEBIdMap - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F** barrel = new TH2F*[TEMPLATESAMPLES];
      double pEBmin[TEMPLATESAMPLES], pEBmax[TEMPLATESAMPLES];
      std::string text[TEMPLATESAMPLES]= {"ThresholdETLow","ThresholdETHigh","RatioLow","RatioHigh","LUT"};
      int EBcnt = 0;

      for(int s = 0; s < TEMPLATESAMPLES; ++s) {
        char *y = new char[text[s].length() + 1];
        std::strcpy(y, text[s].c_str());
        barrel[s]=new TH2F("EB",y, MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      
      }

      uint32_t ThresholdETLow =0;
      uint32_t ThresholdETHigh =0;
      uint32_t RatioLow =0;
      uint32_t RatioHigh =0;
      uint32_t LUT =0;

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGFineGrainEBIdMap> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      if( payload.get() ){

        const std::map<uint32_t, EcalTPGFineGrainConstEB> &towerMap = (*payload).getMap();        
        std::map<uint32_t, EcalTPGFineGrainConstEB>::const_iterator it = towerMap.begin();

        
        for(int iphi = MIN_IPHI-1; iphi <= MAX_IPHI; iphi++){
          for(int ieta= -1*MAX_IETA; ieta < MAX_IETA; ieta++){



            //if(ieta > 0) ieta--;
            

              EcalTPGFineGrainConstEB fg=(*it).second;
              
				      fg.getValues(ThresholdETLow,ThresholdETHigh,RatioLow,RatioHigh,LUT);

              barrel[0]->Fill(iphi, ieta,ThresholdETLow);
              barrel[1]->Fill(iphi, ieta,ThresholdETHigh);
              barrel[2]->Fill(iphi, ieta,RatioLow);
              barrel[3]->Fill(iphi, ieta,RatioHigh);
              barrel[4]->Fill(iphi, ieta,LUT);
            

              if(ThresholdETLow<pEBmin[0])pEBmin[0]=ThresholdETLow;
              if(ThresholdETHigh<pEBmin[1])pEBmin[1]=ThresholdETHigh;
              if(RatioLow<pEBmin[2])pEBmin[2]=RatioLow;
              if(RatioHigh<pEBmin[3])pEBmin[3]=RatioHigh;
              if(LUT<pEBmin[4])pEBmin[4]=LUT;
              
              if(ThresholdETLow>pEBmax[0])pEBmax[0]=ThresholdETLow;
              if(ThresholdETHigh>pEBmax[1])pEBmax[1]=ThresholdETHigh;
              if(RatioLow>pEBmax[2])pEBmax[2]=RatioLow;
              if(RatioHigh>pEBmax[3])pEBmax[3]=RatioHigh;
              if(LUT>pEBmax[4])pEBmax[4]=LUT;




              EBcnt++;
          }     
        }

      

      
      }  // payload

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 2800);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.04);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPGFine Grain EBIdMap, IOV %i", run));
      
      float xmi = 0.24;
      float xma = 0.76;

      TPad** pad = new TPad*[TEMPLATESAMPLES];
      for (int s = 0; s < TEMPLATESAMPLES; s++) {
          float yma = 0.94 - (0.16 * s);
          float ymi = yma - 0.14;
          //pad[s] = new TPad(text[s],Form("Towers %i", EBcnt),xmi, ymi, xma, yma);
          char *y = new char[text[s].length() + 1];
          std::strcpy(y, text[s].c_str());

          pad[s] = new TPad(Form("Towers %i", EBcnt),y,xmi, ymi, xma, yma);
          pad[s]->Draw();
      }


      for(int s = 0; s < TEMPLATESAMPLES; s++) {
        pad[s]->cd();

        if(pEBmin[s] == pEBmax[s]) {   // same values everywhere!..
          pEBmin[s] = pEBmin[s] - 1.e-06;
          pEBmax[s] = pEBmax[s] + 1.e-06;
        }

        
        barrel[s]->SetMaximum(pEBmax[s]);
        barrel[s]->SetMinimum(pEBmin[s]);
        barrel[s]->Draw("colz");

        TLine* l = new TLine(0., 0., 0., 0.);
        l->SetLineWidth(1);
        for(int i = 0; i < MAX_IETA; i++) {
          Double_t x = 4.+ (i * 4);
          l = new TLine(x, -MAX_IETA, x, MAX_IETA);
          l->Draw();
        }

      }


      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());

      return true;
    }// fill method
  };

}


// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGFineGrainEBIdMap){
  PAYLOAD_INSPECTOR_CLASS(EcalTPGFineGrainEBIdMapPlot);
}