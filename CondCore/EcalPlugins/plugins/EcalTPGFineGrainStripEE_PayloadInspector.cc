#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {
  enum {NTCC = 108, NTower = 28, NStrip = 5, NXtal = 5};
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100};// endcaps lower and upper bounds on x and y

  /***********************************************
     2d plot of ECAL TPGStripStatus of 1 IOV
  ************************************************/
  class EcalTPGFineGrainStripEEPlot : public cond::payloadInspector::PlotImage<EcalTPGFineGrainStripEE> {

  public:
    EcalTPGFineGrainStripEEPlot() : cond::payloadInspector::PlotImage<EcalTPGFineGrainStripEE>("EcalTPGFineGrainStripEE - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* endc_thresh_p = new TH2F("EE+","EE+ TPGFineGrainStrip Threshold", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_thresh_m = new TH2F("EE-","EE- TPGFineGrainStrip Threshold", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      
      TH2F* endc_lut_p = new TH2F("EE+","EE+ TPG Crystal Status Lut", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_lut_m = new TH2F("EE-","EE- TPG Crystal Status Lut", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      int EEcount[2] = {0, 0};

      std::string mappingFile = "Geometry/EcalMapping/data/EEMap.txt";   
      std::ifstream f(edm::FileInPath(mappingFile).fullPath().c_str());
      if (!f.good()) {
        std::cout << "EcalTPGFineGrainStripEE File EEMap.txt not found" << std::endl;
        throw cms::Exception("FileNotFound");
      }

      uint32_t rawEE[NTCC][NTower][NStrip][NXtal];
      int NbrawEE[NTCC][NTower][NStrip];
      for(int TCC = 0; TCC < NTCC; TCC++)
        for(int TT = 0; TT < NTower; TT++)
          for(int ST = 0; ST < NStrip; ST++)
            NbrawEE[TCC][TT][ST] = 0;

      while ( ! f.eof()) {
        int ix, iy, iz, CL;
        int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
        int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;
        f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip 
          >> tccid >> tower >> pseudostrip_in_TCC >> pseudostrip_in_TT ;

        EEDetId detid(ix,iy,iz,EEDetId::XYMODE);
        uint32_t rawId = detid.denseIndex();
        if(tccid > NTCC || tower > NTower || pseudostrip_in_TT > NStrip || xtal_in_pseudostrip > NXtal){
        //  std::cout << " tccid " << tccid <<  " tower " << tower << " pseudostrip_in_TT "<< pseudostrip_in_TT
        //      <<" xtal_in_pseudostrip " << xtal_in_pseudostrip << std::endl;
        }else {
          rawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1][xtal_in_pseudostrip - 1] = rawId;
          NbrawEE[tccid - 1][tower - 1][pseudostrip_in_TT - 1]++;
        }
      }   // read EEMap file


      f.close();
      //double wei[2] = {0., 0.};

      auto iov = iovs.front();
      std::shared_ptr<EcalTPGFineGrainStripEE> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);
      double max1=1.0, max2=1.0;

      if( payload.get() ){
        const EcalTPGFineGrainStripEEMap &stripMap = (*payload).getMap();
        //  std::cout << " tower map size " << stripMap.size() << std::endl;
        EcalTPGFineGrainStripEEMapIterator itSt;
        for(itSt = stripMap.begin(); itSt != stripMap.end(); ++itSt) {
          EcalTPGFineGrainStripEE::Item item=(*itSt).second;

            // let's decode the ID
            int strip = itSt->first/8;
            int pseudostrip = strip & 0x7;
            strip /= 8;
            int tt = strip & 0x7F;
            strip /= 128;
            int tccid = strip & 0x7F;
            int NbXtalInStrip = NbrawEE[tccid - 1][tt - 1][pseudostrip - 1];
            //if(NbXtalInStrip != NXtal) std::cout << " Strip TCC " << tccid << " TT " << tt << " ST " << pseudostrip
            //       << " Nx Xtals " << NbXtalInStrip << std::endl;


            for(int Xtal = 0; Xtal < NbXtalInStrip; Xtal++) {
              uint32_t rawId = rawEE[tccid - 1][tt - 1][pseudostrip - 1][Xtal];
              //  std::cout << " rawid " << rawId << std::endl;
              EEDetId detid = EEDetId::detIdFromDenseIndex(rawId);
              float x = (float)detid.ix();
              float y = (float)detid.iy();
              int iz = detid.zside();
              if(iz == -1) iz++;
              //if(Xtal == 0) wei[iz] += 1.;
              if(iz == 0) {
                endc_thresh_m->Fill(x + 0.5, y + 0.5, item.threshold);
                endc_lut_m->Fill(x + 0.5, y + 0.5, item.lut);
                EEcount[0]++;

                if(max1<item.threshold)
                  max1=item.threshold;

                if(max2<item.lut)
                  max2=item.lut;

              }else {
                endc_thresh_p->Fill(x + 0.5, y + 0.5, item.threshold);
                endc_lut_p->Fill(x + 0.5, y + 0.5, item.lut);
                EEcount[1]++;

                if(max1<item.threshold)
                  max1=item.threshold;

                if(max2<item.lut)
                  max2=item.lut;
              }

            }




        }
      }  // payload
      //      std::cout << " nb strip EE- " << wei[0] << " EE+ " << wei[1] << std::endl;

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      const Int_t NRGBs = 5;
      const Int_t NCont = 255;

      Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
      Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
      Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
      Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);

      Double_t w = 1200;
      Double_t h = 1400;
      TCanvas canvas("c", "c", w, h);


      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.04);
      t1.DrawLatex(0.5, 0.96, Form("Ecal TPG Fine Grain StripEE, IOV %i", run));
 

      float xmi[4] = {0.0, 0.5, 0.0, 0.5};
      float xma[4] = {0.5, 1.0, 0.5, 1.0};

      float ymi[4] = {0.47, 0.47, 0.0, 0.0};
      float yma[4] = {0.94, 0.94, 0.47, 0.47};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 4; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), xmi[obj], ymi[obj], xma[obj], yma[obj]);
        pad[obj]->Draw();
      }

     // t1.DrawLatex(0.2, 0.94, Form("%i crystals", EBstat));

      pad[0]->cd();
      DrawEE(endc_thresh_m, 0., max1);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEcount[0]));
      
      pad[1]->cd();
      DrawEE(endc_thresh_p, 0., max1);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEcount[1]));
      

      pad[2]->cd();
      DrawEE(endc_lut_m, 0., max2);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEcount[0]));

      pad[3]->cd();
      DrawEE(endc_lut_p, 0., max2);
      t1.DrawLatex(0.15, 0.92, Form("%i crystals", EEcount[1]));

      std::string ImageName(m_imageFileName);

      canvas.SaveAs(ImageName.c_str());

      delete pad;
      delete endc_lut_p;
      delete endc_lut_m;
      delete endc_thresh_p;
      delete endc_thresh_m;
      return true;
    }// fill method
  };

}

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGFineGrainStripEE){
  PAYLOAD_INSPECTOR_CLASS(EcalTPGFineGrainStripEEPlot);
}