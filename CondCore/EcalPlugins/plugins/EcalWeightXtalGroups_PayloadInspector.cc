#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <string>

namespace {

  enum {kEBChannels = 61200, kEEChannels = 14648, kSides = 2, kRMS = 5, TEMPLATESAMPLES = 12};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100};           // endcaps lower and upper bounds on x and y

 /*****************************************************
    2d plot of Ecal WeightXtal Groups of 1 IOV
 *****************************************************/
  class EcalWeightXtalGroupsPlot : public cond::payloadInspector::PlotImage<EcalWeightXtalGroups> {

   public:
    EcalWeightXtalGroupsPlot() : cond::payloadInspector::PlotImage<EcalWeightXtalGroups>("Ecal Weight Xtal Groups - map ") {
      setSingleIov(true);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* barrel = new TH2F("EB", "mean EB", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "mean EE+", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "mean EE-", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);

      auto iov = iovs.front();
      std::shared_ptr<EcalWeightXtalGroups> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);

      if( payload.get() ){

        if (payload->barrelItems().empty())
          return false;

        for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
          uint32_t rawid = EBDetId::unhashIndex(cellid);

          std::vector<EcalXtalGroupId>::const_iterator value_ptr =  payload->find(rawid);
         // unsigned int id()
          if (value_ptr == payload->end())
            continue; // cell absent from payload
          
          unsigned int weight = (unsigned int)((*value_ptr).id());
          Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
          Double_t eta = (Double_t)(EBDetId(rawid)).ieta();



          if(eta > 0.)
           eta = eta - 0.5;   //   0.5 to 84.5
          else
           eta  = eta + 0.5;         //  -84.5 to -0.5
          
          barrel->Fill(phi, eta, weight);
        }// loop over cellid

        if (payload->endcapItems().empty())
          return false;
        
        // looping over the EE channels
        for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
          for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
            for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
              if(EEDetId::validDetId(ix, iy, iz)) {
                EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                uint32_t rawid = myEEId.rawId();

                std::vector<EcalXtalGroupId>::const_iterator value_ptr =  payload->find(rawid);
                
                if (value_ptr == payload->end())
                  continue; // cell absent from payload
                
                unsigned int weight = (unsigned int)((*value_ptr).id());
                
                if(iz == 1)
                  endc_p->Fill(ix, iy, weight);
                else
                  endc_m->Fill(ix, iy, weight);
              }  // validDetId 
      }    // payload

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 450);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Weight Xtal Groups, IOV %i", run));

      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad** pad = new TPad*;
      for (int obj = 0; obj < 3; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
        pad[obj]->Draw();
      }
      //      EcalDrawMaps ICMap;
      pad[0]->cd();
      //      ICMap.DrawEE(endc_m, 0., 2.);
      DrawEE(endc_m, 0., 2.5);
      pad[1]->cd();
      //      ICMap.DrawEB(barrel, 0., 2.);
      DrawEB(barrel, 0., 2.5);
      pad[2]->cd();
      //      ICMap.DrawEE(endc_p, 0., 2.);
      DrawEE(endc_p, 0., 2.5);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };


}

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalWeightXtalGroups){
  PAYLOAD_INSPECTOR_CLASS(EcalWeightXtalGroupsPlot);
}