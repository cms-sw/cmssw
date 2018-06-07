#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <memory>
#include <sstream>

namespace {


  enum {kEBChannels = 61200, kEEChannels = 14648};
  enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360, EBhistEtaMax = 171};   // barrel lower and upper bounds on eta and phi
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100, EEhistXMax = 220};           // endcaps lower and upper bounds on x and y

  /*******************************************************
   
     2d histogram of ECAL barrel Time Calib Errors of 1 IOV 

  *******************************************************/

  // inherit from one of the predefined plot class: Histogram2D
  class EcalTimeCalibErrorsEBMap : public cond::payloadInspector::Histogram2D<EcalTimeCalibErrors> {

  public:
    EcalTimeCalibErrorsEBMap() : cond::payloadInspector::Histogram2D<EcalTimeCalibErrors>("ECAL Barrel Time Calib Errors- map ",
                          "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI + 1,
                          "ieta", EBhistEtaMax, -MAX_IETA, MAX_IETA + 1) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      for (auto const & iov: iovs) {
        std::shared_ptr<EcalTimeCalibErrors> payload = Base::fetchPayload( std::get<1>(iov) );
        if( payload.get() ){
          // looping over the EB channels, via the dense-index, mapped into EBDetId's
          if (payload->barrelItems().empty()) return false;
          // set to -1 for ieta 0 (no crystal)
          for(int iphi = MIN_IPHI; iphi < MAX_IPHI+1; iphi++) fillWithValue(iphi, 0, -1);

          for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);

            // check the existence of ECAL Time Calib Errors, for a given ECAL barrel channel
            EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
            if (value_ptr == payload->end())
              continue; // cell absent from payload

            float weight = (float)(*value_ptr);

            // fill the Histogram2D here
            fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), weight);
          }// loop over cellid
        }// if payload.get()
      }// loop over IOV's (1 in this case)

      return true;

    }//fill method
  };


 /*******************************************************
   
     2d histogram of ECAL EndCaps Time Calib Errors of 1 IOV 

  *******************************************************/

  class EcalTimeCalibErrorsEEMap : public cond::payloadInspector::Histogram2D<EcalTimeCalibErrors> {

  private:
    int EEhistSplit = 20;

  public:
    EcalTimeCalibErrorsEEMap() : cond::payloadInspector::Histogram2D<EcalTimeCalibErrors>( "ECAL Endcap Time Calib Errors - map ",
                           "ix", EEhistXMax, IX_MIN, EEhistXMax + 1, 
                           "iy", IY_MAX, IY_MIN, IY_MAX + 1) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      for (auto const & iov: iovs) {
        std::shared_ptr<EcalTimeCalibErrors> payload = Base::fetchPayload( std::get<1>(iov) );
        if( payload.get() ){
          if (payload->endcapItems().empty()) return false;

          // set to -1 everywhwere
          for(int ix = IX_MIN; ix < EEhistXMax + 1; ix++)
            for(int iy = IY_MAX; iy < IY_MAX + 1; iy++)
              fillWithValue(ix, iy, -1);

          for (int cellid = 0;  cellid < EEDetId::kSizeForDenseIndexing; ++cellid){    // loop on EE cells
            if (EEDetId::validHashIndex(cellid)){  
              uint32_t rawid = EEDetId::unhashIndex(cellid);
              EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
              if (value_ptr == payload->end())
                continue; // cell absent from payload

              float weight = (float)(*value_ptr);
              EEDetId myEEId(rawid);
              if(myEEId.zside() == -1)
                fillWithValue(myEEId.ix(), myEEId.iy(), weight);
              else
                fillWithValue(myEEId.ix() + IX_MAX + EEhistSplit, myEEId.iy(), weight);
            }  // validDetId 
          }   // loop over cellid

        }    // payload
      }     // loop over IOV's (1 in this case)
      return true;
    }// fill method

  };


 /*************************************************
     2d plot of Ecal Time Calib Errors of 1 IOV
  *************************************************/
  class EcalTimeCalibErrorsPlot : public cond::payloadInspector::PlotImage<EcalTimeCalibErrors> {

  public:
    EcalTimeCalibErrorsPlot() : cond::payloadInspector::PlotImage<EcalTimeCalibErrors>("Ecal Time Calib Errors - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* barrel = new TH2F("EB", "mean EB", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "mean EE+", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "mean EE-", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);

      auto iov = iovs.front();
      std::shared_ptr<EcalTimeCalibErrors> payload = fetchPayload( std::get<1>(iov) );
      unsigned int run = std::get<0>(iov);

      if( payload.get() ){

        if (payload->barrelItems().empty())
          return false;

        for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
          uint32_t rawid = EBDetId::unhashIndex(cellid);
          EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
          if (value_ptr == payload->end())
            continue; // cell absent from payload
          
          float weight = (float)(*value_ptr);
          Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
          Double_t eta = (Double_t)(EBDetId(rawid)).ieta();

          if(eta > 0.)
           eta = eta - 0.5;   //   0.5 to 84.5
          else
           eta  = eta + 0.5;  //  -84.5 to -0.5
          
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
                EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
                if (value_ptr == payload->end())
                  continue; // cell absent from payload
                
                float weight = (float)(*value_ptr);
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
      t1.DrawLatex(0.5, 0.96, Form("Ecal Time Calib Errors, IOV %i", run));

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


  /*****************************************************************
     2d plot of Ecal Time Calib Errors difference between 2 IOVs
  *****************************************************************/
  class EcalTimeCalibErrorsDiff : public cond::payloadInspector::PlotImage<EcalTimeCalibErrors> {

  public:
    EcalTimeCalibErrorsDiff() : cond::payloadInspector::PlotImage<EcalTimeCalibErrors>("Ecal Time Calib Errors difference ") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F* barrel = new TH2F("EB", "mean EB", MAX_IPHI, 0, MAX_IPHI, 2 * MAX_IETA, -MAX_IETA, MAX_IETA);
      TH2F* endc_p = new TH2F("EE+", "mean EE+", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      TH2F* endc_m = new TH2F("EE-", "mean EE-", IX_MAX, IX_MIN, IX_MAX + 1, IY_MAX, IY_MIN, IY_MAX + 1);
      float pEBmin, pEEmin, pEBmax, pEEmax;
      pEBmin = 10.;
      pEEmin = 10.;
      pEBmax = -10.;
      pEEmax = -10.;

      unsigned int run[2], irun = 0;
      float pEB[kEBChannels], pEE[kEEChannels];
      for ( auto const & iov: iovs) {
        std::shared_ptr<EcalTimeCalibErrors> payload = fetchPayload( std::get<1>(iov) );
        run[irun] = std::get<0>(iov);

        if( payload.get() ){

          if (payload->barrelItems().empty())
            return false;

          for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
            uint32_t rawid = EBDetId::unhashIndex(cellid);
            EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
            if (value_ptr == payload->end())
              continue; // cell absent from payload

            float weight = (float)(*value_ptr);
            if(irun == 0) 
              pEB[cellid] = weight;
            else {
              Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
              Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
              
              if(eta > 0.)
                eta = eta - 0.5;   //   0.5 to 84.5
              else
                eta  = eta + 0.5;  //  -84.5 to -0.5
              
              double diff = weight - pEB[cellid];
              
              if(diff < pEBmin)
                pEBmin = diff;
              if(diff > pEBmax)
                pEBmax = diff;

              barrel->Fill(phi, eta, diff);
            }
          }// loop over cellid

          if (payload->endcapItems().empty())
            return false;

          // looping over the EE channels
          for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
            for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
              for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
                if(EEDetId::validDetId(ix, iy, iz)) {
                  EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
                  uint32_t cellid = myEEId.hashedIndex();
                  uint32_t rawid = myEEId.rawId();
                  EcalFloatCondObjectContainer::const_iterator value_ptr =  payload->find(rawid);
                  
                  if (value_ptr == payload->end())
                    continue; // cell absent from payload
                  float weight = (float)(*value_ptr);
                  if(irun == 0) 
                    pEE[cellid] = weight;
                  else {
                    double diff = weight - pEE[cellid];
                    if(diff < pEEmin)
                      pEEmin = diff;

                    if(diff > pEEmax)
                      pEEmax = diff;
                    if(iz == 1)
                      endc_p->Fill(ix, iy, diff);
                    else
                      endc_m->Fill(ix, iy, diff);
                  }
                }  // validDetId 
        }    // payload
        irun++;
      }      // loop over IOVs

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map", 1600, 450);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Ecal Time Calib Errors Diff, IOV %i - %i", run[1], run[0]));

      float xmi[3] = {0.0 , 0.24, 0.76};
      float xma[3] = {0.24, 0.76, 1.00};
      TPad** pad = new TPad*;
      
      for (int obj = 0; obj < 3; obj++) {
        pad[obj] = new TPad(Form("p_%i", obj),Form("p_%i", obj), xmi[obj], 0.0, xma[obj], 0.94);
        pad[obj]->Draw();
      }

      pad[0]->cd();
      DrawEE(endc_m, pEEmin, pEEmax);
      pad[1]->cd();
      DrawEB(barrel, pEBmin, pEBmax);
      pad[2]->cd();
      DrawEE(endc_p, pEEmin, pEEmax);

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };



/*******************************************************
 2d plot of Ecal Time Calib Errors Summary of 1 IOV
 *******************************************************/
class EcalTimeCalibErrorsSummaryPlot: public cond::payloadInspector::PlotImage<EcalTimeCalibErrors>{
  public:
    EcalTimeCalibErrorsSummaryPlot():
      cond::payloadInspector::PlotImage<EcalTimeCalibErrors>("Ecal Time Calib Errors Summary- map "){
        setSingleIov(true);
    }

  bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
    auto iov=iovs.front();
    std::shared_ptr <EcalTimeCalibErrors> payload = fetchPayload(std::get<1> (iov));
    unsigned int run=std::get<0> (iov);
    TH2F* align;
    int NbRows;

    if(payload.get()){
      NbRows=2;
      align=new TH2F("Ecal Time Calib Errors","EB/EE      mean_x      rms        num_x",4,0,4,NbRows,0,NbRows);
      

      float mean_x_EB=0.0f;   
      float mean_x_EE=0.0f;

      float rms_EB=0.0f;
      float rms_EE=0.0f;
                       
      int num_x_EB=0;      
      int num_x_EE=0;


      payload->summary(mean_x_EB, rms_EB, num_x_EB, mean_x_EE, rms_EE, num_x_EE);

      double row = NbRows-0.5;

      align->Fill(0.5,row,1);
      align->Fill(1.5,row,mean_x_EB);
      align->Fill(2.5,row,rms_EB);
      align->Fill(3.5,row,num_x_EB);
      
      row--;
      
      align->Fill(0.5,row,2);
      align->Fill(1.5,row,mean_x_EE);
      align->Fill(2.5,row,rms_EE);
      align->Fill(3.5,row,num_x_EE);
    }else
      return false;

    gStyle->SetPalette(1);
    gStyle->SetOptStat(0);
    TCanvas canvas("CC map", "CC map", 1000, 1000);
    TLatex t1;
    t1.SetNDC();
    t1.SetTextAlign(26);
    t1.SetTextSize(0.04);
    t1.SetTextColor(2);
    t1.DrawLatex(0.5, 0.96,Form("Ecal Time Calib Errors Summary, IOV %i", run));


    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
    pad->Draw();
    pad->cd();
    align->Draw("TEXT");
    TLine* l = new TLine;
    l->SetLineWidth(1);

    for (int i = 1; i < NbRows; i++) {
      double y = (double) i;
      l = new TLine(0., y, 4., y);
      l->Draw();
    }

    for (int i = 1; i < 4; i++) {
      double x = (double) i;
      double y = (double) NbRows;
      l = new TLine(x, 0., x, y);
      l->Draw();
    }

    align->GetXaxis()->SetTickLength(0.);
    align->GetXaxis()->SetLabelSize(0.);
    align->GetYaxis()->SetTickLength(0.);
    align->GetYaxis()->SetLabelSize(0.);

    std::string ImageName(m_imageFileName);
    canvas.SaveAs(ImageName.c_str());

    return true;
  }
};



}


// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalTimeCalibErrors ){
  PAYLOAD_INSPECTOR_CLASS( EcalTimeCalibErrorsEBMap );
  PAYLOAD_INSPECTOR_CLASS( EcalTimeCalibErrorsEEMap );
  PAYLOAD_INSPECTOR_CLASS( EcalTimeCalibErrorsPlot );
  PAYLOAD_INSPECTOR_CLASS( EcalTimeCalibErrorsDiff );
  PAYLOAD_INSPECTOR_CLASS( EcalTimeCalibErrorsSummaryPlot );
}
