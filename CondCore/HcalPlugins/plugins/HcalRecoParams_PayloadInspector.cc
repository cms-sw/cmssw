#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>

namespace {

  /********************************************
     printing float values of reco paramters
  *********************************************/
  class HcalRecoParamsSummary : public cond::payloadInspector::PlotImage<HcalRecoParams> {
  public:
    HcalRecoParamsSummary() : cond::payloadInspector::PlotImage<HcalRecoParams>("HCAL RecoParam Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalRecoParams> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        
        std::string subDetName;
        std::vector<HcalRecoParam> itemsVec;
        std::pair<std::string,int> valMap;


        //TODO: Abstract into a function that takes valMap as the argument
 
        TLatex label,val;
        TLine* ll;
        TLine* lr;
        TLine* lt;
        TLine* lb;
        TCanvas* can = new TCanvas("RecoParamsSummary","RecoParamsSummary",1680, 1680);
        //can->cd();
        //HcalObjRepresent::drawTable(2,2);
        can->Divide(2,2,0,0);
        int i = 1;
        int psID;
        label.SetNDC();
        label.SetTextAlign(26);
        label.SetTextSize(0.05);
        label.SetTextColor(2);
        label.DrawLatex(0.5, 0.96,Form("Hcal Pulse Shape IDs"));

        for(std::pair< std::string, std::vector<HcalRecoParam> > cont : (*payload).getAllContainers()){
            psID = 0;
            subDetName = std::get<0>(cont);
            if(subDetName[0] != 'H' || subDetName == "HT") continue;
            itemsVec = std::get<1>(cont);
            //valMap.insert(std::make_pair(subDetName,itemsVec.front().pulseShapeID()));
            can->cd(i);
            ll = new TLine(0,0,0,1);
            ll->SetLineWidth(4);
            ll->Draw();
            lt = new TLine(0,1,1,1);
            lt->SetLineWidth(4);
            lt->Draw();
            lb = new TLine(0,0,1,0);
            lb->SetLineWidth(4);
            lb->Draw();
            lr = new TLine(1,0,1,1);
            lr->SetLineWidth(4);
            lr->Draw();
	    label.SetNDC();
	    label.SetTextAlign(26);
	    label.SetTextSize(0.15);
	    //label.SetTextColor(2);
	    label.DrawLatex(0.5, 0.75, subDetName.c_str());
	    val.SetNDC();
	    val.SetTextAlign(26);
	    val.SetTextSize(0.1);
//	    val.SetTextColor(1);
            std::vector<HcalRecoParam>::iterator it;
            for(it = itemsVec.begin(); it != itemsVec.end(); it++) {
              psID = (*it).pulseShapeID();
              if(psID != 0) {
                psID = (*it).pulseShapeID(); 
                break;
              }
            }
	    val.DrawLatex(0.5, 0.25, std::to_string(psID).c_str());
            i++;
        }




        std::string ImageName(m_imageFileName);
        can->SaveAs(ImageName.c_str());
        return false;
        } else return false;
    }// fill method
  };
//TODO: Add a Change Summary?

} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalRecoParams){
  PAYLOAD_INSPECTOR_CLASS(HcalRecoParamsSummary);
}
