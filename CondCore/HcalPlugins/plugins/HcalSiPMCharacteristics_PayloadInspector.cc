#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"

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
  class HcalSiPMCharacteristicsSummary : public cond::payloadInspector::PlotImage<HcalSiPMCharacteristics> {
  public:
    HcalSiPMCharacteristicsSummary() : cond::payloadInspector::PlotImage<HcalSiPMCharacteristics>("HCAL RecoParam Ratios - map ") {
      setSingleIov( true );
    }



    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalSiPMCharacteristics> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        
        std::string subDetName;
        std::vector<HcalSiPMCharacteristics> itemsVec;
        std::pair<std::string,int> valMap;


        //TODO: Abstract into a function that takes valMap as the argument
 
        TLatex label,val;
        TLine* ll;
        TCanvas* can = new TCanvas("RecoParamsSummary","RecoParamsSummary",2400, 1680);
        can->cd();
        //HcalObjRepresent::drawTable(2,2);

        label.SetNDC();
        label.SetTextAlign(26);
        label.SetTextSize(0.05);
        label.SetTextColor(2);
        label.DrawLatex(0.5, 0.96,Form("Hcal SiPM Characteristics"));
        std::vector<float> line;
        HcalObjRepresent::drawTable(7,6);
 
        int nTypes = payload->getTypes();
        float startPosY = 0.9, endPosY = 0.1;
        float startPosX = 0.03, endPosX = 0.92;
        int type;
        std::vector<float>::iterator linEle;
        std::vector<std::string>::iterator linStrEle;
        int j = 0;
        float xDiff, yDiff;
        // Header line
        std::vector<std::string> lblline = {"Type", "Pixels", "parLin1", "parLin2", "parLin3","crossTalk"};//, aux1, aux2};
        xDiff = (endPosX - startPosX)/(lblline.size()-1);
        yDiff = (startPosY - endPosY)/nTypes;


        label.SetTextAlign(12);
        label.SetTextSize(0.05);
        label.SetTextColor(1);
        ll = new TLine(startPosX -0.2*xDiff, startPosY + 0.5*yDiff, startPosX - 0.2*xDiff, startPosY - 0.5*yDiff);
        ll->Draw();
        for(linStrEle = lblline.begin(); linStrEle != lblline.end(); ++linStrEle) {
          ll = new TLine(startPosX + (j+0.5)*xDiff, startPosY + 0.5*yDiff, startPosX + (j+0.5)*xDiff, startPosY - 0.5*yDiff);
          label.DrawLatex(startPosX + (j==0 ? -0.1 : (j-0.4))*xDiff , startPosY, (*linStrEle).c_str());
          ll->Draw();
          j++;
        }
        ll = new TLine(0,startPosY - 0.5 * yDiff, 1, startPosY - 0.5 * yDiff);
        ll->Draw();

        
	val.SetNDC();
	val.SetTextAlign(26);
	val.SetTextSize(0.035);
        for(int i = 0;i < nTypes;i++){
            type = payload->getType(i);


            line = {(float)type, (float)payload->getPixels(type), payload->getNonLinearities(type).at(0), payload->getNonLinearities(type).at(1), payload->getNonLinearities(type).at(2), payload->getCrossTalk(type)};
            ll = new TLine(startPosX -0.2*xDiff, startPosY - (i+0.5)*yDiff, startPosX - 0.2*xDiff, startPosY - (i+1.5)*yDiff);
            ll->Draw();
            j = 0;

            for(linEle = line.begin(); linEle != line.end(); ++linEle) {
              ll = new TLine(startPosX + (j+0.5)*xDiff, startPosY - (i+0.5)*yDiff, startPosX + (j+0.5)*xDiff, startPosY - (i+1.5)*yDiff);
              val.DrawLatex(startPosX + j*xDiff , startPosY - (i+1)*yDiff, HcalObjRepresent::SciNotatStr((*linEle)).c_str());
              ll->Draw();
              j++;
            }

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
PAYLOAD_INSPECTOR_MODULE(HcalSiPMCharacteristics){
  PAYLOAD_INSPECTOR_CLASS(HcalSiPMCharacteristicsSummary);
}
