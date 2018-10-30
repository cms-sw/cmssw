#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>
#include <boost/algorithm/string/join.hpp>

namespace {

  /********************************************
     printing float values of reco paramters
  *********************************************/
  class HcalSiPMParametersSummary : public cond::payloadInspector::PlotImage<HcalSiPMParameters> {
  public:
    HcalSiPMParametersSummary() : cond::payloadInspector::PlotImage<HcalSiPMParameters>("HCAL SiPMParameter Summary") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      float fcByPE, darkCurrent1, darkCurrent2, tempVal;
      int type1, type2;
      std::shared_ptr<HcalSiPMParameters> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        
        std::string subDetName;
        std::vector<HcalSiPMParameter> itemsVec;


        //TODO: Abstract into a function that takes valMap as the argument
 
        TLatex label,val;
        std::vector<float> line;
        TLine* ll;
        TLine* lr;
        TLine* lt;
        TLine* lb;
        TCanvas* can = new TCanvas("SiPMParametersSummary","SiPMParametersSummary", 2000, 1680);
        //can->cd();
        //HcalObjRepresent::drawTable(2,2);
        can->Divide(2,2,0,0);
        int i = 1;


        label.SetNDC();
        label.SetTextAlign(26);
        label.SetTextSize(0.05);
        label.SetTextColor(2);
        label.DrawLatex(0.5, 0.96,Form("Hcal SiPM Parameters"));

        for(std::pair< std::string, std::vector<HcalSiPMParameter> > cont : (*payload).getAllContainers()){
            subDetName = std::get<0>(cont);
            if(subDetName[0] != 'H' || subDetName == "HT") continue;
            itemsVec = std::get<1>(cont);
            type1 = type2 = fcByPE = darkCurrent1 = darkCurrent2 = -1.0;
            for(HcalSiPMParameter par : itemsVec) {
              HcalDetId detId = HcalDetId(par.rawId());
              int iphi = detId.iphi();
              int ieta = detId.ieta();
              int depth = detId.depth();  
              //std::cout << "(subDet, eta, phi, depth) : type : DarkCurrent   |   (" << subDetName << ", " << std::to_string(ieta) << ", " << std::to_string(iphi) << ", " << std::to_string(depth) << ") : " << std::to_string(par.getType()) << " : " << std::to_string(par.getDarkCurrent()) << std::endl;
              if(iphi==0 && ieta==0 && depth==0) continue;
              fcByPE = par.getFCByPE();
              tempVal = par.getDarkCurrent();
              if(darkCurrent1 == -1.0) {
                darkCurrent1 = tempVal;
                type1 = par.getType();
              }
              else if(darkCurrent2 == -1.0 && par.getType() != type1){
                darkCurrent2 = tempVal;
                type2 = par.getType();
                //if(fcByPE != -1) break;
              }
            }
            if(type1 == -1) {
              type1 = darkCurrent1 = 0;
            }
            if(type2 == -1) {
              darkCurrent2 = darkCurrent1;
              type2 = type1;
            }

            //Fill Grid of subsystems            
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

            //label subsystem
	    label.SetNDC();
	    label.SetTextAlign(26);
	    label.SetTextSize(0.15);
	    label.SetTextColor(2);
	    label.DrawLatex(0.5, 0.85, subDetName.c_str());

          //table filling
          float startPosY = 0.75, endPosY = 0.2;
          float startPosX = 0.10, endPosX = 0.8;
          std::vector<float>::iterator linEle;
          std::vector<std::string>::iterator linStrEle;
          int j = 0;
          float xDiff, yDiff;
          // Header line
          std::stringstream lbl1, lbl2;
          lbl1 << "(Type " << std::to_string(type1) << ")";
          lbl2 << "(Type " << std::to_string(type2) << ")";
          std::vector<std::string> lblline = {"fcByPE", "Dark Current", "Dark Current"};
          xDiff = (endPosX - startPosX)/(lblline.size()-1);
          yDiff = (startPosY - endPosY);

          //printing header
          label.SetTextAlign(12);
          label.SetTextSize(0.05);
          label.SetTextColor(1);
          //ll = new TLine(startPosX -0.2*xDiff, startPosY, startPosX - 0.2*xDiff, startPosY - 0.5*yDiff);
          //ll->Draw();
          for(linStrEle = lblline.begin(); linStrEle != lblline.end(); ++linStrEle) {
            ll = new TLine(startPosX + (j+0.5)*xDiff, startPosY, startPosX + (j+0.5)*xDiff, startPosY - 0.5*yDiff);
            label.DrawLatex(startPosX + (j==0 ? -0.1 : (j-0.4))*xDiff , startPosY - 0.25*yDiff, (*linStrEle).c_str());
            if(j==1) label.DrawLatex(startPosX + (j==0 ? -0.1 : (j-0.31))*xDiff , startPosY - 0.34*yDiff, (lbl1.str()).c_str());
            if(j==2) label.DrawLatex(startPosX + (j==0 ? -0.1 : (j-0.31))*xDiff , startPosY - 0.34*yDiff, (lbl2.str()).c_str());
            if(j<2)ll->Draw();
            j++;
          }
          ll = new TLine(0,startPosY, 1, startPosY);
          ll->Draw();
          ll = new TLine(0,startPosY - 0.5 * yDiff, 1, startPosY - 0.5 * yDiff);
          ll->Draw();
  
          
  	  val.SetNDC();
  	  val.SetTextAlign(26);
  	  val.SetTextSize(0.055);
          line = {fcByPE, darkCurrent1, darkCurrent2};
          //ll = new TLine(startPosX -0.2*xDiff, startPosY - 0.5*yDiff, startPosX - 0.2*xDiff, startPosY - 1.5*yDiff);
          //ll->Draw();
          j = 0;
  
          for(linEle = line.begin(); linEle != line.end(); ++linEle) {
            ll = new TLine(startPosX + (j+0.5)*xDiff, startPosY - 0.5*yDiff, startPosX + (j+0.5)*xDiff, startPosY - 1.5*yDiff);
            val.DrawLatex(startPosX + j*xDiff , startPosY - yDiff, HcalObjRepresent::SciNotatStr((*linEle)).c_str());
            if(j < 2){ ll->Draw(); }
            j++;
          }
  
          
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
PAYLOAD_INSPECTOR_MODULE(HcalSiPMParameters){
  PAYLOAD_INSPECTOR_CLASS(HcalSiPMParametersSummary);
}
