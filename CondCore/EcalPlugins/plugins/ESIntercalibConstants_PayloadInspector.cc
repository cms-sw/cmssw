#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "CondCore/EcalPlugins/plugins/ESDrawUtils.h"

#include <memory>
#include <sstream>

#include "TStyle.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"


namespace {
  enum {kESChannels = 137216};
  enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 40, IY_MAX = 40};         // endcaps lower and upper bounds on x and y


  /*********************************************************
       2d plot of ES channel status of 1 IOV
  *********************************************************/
  class ESIntercalibConstantsPlot : public cond::payloadInspector::PlotImage<ESIntercalibConstants> {

  public:
    ESIntercalibConstantsPlot() : cond::payloadInspector::PlotImage<ESIntercalibConstants>("ES IntercalibConstants") {
      setSingleIov( true );
    }
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      TH2F*** esmap = new TH2F**[2];
      std::string title[2][2] = {{"ES+F","ES-F"},{"ES+R","ES-R"}};
      for (int plane = 0; plane < 2; plane++) {
	esmap[plane] = new TH2F*[2];
	for (int side = 0; side < 2; side++)
	  esmap[plane][side] = new TH2F(Form("esmap%i%i",plane,side),title[plane][side].c_str(), IX_MAX, 0, IX_MAX, IY_MAX, 0, IY_MAX);
      }
      unsigned int run = 0;
      float valmin = 999.;
      auto iov = iovs.front();
      std::shared_ptr<ESIntercalibConstants> payload = fetchPayload( std::get<1>(iov) );
      run = std::get<0>(iov);
      if( payload.get() ){
	// looping over all the ES channels
	for(int id = 0; id < kESChannels; id++)
	  if(ESDetId::validHashIndex(id)) {
	    ESDetId myESId = ESDetId::unhashIndex(id);
	    int side = myESId.zside();   // -1, 1
	    if(side < 0) side = 1;
	    else side = 0;
	    int plane = myESId.plane() - 1;  // 1, 2
	    if(side < 0 || side > 1 || plane < 0 || plane > 1) {
	      std::cout << " channel " << id << " side " << myESId.zside() << " plane " << myESId.plane() << std::endl;
	      return false;
	    }
	    ESIntercalibConstants::const_iterator IC_it =  payload->find(myESId);
	    float value = *IC_it;
	    //	    if(id < 64) std::cout << " channel " << id << " value " << value << std::endl;
	    if(myESId.strip() == 1) {    // we get 32 times the same value, plot it only once!
	      esmap[plane][side]->Fill(myESId.six() -1, myESId.siy() -1, value);
	      if(value < valmin) valmin = value;
	    }
	  }  // validHashIndex
      } // payload
 
      gStyle->SetOptStat(0);
      gStyle->SetPalette(1);
      TCanvas canvas("CC map","CC map",1680,1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("ES Intercalib Constants, IOV %i", run));
      t1.SetTextSize(0.025);

      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad*** pad = new TPad**[2];
      for (int plane = 0; plane < 2; plane++) {
	pad[plane] = new TPad*[2];
	for (int side = 0; side < 2; side++) {
	  float yma = 0.94 - (0.46 * plane);
	  float ymi = yma - 0.44;
	  pad[plane][side] = new TPad(Form("p_%i_%i", plane, side),Form("p_%i_%i", plane, side),
				   xmi[side], ymi, xma[side], yma);
	  pad[plane][side]->Draw();
	}
      }

      int min = valmin * 10 - 1.;
      valmin = (float)min / 10.;
      for (int side = 0; side < 2; side++) {
	for (int plane = 0; plane < 2; plane++) {
	  pad[plane][side]->cd();
	  esmap[plane][side]->Draw("colz1");
	  esmap[plane][side]->SetMinimum(valmin);
	  DrawES(plane, side);
	}
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

  /************************************************************************
       2d plot of ES channel status difference between 2 IOVs
  ************************************************************************/
  class ESIntercalibConstantsDiff : public cond::payloadInspector::PlotImage<ESIntercalibConstants> {

  public:
    ESIntercalibConstantsDiff() : cond::payloadInspector::PlotImage<ESIntercalibConstants>("ES IntercalibConstants difference") {
      setSingleIov(false);
    }
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      TH2F*** esmap = new TH2F**[2];
      std::string title[2][2] = {{"ES+F","ES-F"},{"ES+R","ES-R"}};
      for (int plane = 0; plane < 2; plane++) {
	esmap[plane] = new TH2F*[2];
	for (int side = 0; side < 2; side++)
	  esmap[plane][side] = new TH2F(Form("esmap%i%i",plane,side),title[plane][side].c_str(), IX_MAX, 0, IX_MAX, IY_MAX, 0, IY_MAX);
      }
      unsigned int run[2], irun = 0;
      float val[kESChannels], valmin = 999.;
      for ( auto const & iov: iovs) {
	std::shared_ptr<ESIntercalibConstants> payload = fetchPayload( std::get<1>(iov) );
	run[irun] = std::get<0>(iov);
	//	std::cout << " irun " << irun << " IOV " << run[irun] << std::endl;
	if( payload.get() ){
	  for(int id = 0; id < kESChannels; id++)	  // looping over all the ES channels
	    if(ESDetId::validHashIndex(id)) {
	      ESDetId myESId = ESDetId::unhashIndex(id);
	      //	      ESIntercalibConstantsCode status_it = (payload->getMap())[myESId];
	      ESIntercalibConstants::const_iterator IC_it =  payload->find(myESId);
	      float value = *IC_it;
	      //	      int status = status_it.getStatusCode();
	      if(irun == 0)
		val[id] = value;
	      else {
		int side = myESId.zside();   // -1, 1
		if(side < 0) side = 1;
		else side = 0;
		int plane = myESId.plane() - 1;  // 1, 2
		if(side < 0 || side > 1 || plane < 0 || plane > 1) {
		  std::cout << " channel " << id << " side " << myESId.zside() << " plane " << myESId.plane() << std::endl;
		  return false;
		}
		//		int diff = status - stat[id];
		int diff = value - val[id];
		if(myESId.strip() == 1) {    // we get 32 times the same value, plot it only once!
		  esmap[plane][side]->Fill(myESId.six() -1, myESId.siy() -1, diff);
		  if(diff < valmin) valmin = diff;
		}
	      }  // 2nd IOV
	    }  // validHashIndex
	} // payload
	irun++;
      }       // loop over IOVs

      gStyle->SetOptStat(0);
      gStyle->SetPalette(1);
      TCanvas canvas("CC map","CC map",1680,1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("ES Intercalib Constants, IOV %i - %i", run[1], run[0]));
      t1.SetTextSize(0.025);

      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad*** pad = new TPad**[2];
      for (int plane = 0; plane < 2; plane++) {
	pad[plane] = new TPad*[2];
	for (int side = 0; side < 2; side++) {
	  float yma = 0.94 - (0.46 * plane);
	  float ymi = yma - 0.44;
	  pad[plane][side] = new TPad(Form("p_%i_%i", plane, side),Form("p_%i_%i", plane, side),
				   xmi[side], ymi, xma[side], yma);
	  pad[plane][side]->Draw();
	}
      }

      int min = valmin * 10 - 1.;
      valmin = (float)min / 10.;
      for (int side = 0; side < 2; side++) {
	for (int plane = 0; plane < 2; plane++) {
	  pad[plane][side]->cd();
	  esmap[plane][side]->Draw("colz1");
	  esmap[plane][side]->SetMinimum(valmin);
	  DrawES(plane, side);
	}
      }

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };
} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(ESIntercalibConstants){
  PAYLOAD_INSPECTOR_CLASS(ESIntercalibConstantsPlot);
  PAYLOAD_INSPECTOR_CLASS(ESIntercalibConstantsDiff);
}
