#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESChannelStatusCode.h"
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
  enum { kESChannels = 137216 };
  enum { IX_MIN = 1, IY_MIN = 1, IX_MAX = 40, IY_MAX = 40 };  // endcaps lower and upper bounds on x and y

  /********************************************
      2d plot of ES channel status of 1 IOV
  *********************************************/
  class ESChannelStatusPlot : public cond::payloadInspector::PlotImage<ESChannelStatus> {
  public:
    ESChannelStatusPlot() : cond::payloadInspector::PlotImage<ESChannelStatus>("ES channel status") {
      setSingleIov(true);
    }
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      TH2F*** esmap = new TH2F**[2];
      std::string title[2][2] = {{"ES+F", "ES-F"}, {"ES+R", "ES-R"}};
      for (int plane = 0; plane < 2; plane++) {
        esmap[plane] = new TH2F*[2];
        for (int side = 0; side < 2; side++)
          esmap[plane][side] = new TH2F(
              Form("esmap%i%i", plane, side), title[plane][side].c_str(), IX_MAX, 0, IX_MAX, IY_MAX, 0, IY_MAX);
      }
      Int_t escount = 0;
      unsigned int run = 0;
      auto iov = iovs.front();
      std::shared_ptr<ESChannelStatus> payload = fetchPayload(std::get<1>(iov));
      run = std::get<0>(iov);
      if (payload.get()) {
        // looping over all the ES channels
        for (int id = 0; id < kESChannels; id++)
          if (ESDetId::validHashIndex(id)) {
            ESDetId myESId = ESDetId::unhashIndex(id);
            int side = myESId.zside();  // -1, 1
            if (side < 0)
              side = 1;
            else
              side = 0;
            int plane = myESId.plane() - 1;  // 1, 2
            if (side < 0 || side > 1 || plane < 0 || plane > 1) {
              std::cout << " channel " << id << " side " << myESId.zside() << " plane " << myESId.plane() << std::endl;
              return false;
            }
            ESChannelStatusCode status_it = (payload->getMap())[myESId];
            int status = status_it.getStatusCode();
            if (status != 0) {
              if (myESId.strip() == 1) {  // we get 32 times the same status, plot it only once!
                esmap[plane][side]->Fill(myESId.six() - 1, myESId.siy() - 1, status);
                //		std::cout << " channel " << id << " side " << myESId.zside() << " plane " << myESId.plane()
                //			  << " x " << myESId.six() << " y " << myESId.siy() << " strip " << myESId.strip()
                //			  << " status " << status << std::endl;
                escount++;
              }
            }
          }  // validHashIndex
      }      // payload

      gStyle->SetOptStat(0);
      gStyle->SetPalette(1);
      TCanvas canvas("CC map", "CC map", 1680, 1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("ES Channel Status, IOV %i", run));
      t1.SetTextSize(0.025);

      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad*** pad = new TPad**[2];
      for (int plane = 0; plane < 2; plane++) {
        pad[plane] = new TPad*[2];
        for (int side = 0; side < 2; side++) {
          float yma = 0.94 - (0.46 * plane);
          float ymi = yma - 0.44;
          pad[plane][side] =
              new TPad(Form("p_%i_%i", plane, side), Form("p_%i_%i", plane, side), xmi[side], ymi, xma[side], yma);
          pad[plane][side]->Draw();
        }
      }

      for (int side = 0; side < 2; side++) {
        for (int plane = 0; plane < 2; plane++) {
          pad[plane][side]->cd();
          esmap[plane][side]->Draw("colz1");
          DrawES(plane, side);
        }
      }
      canvas.cd();
      t1.SetTextSize(0.025);
      int Nbdead = escount * 32;
      //      int ipercent = Nbdead * 1000000 / kESChannels;
      //      float percent = (float)ipercent / 1000000.;    // keep 2 digits
      //      t1.DrawLatex(0.1, 0.94, Form("Number of dead strips %i (%f)", Nbdead, percent));
      t1.DrawLatex(0.5, 0.92, Form("Number of dead strips %i", Nbdead));

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };

  /*************************************************************
      2d plot of ES channel status difference between 2 IOVs
  **************************************************************/
  template <cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class ESChannelStatusDiffBase : public cond::payloadInspector::PlotImage<ESChannelStatus, nIOVs, ntags> {
  public:
    ESChannelStatusDiffBase()
        : cond::payloadInspector::PlotImage<ESChannelStatus, nIOVs, ntags>("ES channel status difference") {}
    bool fill() override {
      TH2F*** esmap = new TH2F**[2];
      std::string title[2][2] = {{"ES+F", "ES-F"}, {"ES+R", "ES-R"}};
      for (int plane = 0; plane < 2; plane++) {
        esmap[plane] = new TH2F*[2];
        for (int side = 0; side < 2; side++)
          esmap[plane][side] = new TH2F(
              Form("esmap%i%i", plane, side), title[plane][side].c_str(), IX_MAX, 0, IX_MAX, IY_MAX, 0, IY_MAX);
      }
      Int_t escount = 0;
      unsigned int run[2] = {0, 0};
      int stat[kESChannels];
      std::string l_tagname[2];
      //      std::cout << " running with " << nIOVs << " IOVs and " << ntags << " tags " << std::endl;
      auto iovs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      l_tagname[0] = cond::payloadInspector::PlotBase::getTag<0>().name;
      auto firstiov = iovs.front();
      run[0] = std::get<0>(firstiov);
      std::tuple<cond::Time_t, cond::Hash> lastiov;
      if (ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname[1] = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = iovs.back();
        l_tagname[1] = l_tagname[0];
      }
      run[1] = std::get<0>(lastiov);
      for (int irun = 0; irun < nIOVs; irun++) {
        std::shared_ptr<ESChannelStatus> payload;
        if (irun == 0) {
          payload = this->fetchPayload(std::get<1>(firstiov));
        } else {
          payload = this->fetchPayload(std::get<1>(lastiov));
        }
        //	std::cout << " irun " << irun << " tag " << l_tagname[irun] << " IOV " << run[irun] << std ::endl;
        if (payload.get()) {
          for (int id = 0; id < kESChannels; id++)  // looping over all the ES channels
            if (ESDetId::validHashIndex(id)) {
              ESDetId myESId = ESDetId::unhashIndex(id);
              ESChannelStatusCode status_it = (payload->getMap())[myESId];
              int status = status_it.getStatusCode();
              if (irun == 0)
                stat[id] = status;
              else {
                int side = myESId.zside();  // -1, 1
                if (side < 0)
                  side = 1;
                else
                  side = 0;
                int plane = myESId.plane() - 1;  // 1, 2
                if (side < 0 || side > 1 || plane < 0 || plane > 1) {
                  std::cout << " channel " << id << " side " << myESId.zside() << " plane " << myESId.plane()
                            << std::endl;
                  return false;
                }
                int diff = status - stat[id];
                if (diff != 0) {
                  if (myESId.strip() == 1) {  // we get 32 times the same status, plot it only once!
                    esmap[plane][side]->Fill(myESId.six() - 1, myESId.siy() - 1, diff);
                    escount++;
                  }
                }
              }  // 2nd IOV
            }    // validHashIndex
        }        // payload
      }          // loop over IOVs

      gStyle->SetOptStat(0);
      gStyle->SetPalette(1);
      TCanvas canvas("CC map", "CC map", 1680, 1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      int len = l_tagname[0].length() + l_tagname[1].length();
      if (ntags == 2) {
        if (len < 60) {
          t1.SetTextSize(0.03);
          t1.DrawLatex(
              0.5, 0.96, Form("%s IOV %i - %s  IOV %i", l_tagname[1].c_str(), run[1], l_tagname[0].c_str(), run[0]));
        } else {
          t1.SetTextSize(0.05);
          t1.DrawLatex(0.5, 0.96, Form("ES Channel Status, IOV %i - %i", run[1], run[0]));
        }
      } else {
        t1.SetTextSize(0.05);
        t1.DrawLatex(0.5, 0.96, Form("%s IOV %i - %i", l_tagname[0].c_str(), run[1], run[0]));
      }

      float xmi[2] = {0.0, 0.5};
      float xma[2] = {0.5, 1.0};
      TPad*** pad = new TPad**[2];
      for (int plane = 0; plane < 2; plane++) {
        pad[plane] = new TPad*[2];
        for (int side = 0; side < 2; side++) {
          float yma = 0.94 - (0.46 * plane);
          float ymi = yma - 0.44;
          pad[plane][side] =
              new TPad(Form("p_%i_%i", plane, side), Form("p_%i_%i", plane, side), xmi[side], ymi, xma[side], yma);
          pad[plane][side]->Draw();
        }
      }

      for (int side = 0; side < 2; side++) {
        for (int plane = 0; plane < 2; plane++) {
          pad[plane][side]->cd();
          esmap[plane][side]->Draw("colz1");
          DrawES(plane, side);
        }
      }
      canvas.cd();
      t1.SetTextSize(0.025);
      int Nbdead = escount * 32;
      //      int ipercent = Nbdead * 1000000 / kESChannels;
      //      float percent = (float)ipercent / 1000000.;    // keep 2 digits
      //      t1.DrawLatex(0.1, 0.94, Form("Number of dead strips %i (%f)", Nbdead, percent));
      t1.DrawLatex(0.5, 0.92, Form("Number of different strips %i", Nbdead));

      std::string ImageName(this->m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }  // fill method
  };
  using ESChannelStatusDiffOneTag = ESChannelStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 1>;
  using ESChannelStatusDiffTwoTags = ESChannelStatusDiffBase<cond::payloadInspector::SINGLE_IOV, 2>;
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(ESChannelStatus) {
  PAYLOAD_INSPECTOR_CLASS(ESChannelStatusPlot);
  PAYLOAD_INSPECTOR_CLASS(ESChannelStatusDiffOneTag);
  PAYLOAD_INSPECTOR_CLASS(ESChannelStatusDiffTwoTags);
}
