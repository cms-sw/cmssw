#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

#include "TLegend.h"
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

  using namespace cond::payloadInspector;

  class HcalRespCorrContainer : public HcalObjRepresent::HcalDataContainer<HcalRespCorrs, HcalRespCorr> {
  public:
    HcalRespCorrContainer(std::shared_ptr<HcalRespCorrs> payload, unsigned int run)
        : HcalObjRepresent::HcalDataContainer<HcalRespCorrs, HcalRespCorr>(payload, run) {}
    float getValue(const HcalRespCorr* rCor) override { return rCor->getValue(); }
  };

  /******************************************
     2d plot of HCAL RespCorr of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     2d plot of HCAL RespCorrs difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsRatioAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload(std::get<1>(iov1));
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload(std::get<1>(iov2));

      if (payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer2->Divide(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     1d plots of HCAL RespCorrs comparison between 2 IOVs
  **********************************************************/
  template <int ntags, IOVMultiplicity nIOVs>
  class HcalRespCorrsComparatorBase : public cond::payloadInspector::PlotImage<HcalRespCorrs, nIOVs, ntags> {
  public:
    HcalRespCorrsComparatorBase()
        : cond::payloadInspector::PlotImage<HcalRespCorrs, nIOVs, ntags>("HcalRespCorrs Comparison") {}

    using MetaData = std::tuple<cond::Time_t, cond::Hash>;

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<HcalRespCorrs> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<HcalRespCorrs> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      HcalRespCorrContainer* last_objContainer = new HcalRespCorrContainer(last_payload, std::get<0>(lastiov));
      HcalRespCorrContainer* first_objContainer = new HcalRespCorrContainer(first_payload, std::get<0>(firstiov));

      const auto& lastItems = last_objContainer->getAllItems();
      const auto& firstItems = first_objContainer->getAllItems();

      assert(lastItems.size() == firstItems.size());

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      canvas.Divide(3, 2);

      std::map<std::string, std::shared_ptr<TH1F>> first_plots;
      std::map<std::string, std::shared_ptr<TH1F>> last_plots;

      // Prepare the output vector
      std::vector<std::string> parts(lastItems.size());

      // Use std::transform to extract the strings
      std::transform(lastItems.begin(),
                     lastItems.end(),
                     parts.begin(),
                     [](const std::pair<std::string, std::vector<HcalRespCorr>>& pair) {
                       return pair.first;  // Extract the std::string part
                     });

      auto legend = TLegend(0.13, 0.85, 0.98, 0.95);
      legend.SetTextSize(0.032);

      unsigned int count{0};
      for (const auto& part : parts) {
        count++;
        first_plots[part] =
            std::make_shared<TH1F>(Form("f_%s_%s", part.c_str(), firstIOVsince.c_str()),
                                   Form("Response corrections [%s];correction factor;entries", part.c_str()),
                                   100,
                                   0.,
                                   3.);

        // Use std::find_if to find the matching pair
        auto it = std::find_if(
            firstItems.begin(),
            firstItems.end(),
            [&part](const std::pair<std::string, std::vector<HcalRespCorr>>& pair) { return pair.first == part; });

        // Check if we found the element
        if (it != firstItems.end()) {
          const std::vector<HcalRespCorr>& result = it->second;  // Retrieve the vector<HcalRespCorr>
          if (DEBUG) {
            std::cout << "Found vector<HcalRespCorr> for key: " << part << std::endl;
          }
          for (auto& item : result) {
            HcalDetId detId = HcalDetId(item.rawId());
            if (DEBUG) {
              int iphi = detId.iphi();
              int ieta = detId.ieta();
              int depth = detId.depth();

              std::cout << detId << " iphi" << iphi << " ieta: " << ieta << " depth:" << depth << std::endl;
            }
            if (detId != HcalDetId())
              first_plots[part]->Fill(first_objContainer->getValue(&item));
          }

          // You can now work with the result vector
        } else {
          std::cout << "Key not found: " << part << std::endl;
        }

        last_plots[part] =
            std::make_shared<TH1F>(Form("l_%s_%s", part.c_str(), lastIOVsince.c_str()),
                                   Form("Response Corrections [%s];correction factor;entries", part.c_str()),
                                   100,
                                   0,
                                   3.);

        // Use std::find_if to find the matching pair
        auto it2 = std::find_if(
            lastItems.begin(), lastItems.end(), [&part](const std::pair<std::string, std::vector<HcalRespCorr>>& pair) {
              return pair.first == part;
            });

        // Check if we found the element
        if (it2 != lastItems.end()) {
          const std::vector<HcalRespCorr>& result = it2->second;  // Retrieve the vector<HcalRespCorr>
          if (DEBUG) {
            std::cout << "Found vector<HcalRespCorr> for key: " << part << std::endl;
          }
          for (auto& item : result) {
            HcalDetId detId = HcalDetId(item.rawId());
            if (DEBUG) {
              int iphi = detId.iphi();
              int ieta = detId.ieta();
              int depth = detId.depth();

              std::cout << detId << " iphi" << iphi << " ieta: " << ieta << " depth:" << depth << std::endl;
            }
            if (detId != HcalDetId())
              last_plots[part]->Fill(last_objContainer->getValue(&item));
          }

          // You can now work with the result vector
        } else {
          std::cout << "Key not found: " << part << std::endl;
        }

        canvas.cd(count);

        canvas.cd(count)->SetTopMargin(0.05);
        canvas.cd(count)->SetLeftMargin(0.13);
        canvas.cd(count)->SetRightMargin(0.02);

        if (count == 1) {
          legend.AddEntry(first_plots[part].get(), (std::get<1>(firstiov)).c_str(), "L");
          legend.AddEntry(last_plots[part].get(), (std::get<1>(lastiov)).c_str(), "L");
        }

        const auto& extrema = getExtrema(first_plots[part].get(), last_plots[part].get());
        first_plots[part]->SetMaximum(1.1 * extrema.second);

        beautifyPlot(first_plots[part], kBlue);
        first_plots[part]->Draw();
        beautifyPlot(last_plots[part], kRed);
        last_plots[part]->Draw("same");

        // Add the first TPaveText box with mean and RMS at the top
        addStatisticsPaveText(first_plots[part].get(), "Resp Corr", 0.80);

        // Add the second TPaveText box with mean and RMS slightly lower
        addStatisticsPaveText(last_plots[part].get(), "Resp Corr", 0.70);

        legend.Draw("same");
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    bool DEBUG{false};

    void beautifyPlot(std::shared_ptr<TH1F> hist, int kColor) {
      hist->SetStats(kFALSE);
      hist->SetLineWidth(2);
      hist->SetLineColor(kColor);
      hist->GetXaxis()->CenterTitle(true);
      hist->GetYaxis()->CenterTitle(true);
      hist->GetXaxis()->SetTitleFont(42);
      hist->GetYaxis()->SetTitleFont(42);
      hist->GetXaxis()->SetTitleSize(0.05);
      hist->GetYaxis()->SetTitleSize(0.05);
      hist->GetXaxis()->SetTitleOffset(0.9);
      hist->GetYaxis()->SetTitleOffset(1.5);
      hist->GetXaxis()->SetLabelFont(42);
      hist->GetYaxis()->SetLabelFont(42);
      hist->GetYaxis()->SetLabelSize(.05);
      hist->GetXaxis()->SetLabelSize(.05);
    }

    std::pair<float, float> getExtrema(TH1* h1, TH1* h2) {
      float theMax(-9999.);
      float theMin(9999.);
      theMax = h1->GetMaximum() > h2->GetMaximum() ? h1->GetMaximum() : h2->GetMaximum();
      theMin = h1->GetMinimum() < h2->GetMaximum() ? h1->GetMinimum() : h2->GetMinimum();

      float add_min = theMin > 0. ? -0.05 : 0.05;
      float add_max = theMax > 0. ? 0.05 : -0.05;

      auto result = std::make_pair(theMin * (1 + add_min), theMax * (1 + add_max));
      return result;
    }

    // Function to add a TPaveText with mean and RMS at a specific vertical position
    void addStatisticsPaveText(TH1* hist, const std::string& label, double yPosition) {
      // Get the histogram's statistics
      double mean = hist->GetMean();
      double rms = hist->GetRMS();

      // Format mean and RMS to 4 significant digits
      std::ostringstream meanStream;
      meanStream << std::setprecision(4) << std::scientific << mean;
      std::string meanStr = meanStream.str();

      std::ostringstream rmsStream;
      rmsStream << std::setprecision(4) << std::scientific << rms;
      std::string rmsStr = rmsStream.str();

      // Calculate NDC positions
      double x1 = 0.55;
      double x2 = 0.95;
      double y1 = yPosition - 0.1;  // Height of 0.05 NDC units
      double y2 = yPosition;

      // Create a TPaveText for mean and RMS
      TPaveText* statsPave = new TPaveText(x1, y1, x2, y2, "NDC");
      statsPave->SetFillColor(0);  // Transparent background
      statsPave->SetBorderSize(1);
      statsPave->SetLineColor(hist->GetLineColor());
      statsPave->SetTextColor(hist->GetLineColor());
      statsPave->SetTextAlign(12);  // Align left and vertically centered

      // Add mean and RMS to the TPaveText
      statsPave->AddText((label + " Mean: " + meanStr).c_str());
      statsPave->AddText((label + " RMS: " + rmsStr).c_str());

      // Draw the TPaveText
      statsPave->Draw();
    }
  };

  using HcalRespCorrsComparatorSingleTag = HcalRespCorrsComparatorBase<1, MULTI_IOV>;
  using HcalRespCorrsComparatorTwoTags = HcalRespCorrsComparatorBase<2, SINGLE_IOV>;

  /**********************************************************
     2d plots of HCAL RespCorrs comparison between 2 IOVs
  **********************************************************/
  template <int ntags, IOVMultiplicity nIOVs>
  class HcalRespCorrsCorrelationBase : public cond::payloadInspector::PlotImage<HcalRespCorrs, nIOVs, ntags> {
  public:
    HcalRespCorrsCorrelationBase()
        : cond::payloadInspector::PlotImage<HcalRespCorrs, nIOVs, ntags>("HcalRespCorrs Comparison") {}

    using MetaData = std::tuple<cond::Time_t, cond::Hash>;

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<HcalRespCorrs> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<HcalRespCorrs> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      HcalRespCorrContainer* last_objContainer = new HcalRespCorrContainer(last_payload, std::get<0>(lastiov));
      HcalRespCorrContainer* first_objContainer = new HcalRespCorrContainer(first_payload, std::get<0>(firstiov));

      const auto& lastItems = last_objContainer->getAllItems();
      const auto& firstItems = first_objContainer->getAllItems();

      assert(lastItems.size() == firstItems.size());

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      canvas.Divide(3, 2);

      std::map<std::string, std::shared_ptr<TH2F>> plots;

      // Prepare the output vector
      std::vector<std::string> parts(lastItems.size());

      // Use std::transform to extract the strings
      std::transform(lastItems.begin(),
                     lastItems.end(),
                     parts.begin(),
                     [](const std::pair<std::string, std::vector<HcalRespCorr>>& pair) {
                       return pair.first;  // Extract the std::string part
                     });

      auto legend = TLegend(0.13, 0.85, 0.98, 0.95);
      legend.SetTextSize(0.032);

      unsigned int count{0};
      for (const auto& part : parts) {
        count++;
        plots[part] =
            std::make_shared<TH2F>(Form("%s_%s_vs_%s", part.c_str(), firstIOVsince.c_str(), lastIOVsince.c_str()),
                                   Form("%s;correction factor (#bf{#color[2]{%s}}, IOV #bf{#color[2]{%s}});correction "
                                        "factor (#bf{#color[4]{%s}}, IOV #bf{#color[4]{%s}})",
                                        part.c_str(),
                                        tagname1.c_str(),
                                        firstIOVsince.c_str(),
                                        tagname2.c_str(),
                                        lastIOVsince.c_str()),
                                   100,
                                   0.,
                                   3.,
                                   100,
                                   0.,
                                   3.);

        // Use std::find_if to find the matching pair
        auto it = std::find_if(
            firstItems.begin(),
            firstItems.end(),
            [&part](const std::pair<std::string, std::vector<HcalRespCorr>>& pair) { return pair.first == part; });

        // Use std::find_if to find the matching pair
        auto it2 = std::find_if(
            lastItems.begin(), lastItems.end(), [&part](const std::pair<std::string, std::vector<HcalRespCorr>>& pair) {
              return pair.first == part;
            });

        // Check if we found the element
        if (it != firstItems.end() && it2 != lastItems.end()) {
          const std::vector<HcalRespCorr>& result = it->second;    // Retrieve the vector<HcalRespCorr>
          const std::vector<HcalRespCorr>& result2 = it2->second;  // Retrieve the vector<HcalRespCorr>

          for (auto& item : result) {
            HcalDetId detId = HcalDetId(item.rawId());
            if (detId == HcalDetId()) {
              continue;
            }
            for (auto& item2 : result2) {
              HcalDetId detId2 = HcalDetId(item2.rawId());
              if (detId == detId2) {
                plots[part]->Fill(first_objContainer->getValue(&item), last_objContainer->getValue(&item2));
              }
            }
          }
        }

        canvas.cd(count);

        canvas.cd(count)->SetTopMargin(0.05);
        canvas.cd(count)->SetLeftMargin(0.13);
        canvas.cd(count)->SetRightMargin(0.02);

        beautifyPlot(plots[part]);
        plots[part]->Draw();
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    void beautifyPlot(std::shared_ptr<TH2F> hist) {
      hist->SetStats(kFALSE);
      hist->GetXaxis()->CenterTitle(true);
      hist->GetYaxis()->CenterTitle(true);
      hist->GetXaxis()->SetTitleFont(42);
      hist->GetYaxis()->SetTitleFont(42);
      hist->GetXaxis()->SetTitleSize(0.035);
      hist->GetYaxis()->SetTitleSize(0.035);
      hist->GetXaxis()->SetTitleOffset(1.55);
      hist->GetYaxis()->SetTitleOffset(1.55);
      hist->GetXaxis()->SetLabelFont(42);
      hist->GetYaxis()->SetLabelFont(42);
      hist->GetYaxis()->SetLabelSize(.05);
      hist->GetXaxis()->SetLabelSize(.05);
    }
  };

  using HcalRespCorrsCorrelationSingleTag = HcalRespCorrsCorrelationBase<1, MULTI_IOV>;
  using HcalRespCorrsCorrelationTwoTags = HcalRespCorrsCorrelationBase<2, SINGLE_IOV>;

  /******************************************
     2d plot of HCAL RespCorr of 1 IOV
  ******************************************/
  class HcalRespCorrsEtaPlotAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsEtaPlotAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     2d plot of HCAL RespCorrs difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsEtaRatioAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsEtaRatioAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload(std::get<1>(iov1));
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload(std::get<1>(iov2));

      if (payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer2->Divide(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };
  /******************************************
     2d plot of HCAL RespCorr of 1 IOV
  ******************************************/
  class HcalRespCorrsPhiPlotAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPhiPlotAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     2d plot of HCAL RespCorrs difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsPhiRatioAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPhiRatioAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload(std::get<1>(iov1));
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload(std::get<1>(iov2));

      if (payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer2->Divide(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };
  /******************************************
     2d plot of HCAL RespCorrs of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotHBHO : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotHBHO() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasHBHO()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     2d plot of HCAL RespCorr difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioHBHO : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsRatioHBHO() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload(std::get<1>(iov1));
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload(std::get<1>(iov2));

      if (payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer2->Divide(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasHBHO()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };
  /******************************************
     2d plot of HCAL RespCorr of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotHE : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotHE() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasHE()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     2d plot of HCAL RespCorr difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioHE : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsRatioHE() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload(std::get<1>(iov1));
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload(std::get<1>(iov2));

      if (payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer2->Divide(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasHE()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };
  /******************************************
     2d plot of HCAL RespCorr of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotHF : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotHF() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasHF()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };

  /**********************************************************
     2d plot of HCAL RespCorrRatios difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioHF : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsRatioHF() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload(std::get<1>(iov1));
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload(std::get<1>(iov2));

      if (payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer2->Divide(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasHF()->SaveAs(ImageName.c_str());
        return true;
      } else
        return false;
    }  // fill method
  };
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalRespCorrs) {
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsComparatorSingleTag);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsCorrelationSingleTag);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsCorrelationTwoTags);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsEtaPlotAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsEtaRatioAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPhiPlotAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPhiRatioAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotHBHO);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioHBHO);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotHE);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioHE);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotHF);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioHF);
}
