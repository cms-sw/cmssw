#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondFormats/Common/interface/BasicPayload.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include <memory>
#include <sstream>

#include "TH2D.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"

namespace {

  class BasicPayload_data0 : public cond::payloadInspector::HistoryPlot<cond::BasicPayload, float> {
  public:
    BasicPayload_data0() : cond::payloadInspector::HistoryPlot<cond::BasicPayload, float>("Example Trend", "data0") {}
    ~BasicPayload_data0() override = default;
    float getFromPayload(cond::BasicPayload& payload) override { return payload.m_data0; }
  };

  class BasicPayload_data0_withInput : public cond::payloadInspector::HistoryPlot<cond::BasicPayload, float> {
  public:
    BasicPayload_data0_withInput()
        : cond::payloadInspector::HistoryPlot<cond::BasicPayload, float>("Example Trend", "data0") {
      cond::payloadInspector::PlotBase::addInputParam("Offset");
      cond::payloadInspector::PlotBase::addInputParam("Factor");
      cond::payloadInspector::PlotBase::addInputParam("Scale");
    }
    ~BasicPayload_data0_withInput() override = default;
    float getFromPayload(cond::BasicPayload& payload) override {
      float v = payload.m_data0;
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Factor");
      if (ip != paramValues.end()) {
        v = v * boost::lexical_cast<float>(ip->second);
      }
      ip = paramValues.find("Offset");
      if (ip != paramValues.end()) {
        v = v + boost::lexical_cast<float>(ip->second);
      }
      ip = paramValues.find("Scale");
      if (ip != paramValues.end()) {
        v = v * boost::lexical_cast<float>(ip->second);
      }
      return v;
    }
  };

  class BasicPayload_data1 : public cond::payloadInspector::RunHistoryPlot<cond::BasicPayload, float> {
  public:
    BasicPayload_data1()
        : cond::payloadInspector::RunHistoryPlot<cond::BasicPayload, float>("Example Run-based Trend", "data0") {}
    ~BasicPayload_data1() override = default;
    float getFromPayload(cond::BasicPayload& payload) override { return payload.m_data0; }
  };

  class BasicPayload_data2 : public cond::payloadInspector::TimeHistoryPlot<cond::BasicPayload, float> {
  public:
    BasicPayload_data2()
        : cond::payloadInspector::TimeHistoryPlot<cond::BasicPayload, float>("Example Time-based Trend", "data0") {}
    ~BasicPayload_data2() override = default;

    float getFromPayload(cond::BasicPayload& payload) override { return payload.m_data0; }
  };

  class BasicPayload_data3 : public cond::payloadInspector::ScatterPlot<cond::BasicPayload, float, float> {
  public:
    BasicPayload_data3()
        : cond::payloadInspector::ScatterPlot<cond::BasicPayload, float, float>("Example Scatter", "data0", "data1") {}
    ~BasicPayload_data3() override = default;

    std::tuple<float, float> getFromPayload(cond::BasicPayload& payload) override {
      return std::make_tuple(payload.m_data0, payload.m_data1);
    }
  };

  class BasicPayload_data4 : public cond::payloadInspector::Histogram1D<cond::BasicPayload> {
  public:
    BasicPayload_data4() : cond::payloadInspector::Histogram1D<cond::BasicPayload>("Example Histo1d", "x", 10, 0, 10) {
      Base::setSingleIov(true);
    }
    ~BasicPayload_data4() override = default;

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto iov : iovs) {
        std::shared_ptr<cond::BasicPayload> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          for (size_t j = 0; j < 100; j++) {
            fillWithValue(j, payload->m_vec[j]);
          }
        }
      }
      return true;
    }
  };

  class BasicPayload_data5 : public cond::payloadInspector::Histogram2D<cond::BasicPayload> {
  public:
    BasicPayload_data5()
        : cond::payloadInspector::Histogram2D<cond::BasicPayload>("Example Histo2d", "x", 10, 0, 10, "y", 10, 0, 10) {
      Base::setSingleIov(true);
    }
    ~BasicPayload_data5() override = default;

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto iov : iovs) {
        std::shared_ptr<cond::BasicPayload> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          for (size_t i = 0; i < 10; i++)
            for (size_t j = 0; j < 10; j++) {
              fillWithValue(j, i, payload->m_vec[i * 10 + j]);
            }
        }
      }
      return true;
    }
  };

  class BasicPayload_data6 : public cond::payloadInspector::PlotImage<cond::BasicPayload> {
  public:
    BasicPayload_data6() : cond::payloadInspector::PlotImage<cond::BasicPayload>("Example delivery picture") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<cond::BasicPayload> payload = fetchPayload(std::get<1>(iov));

      double xmax(100.), ymax(100.);

      TH2D h2D("h2D", "Example", 100, 0., xmax, 100, 0., ymax);

      if (payload.get()) {
        if (payload->m_vec.size() == 10000) {
          for (size_t i = 0; i < 100; i++)
            for (size_t j = 0; j < 100; j++) {
              h2D.Fill(i, j, payload->m_vec[i * 100 + j]);
            }
          h2D.SetStats(false);
        }
      }

      TCanvas c("c", "", 10, 10, 900, 500);
      c.cd();
      c.SetLogz();
      h2D.SetNdivisions(18, "X");
      h2D.GetXaxis()->SetTickLength(0.00);
      h2D.GetYaxis()->SetTickLength(0.00);
      h2D.GetXaxis()->SetTitle("iphi");
      h2D.GetYaxis()->SetTitle("ieta");
      h2D.Draw("col");

      //======= drawing lines ========
      ///// this is quite specific to the line style they need

      TLine l;
      l.SetLineStyle(2);
      l.DrawLine(0., ymax / 2., xmax, ymax / 2.);
      for (int m = 0; m < int(xmax); m += 10) {
        l.DrawLine(m, 0., m, 100.);
      }

      c.RedrawAxis();

      //========== writing text in the canvas==============
      //// This is again quite specific part. I just tried to emulate what is there in DQM for EB.

      TLatex Tl;
      TLatex Tll;
      Tl.SetTextAlign(23);
      Tl.SetTextSize(0.04);

      Tll.SetTextAlign(23);
      Tll.SetTextSize(0.04);

      int j = 0;
      for (int i = 1; i <= 10; i++) {
        std::string s = "+" + std::to_string(i);
        char const* pchar = s.c_str();
        j += 10;
        Tl.DrawLatex(j - 5, int(ymax) / 1.33, pchar);
      }

      int z = 0;
      for (int g = -10; g < 0; g++) {
        std::string ss = std::to_string(g);
        char const* pchar1 = ss.c_str();
        z += 10;
        Tll.DrawLatex(z - 5, int(ymax) / 4, pchar1);
      }
      //=========================

      std::string fileName(m_imageFileName);
      c.SaveAs(fileName.c_str());

      return true;
    }
  };

  class BasicPayload_data7 : public cond::payloadInspector::PlotImage<cond::BasicPayload> {
  public:
    BasicPayload_data7() : cond::payloadInspector::PlotImage<cond::BasicPayload>("Example delivery picture") {
      setTwoTags(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov0 = iovs.front();
      auto iov1 = iovs.back();
      std::shared_ptr<cond::BasicPayload> payload0 = fetchPayload(std::get<1>(iov0));
      std::shared_ptr<cond::BasicPayload> payload1 = fetchPayload(std::get<1>(iov1));

      double xmax(100.), ymax(100.);

      TH2D h2D("h2D", "Example", 100, 0., xmax, 100, 0., ymax);

      if (payload0.get() && payload1.get()) {
        if (payload0->m_vec.size() == 10000 && payload1->m_vec.size() == 10000) {
          for (size_t i = 0; i < 100; i++)
            for (size_t j = 0; j < 100; j++) {
              auto diff = abs(payload0->m_vec[i * 100 + j] - payload1->m_vec[i * 100 + j]);
              h2D.Fill(i, j, diff);
            }
          h2D.SetStats(false);
        }
      }

      TCanvas c("c", "", 20, 20, 900, 500);
      c.cd();
      c.SetLogz();
      h2D.SetNdivisions(18, "X");
      h2D.GetXaxis()->SetTickLength(0.00);
      h2D.GetYaxis()->SetTickLength(0.00);
      h2D.GetXaxis()->SetTitle("iphi");
      h2D.GetYaxis()->SetTitle("ieta");
      h2D.Draw("col");

      //======= drawing lines ========
      ///// this is quite specific to the line style they need

      TLine l;
      l.SetLineStyle(2);
      l.DrawLine(0., ymax / 2., xmax, ymax / 2.);
      for (int m = 0; m < int(xmax); m += 10) {
        l.DrawLine(m, 0., m, 100.);
      }

      c.RedrawAxis();

      //========== writing text in the canvas==============
      //// This is again quite specific part. I just tried to emulate what is there in DQM for EB.

      TLatex Tl;
      TLatex Tll;
      Tl.SetTextAlign(23);
      Tl.SetTextSize(0.04);

      Tll.SetTextAlign(23);
      Tll.SetTextSize(0.04);

      int j = 0;
      for (int i = 1; i <= 10; i++) {
        std::string s = "+" + std::to_string(i);
        char const* pchar = s.c_str();
        j += 10;
        Tl.DrawLatex(j - 5, int(ymax) / 1.33, pchar);
      }

      int z = 0;
      for (int g = -10; g < 0; g++) {
        std::string ss = std::to_string(g);
        char const* pchar1 = ss.c_str();
        z += 10;
        Tll.DrawLatex(z - 5, int(ymax) / 4, pchar1);
      }
      //=========================

      std::string fileName(m_imageFileName);
      c.SaveAs(fileName.c_str());

      return true;
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(BasicPayload) {
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data0);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data0_withInput);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data1);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data2);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data3);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data4);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data5);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data6);
  PAYLOAD_INSPECTOR_CLASS(BasicPayload_data7);
}
