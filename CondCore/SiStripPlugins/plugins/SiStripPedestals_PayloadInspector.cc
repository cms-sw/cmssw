/*!
  \file SiStripPedestals_PayloadInspector
  \Payload Inspector Plugin for SiStrip Noises
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/09/22 11:02:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TGaxis.h"
#include "TPaveStats.h"

namespace {

  /************************************************
    test class
  *************************************************/

  class SiStripPedestalsTest : public cond::payloadInspector::Histogram1D<SiStripPedestals> {
  public:
    SiStripPedestalsTest()
        : cond::payloadInspector::Histogram1D<SiStripPedestals>(
              "SiStrip Pedestals test", "SiStrip Pedestals test", 10, 0.0, 10.0),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto const& iov : iovs) {
        std::shared_ptr<SiStripPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          fillWithValue(1.);

          std::stringstream ss;
          ss << "Summary of strips pedestals:" << std::endl;

          //payload->printDebug(ss);
          payload->printSummary(ss, &m_trackerTopo);

          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          // for (const auto & d : detid) {
          //   int nstrip=0;
          //   SiStripPedestals::Range range=payload->getRange(d);
          //   for( int it=0; it < (range.second-range.first)*8/10; ++it ){
          //     auto ped = payload->getPed(it,range);
          //     nstrip++;
          //     ss << "DetId="<< d << " Strip=" << nstrip <<": "<< ped << std::endl;
          //   } // end of loop on strips
          // } // end of loop on detIds

          std::cout << ss.str() << std::endl;

        }  // payload
      }    // iovs
      return true;
    }  // fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    SiStrip Pedestals Profile of 1 IOV for one selected DetId
  *************************************************/

  class SiStripPedestalPerDetId : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
    SiStripPedestalPerDetId()
        : cond::payloadInspector::PlotImage<SiStripPedestals>("SiStrip Pedestal values per DetId") {
      cond::payloadInspector::PlotBase::addInputParam("DetId");
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();

      unsigned int the_detid(0xFFFFFFFF);

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("DetId");
      if (ip != paramValues.end()) {
        the_detid = boost::lexical_cast<unsigned int>(ip->second);
      }

      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));
      if (payload.get()) {
        //=========================
        TCanvas canvas("ByDetId", "ByDetId", 1200, 1000);

        edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
        SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
        unsigned int nAPVs = reader->getNumberOfApvsAndStripLength(the_detid).first;

        auto hnoise = std::unique_ptr<TH1F>(
            new TH1F("Pedestal profile",
                     Form("SiStrip Pedestal profile for DetId: %s;Strip number;SiStrip Pedestal [ADC counts]",
                          std::to_string(the_detid).c_str()),
                     128 * nAPVs,
                     -0.5,
                     (128 * nAPVs) - 0.5));
        hnoise->SetStats(false);

        std::vector<uint32_t> detid;
        payload->getDetIds(detid);

        int nstrip = 0;
        SiStripPedestals::Range range = payload->getRange(the_detid);
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          auto noise = payload->getPed(it, range);
          nstrip++;
          hnoise->SetBinContent(nstrip, noise);
        }  // end of loop on strips

        canvas.cd();
        canvas.SetBottomMargin(0.11);
        canvas.SetTopMargin(0.07);
        canvas.SetLeftMargin(0.13);
        canvas.SetRightMargin(0.05);
        hnoise->Draw();
        hnoise->GetYaxis()->SetRangeUser(0, hnoise->GetMaximum() * 1.2);
        //hnoise->Draw("Psame");
        canvas.Update();

        std::vector<int> boundaries;
        for (size_t b = 0; b < nAPVs; b++)
          boundaries.push_back(b * 128);

        TLine l[nAPVs];
        unsigned int i = 0;
        for (const auto& line : boundaries) {
          l[i] = TLine(hnoise->GetBinLowEdge(line), canvas.GetUymin(), hnoise->GetBinLowEdge(line), canvas.GetUymax());
          l[i].SetLineWidth(1);
          l[i].SetLineStyle(9);
          l[i].SetLineColor(2);
          l[i].Draw("same");
          i++;
        }

        TLegend legend = TLegend(0.52, 0.82, 0.95, 0.93);
        legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
        legend.AddEntry(hnoise.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
        legend.SetTextSize(0.025);
        legend.Draw("same");

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }  // payload
      return true;
    }  // fill
  };

  /************************************************
    1d histogram of SiStripPedestals of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripPedestalsValue : public cond::payloadInspector::Histogram1D<SiStripPedestals> {
  public:
    SiStripPedestalsValue()
        : cond::payloadInspector::Histogram1D<SiStripPedestals>(
              "SiStrip Pedestals values", "SiStrip Pedestals values", 300, 0.0, 300.0) {
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto const& iov : iovs) {
        std::shared_ptr<SiStripPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<uint32_t> detid;
          payload->getDetIds(detid);

          for (const auto& d : detid) {
            SiStripPedestals::Range range = payload->getRange(d);
            for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
              auto ped = payload->getPed(it, range);
              //to be used to fill the histogram
              fillWithValue(ped);
            }  // loop over APVs
          }    // loop over detIds
        }      // payload
      }        // iovs
      return true;
    }  // fill
  };

  /************************************************
    1d histogram of SiStripPedestals of 1 IOV per Detid
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripPedestalsValuePerDetId : public cond::payloadInspector::Histogram1D<SiStripPedestals> {
  public:
    SiStripPedestalsValuePerDetId()
        : cond::payloadInspector::Histogram1D<SiStripPedestals>(
              "SiStrip Pedestal values per DetId", "SiStrip Pedestal values per DetId", 100, 0.0, 10.0) {
      cond::payloadInspector::PlotBase::addInputParam("DetId");
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      for (auto const& iov : iovs) {
        std::shared_ptr<SiStripPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        unsigned int the_detid(0xFFFFFFFF);
        auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
        auto ip = paramValues.find("DetId");
        if (ip != paramValues.end()) {
          the_detid = boost::lexical_cast<unsigned int>(ip->second);
        }

        if (payload.get()) {
          SiStripPedestals::Range range = payload->getRange(the_detid);
          for (int it = 0; it < (range.second - range.first) * 8 / 9; ++it) {
            auto noise = payload->getPed(it, range);
            //to be used to fill the histogram
            fillWithValue(noise);
          }  // loop over APVs
        }    // payload
      }      // iovs
      return true;
    }  // fill
  };

  /************************************************
    templated 1d histogram of SiStripPedestals of 1 IOV
  *************************************************/

  // inherit from one of the predefined plot class: PlotImage
  template <SiStripPI::OpMode op_mode_>
  class SiStripPedestalDistribution : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
    SiStripPedestalDistribution() : cond::payloadInspector::PlotImage<SiStripPedestals>("SiStrip Pedestal values") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();

      TGaxis::SetMaxDigits(3);
      gStyle->SetOptStat("emr");

      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      auto mon1D = std::unique_ptr<SiStripPI::Monitor1D>(new SiStripPI::Monitor1D(
          op_mode_,
          "Pedestal",
          Form("#LT Pedestal #GT per %s for IOV [%s];#LTStrip Pedestal per %s#GT [ADC counts];n. %ss",
               opType(op_mode_).c_str(),
               std::to_string(std::get<0>(iov)).c_str(),
               opType(op_mode_).c_str(),
               opType(op_mode_).c_str()),
          300,
          0.,
          300.0));

      unsigned int prev_det = 0, prev_apv = 0;
      SiStripPI::Entry epedestal;

      std::vector<uint32_t> detids;
      payload->getDetIds(detids);

      // loop on payload
      for (const auto& d : detids) {
        SiStripPedestals::Range range = payload->getRange(d);

        unsigned int istrip = 0;

        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          auto pedestal = payload->getPed(it, range);
          bool flush = false;
          switch (op_mode_) {
            case (SiStripPI::APV_BASED):
              flush = (prev_det != 0 && prev_apv != istrip / 128);
              break;
            case (SiStripPI::MODULE_BASED):
              flush = (prev_det != 0 && prev_det != d);
              break;
            case (SiStripPI::STRIP_BASED):
              flush = (istrip != 0);
              break;
          }

          if (flush) {
            mon1D->Fill(prev_apv, prev_det, epedestal.mean());
            epedestal.reset();
          }

          epedestal.add(std::min<float>(pedestal, 300.));
          prev_apv = istrip / 128;
          istrip++;
        }
        prev_det = d;
      }

      //=========================
      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.cd();
      canvas.SetBottomMargin(0.11);
      canvas.SetTopMargin(0.07);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      auto hist = mon1D->getHist();
      SiStripPI::makeNicePlotStyle(&hist);
      hist.SetStats(kTRUE);
      hist.SetFillColorAlpha(kRed, 0.35);
      hist.Draw();

      canvas.Update();

      TPaveStats* st = (TPaveStats*)hist.GetListOfFunctions()->FindObject("stats");
      st->SetLineColor(kRed);
      st->SetTextColor(kRed);
      st->SetX1NDC(.75);
      st->SetX2NDC(.95);
      st->SetY1NDC(.83);
      st->SetY2NDC(.93);

      TLegend legend = TLegend(0.13, 0.83, 0.43, 0.93);
      legend.SetHeader(Form("SiStrip Pedestal values per %s", opType(op_mode_).c_str()),
                       "C");  // option "C" allows to center the header
      legend.AddEntry(&hist, ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    std::string opType(SiStripPI::OpMode mode) {
      std::string types[3] = {"Strip", "APV", "Module"};
      return types[mode];
    }
  };

  typedef SiStripPedestalDistribution<SiStripPI::STRIP_BASED> SiStripPedestalValuePerStrip;
  typedef SiStripPedestalDistribution<SiStripPI::APV_BASED> SiStripPedestalValuePerAPV;
  typedef SiStripPedestalDistribution<SiStripPI::MODULE_BASED> SiStripPedestalValuePerModule;

  /************************************************
  template 1d histogram comparison of SiStripPedestals of 1 IOV
  *************************************************/

  // inherit from one of the predefined plot class: PlotImage

  template <SiStripPI::OpMode op_mode_>
  class SiStripPedestalDistributionComparisonBase : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
    SiStripPedestalDistributionComparisonBase()
        : cond::payloadInspector::PlotImage<SiStripPedestals>("SiStrip Pedestal values comparison") {}

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      std::vector<std::tuple<cond::Time_t, cond::Hash> > sorted_iovs = iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const& t1, auto const& t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<SiStripPedestals> f_payload = fetchPayload(std::get<1>(firstiov));
      std::shared_ptr<SiStripPedestals> l_payload = fetchPayload(std::get<1>(lastiov));

      auto f_mon = std::unique_ptr<SiStripPI::Monitor1D>(new SiStripPI::Monitor1D(
          op_mode_,
          "f_Pedestal",
          Form("#LT Strip Pedestal #GT per %s for IOV [%s,%s];#LTStrip Pedestal per %s#GT [ADC counts];n. %ss",
               opType(op_mode_).c_str(),
               std::to_string(std::get<0>(firstiov)).c_str(),
               std::to_string(std::get<0>(lastiov)).c_str(),
               opType(op_mode_).c_str(),
               opType(op_mode_).c_str()),
          300,
          0.,
          300.));

      auto l_mon = std::unique_ptr<SiStripPI::Monitor1D>(new SiStripPI::Monitor1D(
          op_mode_,
          "l_Pedestal",
          Form("#LT Strip Pedestal #GT per %s for IOV [%s,%s];#LTStrip Pedestal per %s#GT [ADC counts];n. %ss",
               opType(op_mode_).c_str(),
               std::to_string(std::get<0>(lastiov)).c_str(),
               std::to_string(std::get<0>(lastiov)).c_str(),
               opType(op_mode_).c_str(),
               opType(op_mode_).c_str()),
          300,
          0.,
          300.));

      unsigned int prev_det = 0, prev_apv = 0;
      SiStripPI::Entry epedestal;

      std::vector<uint32_t> f_detid;
      f_payload->getDetIds(f_detid);

      // loop on first payload
      for (const auto& d : f_detid) {
        SiStripPedestals::Range range = f_payload->getRange(d);

        unsigned int istrip = 0;
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          float pedestal = f_payload->getPed(it, range);
          //to be used to fill the histogram

          bool flush = false;
          switch (op_mode_) {
            case (SiStripPI::APV_BASED):
              flush = (prev_det != 0 && prev_apv != istrip / 128);
              break;
            case (SiStripPI::MODULE_BASED):
              flush = (prev_det != 0 && prev_det != d);
              break;
            case (SiStripPI::STRIP_BASED):
              flush = (istrip != 0);
              break;
          }

          if (flush) {
            f_mon->Fill(prev_apv, prev_det, epedestal.mean());
            epedestal.reset();
          }
          epedestal.add(std::min<float>(pedestal, 300.));
          prev_apv = istrip / 128;
          istrip++;
        }
        prev_det = d;
      }

      prev_det = 0, prev_apv = 0;
      epedestal.reset();

      std::vector<uint32_t> l_detid;
      l_payload->getDetIds(l_detid);

      // loop on first payload
      for (const auto& d : l_detid) {
        SiStripPedestals::Range range = l_payload->getRange(d);

        unsigned int istrip = 0;
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          float pedestal = l_payload->getPed(it, range);

          bool flush = false;
          switch (op_mode_) {
            case (SiStripPI::APV_BASED):
              flush = (prev_det != 0 && prev_apv != istrip / 128);
              break;
            case (SiStripPI::MODULE_BASED):
              flush = (prev_det != 0 && prev_det != d);
              break;
            case (SiStripPI::STRIP_BASED):
              flush = (istrip != 0);
              break;
          }

          if (flush) {
            l_mon->Fill(prev_apv, prev_det, epedestal.mean());
            epedestal.reset();
          }

          epedestal.add(std::min<float>(pedestal, 300.));
          prev_apv = istrip / 128;
          istrip++;
        }
        prev_det = d;
      }

      auto h_first = f_mon->getHist();
      h_first.SetStats(kFALSE);
      auto h_last = l_mon->getHist();
      h_last.SetStats(kFALSE);

      SiStripPI::makeNicePlotStyle(&h_first);
      SiStripPI::makeNicePlotStyle(&h_last);

      h_first.GetYaxis()->CenterTitle(true);
      h_last.GetYaxis()->CenterTitle(true);

      h_first.GetXaxis()->CenterTitle(true);
      h_last.GetXaxis()->CenterTitle(true);

      h_first.SetLineWidth(2);
      h_last.SetLineWidth(2);

      h_first.SetLineColor(kBlack);
      h_last.SetLineColor(kBlue);

      //=========================
      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.cd();
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      float theMax = (h_first.GetMaximum() > h_last.GetMaximum()) ? h_first.GetMaximum() : h_last.GetMaximum();

      h_first.SetMaximum(theMax * 1.30);
      h_last.SetMaximum(theMax * 1.30);

      h_first.Draw();
      h_last.Draw("same");

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader("SiStrip Pedestal comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(&h_first, ("IOV: " + std::to_string(std::get<0>(firstiov))).c_str(), "F");
      legend.AddEntry(&h_last, ("IOV: " + std::to_string(std::get<0>(lastiov))).c_str(), "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    std::string opType(SiStripPI::OpMode mode) {
      std::string types[3] = {"Strip", "APV", "Module"};
      return types[mode];
    }
  };

  template <SiStripPI::OpMode op_mode_>
  class SiStripPedestalDistributionComparisonSingleTag : public SiStripPedestalDistributionComparisonBase<op_mode_> {
  public:
    SiStripPedestalDistributionComparisonSingleTag() : SiStripPedestalDistributionComparisonBase<op_mode_>() {
      this->setSingleIov(false);
    }
  };

  template <SiStripPI::OpMode op_mode_>
  class SiStripPedestalDistributionComparisonTwoTags : public SiStripPedestalDistributionComparisonBase<op_mode_> {
  public:
    SiStripPedestalDistributionComparisonTwoTags() : SiStripPedestalDistributionComparisonBase<op_mode_>() {
      this->setTwoTags(true);
    }
  };

  typedef SiStripPedestalDistributionComparisonSingleTag<SiStripPI::STRIP_BASED>
      SiStripPedestalValueComparisonPerStripSingleTag;
  typedef SiStripPedestalDistributionComparisonSingleTag<SiStripPI::APV_BASED>
      SiStripPedestalValueComparisonPerAPVSingleTag;
  typedef SiStripPedestalDistributionComparisonSingleTag<SiStripPI::MODULE_BASED>
      SiStripPedestalValueComparisonPerModuleSingleTag;

  typedef SiStripPedestalDistributionComparisonTwoTags<SiStripPI::STRIP_BASED>
      SiStripPedestalValueComparisonPerStripTwoTags;
  typedef SiStripPedestalDistributionComparisonTwoTags<SiStripPI::APV_BASED> SiStripPedestalValueComparisonPerAPVTwoTags;
  typedef SiStripPedestalDistributionComparisonTwoTags<SiStripPI::MODULE_BASED>
      SiStripPedestalValueComparisonPerModuleTwoTags;

  /************************************************
    1d histogram of fraction of Zero SiStripPedestals of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripZeroPedestalsFraction_TrackerMap : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
    SiStripZeroPedestalsFraction_TrackerMap()
        : cond::payloadInspector::PlotImage<SiStripPedestals>(
              "Tracker Map of Zero SiStripPedestals fraction per module") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::string titleMap =
          "Tracker Map of Zero SiStrip Pedestals fraction per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripPedestals"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<uint32_t, int> zeropeds_per_detid;

      for (const auto& d : detid) {
        int nstrips = 0;
        SiStripPedestals::Range range = payload->getRange(d);
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          nstrips++;
          auto ped = payload->getPed(it, range);
          if (ped == 0.) {
            zeropeds_per_detid[d] += 1;
          }
        }  // end of loop on strips
        float fraction = zeropeds_per_detid[d] / (128. * reader->getNumberOfApvsAndStripLength(d).first);
        if (fraction > 0.) {
          tmap->fill(d, fraction);
          std::cout << "detid: " << d << " (n. APVs=" << reader->getNumberOfApvsAndStripLength(d).first << ") has "
                    << std::setw(4) << zeropeds_per_detid[d]
                    << " zero-pedestals strips (i.e. a fraction:" << std::setprecision(5) << fraction << ")"
                    << std::endl;
        }
      }  // end of loop on detids

      std::string fileName(m_imageFileName);
      tmap->save(true, 0., 0., fileName);

      return true;
    }
  };

  /************************************************
    Tracker Map of SiStrip Pedestals
  *************************************************/

  template <SiStripPI::estimator est>
  class SiStripPedestalsTrackerMap : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
    SiStripPedestalsTrackerMap()
        : cond::payloadInspector::PlotImage<SiStripPedestals>("Tracker Map of SiStripPedestals " + estimatorType(est) +
                                                              " per module") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      std::string titleMap =
          "Tracker Map of SiStrip Pedestals " + estimatorType(est) + " per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripPedestals"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<unsigned int, float> info_per_detid;

      for (const auto& d : detid) {
        int nstrips = 0;
        double mean(0.), rms(0.), min(10000.), max(0.);
        SiStripPedestals::Range range = payload->getRange(d);
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          nstrips++;
          auto ped = payload->getPed(it, range);
          mean += ped;
          rms += (ped * ped);
          if (ped < min)
            min = ped;
          if (ped > max)
            max = ped;
        }  // end of loop on strips

        mean /= nstrips;
        if ((rms / nstrips - mean * mean) > 0.) {
          rms = sqrt(rms / nstrips - mean * mean);
        } else {
          rms = 0.;
        }

        switch (est) {
          case SiStripPI::min:
            info_per_detid[d] = min;
            break;
          case SiStripPI::max:
            info_per_detid[d] = max;
            break;
          case SiStripPI::mean:
            info_per_detid[d] = mean;
            break;
          case SiStripPI::rms:
            info_per_detid[d] = rms;
            break;
          default:
            edm::LogWarning("LogicError") << "Unknown estimator: " << est;
            break;
        }
      }  // end of loop on detids

      for (const auto& d : detid) {
        tmap->fill(d, info_per_detid[d]);
      }

      std::string fileName(m_imageFileName);
      tmap->save(true, 0., 0., fileName);

      return true;
    }
  };

  typedef SiStripPedestalsTrackerMap<SiStripPI::min> SiStripPedestalsMin_TrackerMap;
  typedef SiStripPedestalsTrackerMap<SiStripPI::max> SiStripPedestalsMax_TrackerMap;
  typedef SiStripPedestalsTrackerMap<SiStripPI::mean> SiStripPedestalsMean_TrackerMap;
  typedef SiStripPedestalsTrackerMap<SiStripPI::rms> SiStripPedestalsRMS_TrackerMap;

  /************************************************
    Tracker Map of SiStrip Pedestals Summaries
  *************************************************/

  template <SiStripPI::estimator est>
  class SiStripPedestalsByRegion : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
    SiStripPedestalsByRegion()
        : cond::payloadInspector::PlotImage<SiStripPedestals>("SiStrip Pedestals " + estimatorType(est) + " by Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      SiStripDetSummary summaryPedestals{&m_trackerTopo};
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto& d : detid) {
        int nstrips = 0;
        double mean(0.), rms(0.), min(10000.), max(0.);
        SiStripPedestals::Range range = payload->getRange(d);
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          nstrips++;
          auto ped = payload->getPed(it, range);
          mean += ped;
          rms += (ped * ped);
          if (ped < min)
            min = ped;
          if (ped > max)
            max = ped;
        }  // end of loop on strips

        mean /= nstrips;
        if ((rms / nstrips - mean * mean) > 0.) {
          rms = sqrt(rms / nstrips - mean * mean);
        } else {
          rms = 0.;
        }

        switch (est) {
          case SiStripPI::min:
            summaryPedestals.add(d, min);
            break;
          case SiStripPI::max:
            summaryPedestals.add(d, max);
            break;
          case SiStripPI::mean:
            summaryPedestals.add(d, mean);
            break;
          case SiStripPI::rms:
            summaryPedestals.add(d, rms);
            break;
          default:
            edm::LogWarning("LogicError") << "Unknown estimator: " << est;
            break;
        }
      }  // loop on the detIds

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryPedestals.getCounts();
      //=========================

      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.cd();
      auto h1 = std::unique_ptr<TH1F>(new TH1F(
          "byRegion",
          Form("Average by partition of %s SiStrip Pedestals per module;;average SiStrip Pedestals %s [ADC counts]",
               estimatorType(est).c_str(),
               estimatorType(est).c_str()),
          map.size(),
          0.,
          map.size()));
      h1->SetStats(false);
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.17);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin = 0;

      std::string detector;
      std::string currentDetector;

      for (const auto& element : map) {
        iBin++;
        int count = element.second.count;
        double mean = (element.second.mean) / count;
        double rms = (element.second.rms) / count - mean * mean;

        if (rms <= 0)
          rms = 0;
        else
          rms = sqrt(rms);

        if (currentDetector.empty())
          currentDetector = "TIB";

        switch ((element.first) / 1000) {
          case 1:
            detector = "TIB";
            break;
          case 2:
            detector = "TOB";
            break;
          case 3:
            detector = "TEC";
            break;
          case 4:
            detector = "TID";
            break;
        }

        h1->SetBinContent(iBin, mean);
        h1->GetXaxis()->SetBinLabel(iBin, SiStripPI::regionType(element.first).second);
        h1->GetXaxis()->LabelsOption("v");

        if (detector != currentDetector) {
          boundaries.push_back(iBin);
          currentDetector = detector;
        }
      }

      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->SetMaximum(h1->GetMaximum() * 1.1);
      h1->Draw("HIST");
      h1->Draw("Psame");

      canvas.Update();

      TLine l[boundaries.size()];
      unsigned int i = 0;
      for (const auto& line : boundaries) {
        l[i] = TLine(h1->GetBinLowEdge(line), canvas.GetUymin(), h1->GetBinLowEdge(line), canvas.GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLegend legend = TLegend(0.52, 0.82, 0.95, 0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(), "C");  // option "C" allows to center the header
      legend.AddEntry(h1.get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    TrackerTopology m_trackerTopo;
  };

  typedef SiStripPedestalsByRegion<SiStripPI::mean> SiStripPedestalsMeanByRegion;
  typedef SiStripPedestalsByRegion<SiStripPI::min> SiStripPedestalsMinByRegion;
  typedef SiStripPedestalsByRegion<SiStripPI::max> SiStripPedestalsMaxByRegion;
  typedef SiStripPedestalsByRegion<SiStripPI::rms> SiStripPedestalsRMSByRegion;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiStripPedestals) {
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalPerDetId);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsValuePerDetId);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValuePerStrip);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValuePerAPV);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValuePerModule);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValueComparisonPerStripSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValueComparisonPerAPVSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValueComparisonPerModuleSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValueComparisonPerStripTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValueComparisonPerAPVTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalValueComparisonPerModuleTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiStripZeroPedestalsFraction_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMin_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMax_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMean_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsRMS_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMeanByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMinByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMaxByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsRMSByRegion);
}
