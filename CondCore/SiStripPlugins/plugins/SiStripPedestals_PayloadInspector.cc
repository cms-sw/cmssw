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
#include "SiStripCondObjectRepresent.h"

#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <boost/tokenizer.hpp>

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
  using namespace cond::payloadInspector;

  class SiStripPedestalContainer : public SiStripCondObjectRepresent::SiStripDataContainer<SiStripPedestals, float> {
  public:
    SiStripPedestalContainer(const std::shared_ptr<SiStripPedestals>& payload,
                             const SiStripPI::MetaData& metadata,
                             const std::string& tagName)
        : SiStripCondObjectRepresent::SiStripDataContainer<SiStripPedestals, float>(payload, metadata, tagName) {
      payloadType_ = "SiStripPedestals";
      setGranularity(SiStripCondObjectRepresent::PERSTRIP);
    }

    void storeAllValues() override {
      std::vector<uint32_t> detid;
      payload_->getDetIds(detid);

      for (const auto& d : detid) {
        SiStripPedestals::Range range = payload_->getRange(d);
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          // to be used to fill the histogram
          SiStripCondData_.fillByPushBack(d, payload_->getPed(it, range));
        }
      }
    }
  };

  class SiStripPedestalCompareByPartition : public PlotImage<SiStripPedestals, MULTI_IOV, 2> {
  public:
    SiStripPedestalCompareByPartition()
        : PlotImage<SiStripPedestals, MULTI_IOV, 2>("SiStrip Compare Pedestals By Partition") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      auto tag2iovs = PlotBase::getTag<1>().iovs;
      auto tagname2 = PlotBase::getTag<1>().name;
      SiStripPI::MetaData firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov = tag2iovs.front();

      std::shared_ptr<SiStripPedestals> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripPedestals> first_payload = fetchPayload(std::get<1>(firstiov));

      SiStripPedestalContainer* l_objContainer = new SiStripPedestalContainer(last_payload, lastiov, tagname1);
      SiStripPedestalContainer* f_objContainer = new SiStripPedestalContainer(first_payload, firstiov, tagname2);

      l_objContainer->compare(f_objContainer);

      //l_objContainer->printAll();

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      l_objContainer->fillByPartition(canvas, 300, 0.1, 300.);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }  // fill
  };

  class SiStripPedestalDiffByPartition : public PlotImage<SiStripPedestals, MULTI_IOV, 2> {
  public:
    SiStripPedestalDiffByPartition()
        : PlotImage<SiStripPedestals, MULTI_IOV, 2>("SiStrip Diff Pedestals By Partition") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      auto tag2iovs = PlotBase::getTag<1>().iovs;
      auto tagname2 = PlotBase::getTag<1>().name;
      SiStripPI::MetaData firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov = tag2iovs.front();

      std::shared_ptr<SiStripPedestals> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripPedestals> first_payload = fetchPayload(std::get<1>(firstiov));

      SiStripPedestalContainer* l_objContainer = new SiStripPedestalContainer(last_payload, lastiov, tagname1);
      SiStripPedestalContainer* f_objContainer = new SiStripPedestalContainer(first_payload, firstiov, tagname2);

      l_objContainer->subtract(f_objContainer);

      //l_objContainer->printAll();

      TCanvas canvas("Partition summary", "partition summary", 1400, 1000);
      l_objContainer->fillByPartition(canvas, 100, -30., 30.);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }  // fill
  };

  class SiStripPedestalCorrelationByPartition : public PlotImage<SiStripPedestals> {
  public:
    SiStripPedestalCorrelationByPartition()
        : PlotImage<SiStripPedestals>("SiStrip Pedestals Correlation By Partition") {
      setSingleIov(false);
    }

    bool fill(const std::vector<SiStripPI::MetaData>& iovs) override {
      std::vector<SiStripPI::MetaData> sorted_iovs = iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const& t1, auto const& t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<SiStripPedestals> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<SiStripPedestals> first_payload = fetchPayload(std::get<1>(firstiov));

      SiStripPedestalContainer* l_objContainer = new SiStripPedestalContainer(last_payload, lastiov, "");
      SiStripPedestalContainer* f_objContainer = new SiStripPedestalContainer(first_payload, firstiov, "");

      l_objContainer->compare(f_objContainer);

      TCanvas canvas("Partition summary", "partition summary", 1200, 1200);
      l_objContainer->fillCorrelationByPartition(canvas, 100, 0., 300.);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }  // fill
  };

  /************************************************
    test class
  *************************************************/

  class SiStripPedestalsTest : public Histogram1D<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalsTest()
        : Histogram1D<SiStripPedestals, SINGLE_IOV>("SiStrip Pedestals test", "SiStrip Pedestals test", 10, 0.0, 10.0),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
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

  class SiStripPedestalPerDetId : public PlotImage<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalPerDetId() : PlotImage<SiStripPedestals, SINGLE_IOV>("SiStrip Pedestal values per DetId") {
      PlotBase::addInputParam("DetIds");
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      std::vector<uint32_t> the_detids = {};

      auto paramValues = PlotBase::inputParamValues();
      auto ip = paramValues.find("DetIds");
      if (ip != paramValues.end()) {
        auto input = ip->second;
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
        boost::char_separator<char> sep{","};
        tokenizer tok{input, sep};
        for (const auto& t : tok) {
          the_detids.push_back(atoi(t.c_str()));
        }
      } else {
        edm::LogWarning("SiStripNoisePerDetId")
            << "\n WARNING!!!! \n The needed parameter DetIds has not been passed. Will use all Strip DetIds! \n\n";
        the_detids.push_back(0xFFFFFFFF);
      }

      size_t ndets = the_detids.size();
      std::vector<std::shared_ptr<TH1F>> hpedestal;
      std::vector<std::shared_ptr<TLegend>> legends;
      std::vector<unsigned int> v_nAPVs;
      std::vector<std::vector<std::shared_ptr<TLine>>> lines;
      hpedestal.reserve(ndets);
      legends.reserve(ndets);

      auto sides = getClosestFactors(the_detids.size());
      edm::LogPrint("SiStripPedestalPerDetId") << "Aspect ratio: " << sides.first << ":" << sides.second << std::endl;

      if (payload.get()) {
        //=========================
        TCanvas canvas("ByDetId", "ByDetId", sides.second * 800, sides.first * 600);
        canvas.Divide(sides.second, sides.first);
        const auto detInfo =
            SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
        for (const auto& the_detid : the_detids) {
          edm::LogPrint("SiStripNoisePerDetId") << "DetId:" << the_detid << std::endl;

          unsigned int nAPVs = detInfo.getNumberOfApvsAndStripLength(the_detid).first;
          if (nAPVs == 0)
            nAPVs = 6;
          v_nAPVs.push_back(nAPVs);

          auto histo = std::make_shared<TH1F>(
              Form("Pedestal profile %s", std::to_string(the_detid).c_str()),
              Form("SiStrip Pedestal profile for DetId: %s;Strip number;SiStrip Pedestal [ADC counts]",
                   std::to_string(the_detid).c_str()),
              sistrip::STRIPS_PER_APV * nAPVs,
              -0.5,
              (sistrip::STRIPS_PER_APV * nAPVs) - 0.5);

          histo->SetStats(false);
          histo->SetTitle("");

          if (the_detid != 0xFFFFFFFF) {
            fillHisto(payload, histo, the_detid);
          } else {
            auto allDetIds = detInfo.getAllDetIds();
            for (const auto& id : allDetIds) {
              fillHisto(payload, histo, id);
            }
          }

          SiStripPI::makeNicePlotStyle(histo.get());
          histo->GetYaxis()->SetTitleOffset(1.0);
          hpedestal.push_back(histo);
        }  // loop on the detids

        for (size_t index = 0; index < ndets; index++) {
          canvas.cd(index + 1);
          canvas.cd(index + 1)->SetBottomMargin(0.11);
          canvas.cd(index + 1)->SetTopMargin(0.06);
          canvas.cd(index + 1)->SetLeftMargin(0.10);
          canvas.cd(index + 1)->SetRightMargin(0.02);
          hpedestal.at(index)->Draw();
          hpedestal.at(index)->GetYaxis()->SetRangeUser(0, hpedestal.at(index)->GetMaximum() * 1.2);
          canvas.cd(index)->Update();

          std::vector<int> boundaries;
          for (size_t b = 0; b < v_nAPVs.at(index); b++) {
            boundaries.push_back(b * sistrip::STRIPS_PER_APV);
          }

          std::vector<std::shared_ptr<TLine>> linesVec;
          for (const auto& bound : boundaries) {
            auto line = std::make_shared<TLine>(hpedestal.at(index)->GetBinLowEdge(bound),
                                                canvas.cd(index + 1)->GetUymin(),
                                                hpedestal.at(index)->GetBinLowEdge(bound),
                                                canvas.cd(index + 1)->GetUymax());
            line->SetLineWidth(1);
            line->SetLineStyle(9);
            line->SetLineColor(2);
            linesVec.push_back(line);
          }
          lines.push_back(linesVec);

          for (const auto& line : lines.at(index)) {
            line->Draw("same");
          }

          canvas.cd(index + 1);

          auto ltx = TLatex();
          ltx.SetTextFont(62);
          ltx.SetTextSize(0.05);
          ltx.SetTextAlign(11);
          ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                           1 - gPad->GetTopMargin() + 0.01,
                           Form("SiStrip Pedestals profile for DetId %s", std::to_string(the_detids[index]).c_str()));

          legends.push_back(std::make_shared<TLegend>(0.45, 0.83, 0.95, 0.93));
          legends.at(index)->SetHeader(tagname.c_str(), "C");  // option "C" allows to center the header
          legends.at(index)->AddEntry(
              hpedestal.at(index).get(), ("IOV: " + std::to_string(std::get<0>(iov))).c_str(), "PL");
          legends.at(index)->SetTextSize(0.045);
          legends.at(index)->Draw("same");
        }

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }  // payload
      return true;
    }  // fill

  private:
    int nextPerfectSquare(int N) { return std::floor(sqrt(N)) + 1; }

    std::pair<int, int> getClosestFactors(int input) {
      if ((input % 2 != 0) && input > 1) {
        input += 1;
      }

      int testNum = (int)sqrt(input);
      while (input % testNum != 0) {
        testNum--;
      }
      return std::make_pair(testNum, input / testNum);
    }

    void fillHisto(const std::shared_ptr<SiStripPedestals> payload, std::shared_ptr<TH1F>& histo, uint32_t the_detid) {
      int nstrip = 0;
      SiStripPedestals::Range range = payload->getRange(the_detid);
      for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
        auto pedestal = payload->getPed(it, range);
        nstrip++;
        histo->AddBinContent(nstrip, pedestal);
      }  // end of loop on strips
    }
  };

  /************************************************
    1d histogram of SiStripPedestals of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripPedestalsValue : public Histogram1D<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalsValue()
        : Histogram1D<SiStripPedestals, SINGLE_IOV>(
              "SiStrip Pedestals values", "SiStrip Pedestals values", 300, 0.0, 300.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
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
  class SiStripPedestalsValuePerDetId : public Histogram1D<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalsValuePerDetId()
        : Histogram1D<SiStripPedestals, SINGLE_IOV>(
              "SiStrip Pedestal values per DetId", "SiStrip Pedestal values per DetId", 100, 0.0, 10.0) {
      PlotBase::addInputParam("DetId");
    }

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<SiStripPedestals> payload = Base::fetchPayload(std::get<1>(iov));
        unsigned int the_detid(0xFFFFFFFF);
        auto paramValues = PlotBase::inputParamValues();
        auto ip = paramValues.find("DetId");
        if (ip != paramValues.end()) {
          the_detid = std::stoul(ip->second);
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
  class SiStripPedestalDistribution : public PlotImage<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalDistribution() : PlotImage<SiStripPedestals, SINGLE_IOV>("SiStrip Pedestal values") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
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
              flush = (prev_det != 0 && prev_apv != istrip / sistrip::STRIPS_PER_APV);
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
          prev_apv = istrip / sistrip::STRIPS_PER_APV;
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

  template <SiStripPI::OpMode op_mode_, int ntags, IOVMultiplicity nIOVs>
  class SiStripPedestalDistributionComparisonBase : public PlotImage<SiStripPedestals, nIOVs, ntags> {
  public:
    SiStripPedestalDistributionComparisonBase()
        : PlotImage<SiStripPedestals, nIOVs, ntags>("SiStrip Pedestal values comparison") {}

    bool fill() override {
      TGaxis::SetExponentOffset(-0.1, 0.01, "y");  // X and Y offset for Y axis

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      SiStripPI::MetaData lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<SiStripPedestals> f_payload = this->fetchPayload(std::get<1>(firstiov));
      std::shared_ptr<SiStripPedestals> l_payload = this->fetchPayload(std::get<1>(lastiov));

      auto f_mon = std::unique_ptr<SiStripPI::Monitor1D>(new SiStripPI::Monitor1D(
          op_mode_,
          "f_Pedestal",
          Form(";#LTStrip Pedestal per %s#GT [ADC counts];n. %ss", opType(op_mode_).c_str(), opType(op_mode_).c_str()),
          300,
          0.,
          300.));

      auto l_mon = std::unique_ptr<SiStripPI::Monitor1D>(new SiStripPI::Monitor1D(
          op_mode_,
          "l_Pedestal",
          Form(";#LTStrip Pedestal per %s#GT [ADC counts];n. %ss", opType(op_mode_).c_str(), opType(op_mode_).c_str()),
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
              flush = (prev_det != 0 && prev_apv != istrip / sistrip::STRIPS_PER_APV);
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
          prev_apv = istrip / sistrip::STRIPS_PER_APV;
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
              flush = (prev_det != 0 && prev_apv != istrip / sistrip::STRIPS_PER_APV);
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
          prev_apv = istrip / sistrip::STRIPS_PER_APV;
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
      canvas.SetTopMargin(0.06);
      canvas.SetBottomMargin(0.10);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      float theMax = (h_first.GetMaximum() > h_last.GetMaximum()) ? h_first.GetMaximum() : h_last.GetMaximum();

      h_first.SetMaximum(theMax * 1.20);
      h_last.SetMaximum(theMax * 1.20);

      h_first.Draw();
      h_last.SetFillColorAlpha(kBlue, 0.15);
      h_last.Draw("same");

      TLegend legend = TLegend(0.13, 0.83, 0.95, 0.94);
      if (this->m_plotAnnotations.ntags == 2) {
        legend.SetHeader("#bf{Two Tags Comparison}", "C");  // option "C" allows to center the header
        legend.AddEntry(&h_first, (tagname1 + " : " + std::to_string(std::get<0>(firstiov))).c_str(), "F");
        legend.AddEntry(&h_last, (tagname2 + " : " + std::to_string(std::get<0>(lastiov))).c_str(), "F");
      } else {
        legend.SetHeader(("tag: #bf{" + tagname1 + "}").c_str(), "C");  // option "C" allows to center the header
        legend.AddEntry(&h_first, ("IOV since: " + std::to_string(std::get<0>(firstiov))).c_str(), "F");
        legend.AddEntry(&h_last, ("IOV since: " + std::to_string(std::get<0>(lastiov))).c_str(), "F");
      }
      legend.SetTextSize(0.025);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       Form("#LTSiStrip Pedestals#GT Comparison per %s", opType(op_mode_).c_str()));

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    std::string opType(SiStripPI::OpMode mode) {
      std::string types[3] = {"Strip", "APV", "Module"};
      return types[mode];
    }
  };

  template <SiStripPI::OpMode op_mode_>
  using SiStripPedestalDistributionComparisonSingleTag =
      SiStripPedestalDistributionComparisonBase<op_mode_, 1, MULTI_IOV>;

  template <SiStripPI::OpMode op_mode_>
  using SiStripPedestalDistributionComparisonTwoTags =
      SiStripPedestalDistributionComparisonBase<op_mode_, 2, SINGLE_IOV>;

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

  // inherit from one of the predefined plot class PlotImage
  class SiStripZeroPedestalsFraction_TrackerMap : public PlotImage<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripZeroPedestalsFraction_TrackerMap()
        : PlotImage<SiStripPedestals, SINGLE_IOV>("Tracker Map of Zero SiStripPedestals fraction per module") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      const auto detInfo =
          SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());

      std::string titleMap =
          "Tracker Map of Zero SiStrip Pedestals fraction per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripPedestals");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<uint32_t, int> zeropeds_per_detid;

      for (const auto& d : detid) {
        SiStripPedestals::Range range = payload->getRange(d);
        for (int it = 0; it < (range.second - range.first) * 8 / 10; ++it) {
          auto ped = payload->getPed(it, range);
          if (ped == 0.) {
            zeropeds_per_detid[d] += 1;
          }
        }  // end of loop on strips
        float fraction =
            zeropeds_per_detid[d] / (sistrip::STRIPS_PER_APV * detInfo.getNumberOfApvsAndStripLength(d).first);
        if (fraction > 0.) {
          tmap->fill(d, fraction);
          std::cout << "detid: " << d << " (n. APVs=" << detInfo.getNumberOfApvsAndStripLength(d).first << ") has "
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
  class SiStripPedestalsTrackerMap : public PlotImage<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalsTrackerMap()
        : PlotImage<SiStripPedestals, SINGLE_IOV>("Tracker Map of SiStripPedestals " + estimatorType(est) +
                                                  " per module") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiStripPedestals> payload = fetchPayload(std::get<1>(iov));

      std::string titleMap =
          "Tracker Map of SiStrip Pedestals " + estimatorType(est) + " per module (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("SiStripPedestals");
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
  class SiStripPedestalsByRegion : public PlotImage<SiStripPedestals, SINGLE_IOV> {
  public:
    SiStripPedestalsByRegion()
        : PlotImage<SiStripPedestals, SINGLE_IOV>("SiStrip Pedestals " + estimatorType(est) + " by Region"),
          m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
              edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
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
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalCompareByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalDiffByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripPedestalCorrelationByPartition);
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
