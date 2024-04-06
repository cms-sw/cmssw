/*!
  \file TrackerAlignmentErrorExtended_PayloadInspector
  \Payload Inspector Plugin for Tracker Alignment 
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/07/10 10:59:24 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/Alignment/interface/Alignments.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

//#include "CondFormats/Alignment/interface/Definitions.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// needed for mapping
#include "CondCore/AlignmentPlugins/interface/AlignmentPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"

#include <boost/range/adaptor/indexed.hpp>
#include <iomanip>  // std::setprecision
#include <iostream>
#include <memory>
#include <sstream>

// include ROOT
#include "TH2F.h"
#include "TGaxis.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TMarker.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  // M.M. 2017/09/29
  // Hardcoded Tracker Global Position Record
  // Without accessing the ES, it is not possible to access to the GPR with the PI technology,
  // so this needs to be hardcoded.
  // Anyway it is not likely to change until a new Tracker is installed.
  // Details at:
  // - https://indico.cern.ch/event/238026/contributions/513928/attachments/400000/556192/mm_TkAlMeeting_28_03_2013.pdf
  // - https://twiki.cern.ch/twiki/bin/view/CMS/TkAlignmentPixelPosition

  const std::map<AlignmentPI::coordinate, float> hardcodeGPR = {
      {AlignmentPI::t_x, -9.00e-02}, {AlignmentPI::t_y, -1.10e-01}, {AlignmentPI::t_z, -1.70e-01}};

  //*******************************************/
  // Size of the movement over all partitions,
  // one at a time
  //******************************************//

  enum RegionCategory { ALL = 0, INNER = 1, OUTER = 2 };

  template <int ntags, IOVMultiplicity nIOVs, RegionCategory cat>
  class TrackerAlignmentCompareAll : public PlotImage<Alignments, nIOVs, ntags> {
  public:
    TrackerAlignmentCompareAll()
        : PlotImage<Alignments, nIOVs, ntags>("comparison of all coordinates between two geometries") {}

    bool fill() override {
      TGaxis::SetExponentOffset(-0.12, 0.01, "y");  // Y offset

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<AlignTransform> ref_ali = first_payload->m_align;
      std::vector<AlignTransform> target_ali = last_payload->m_align;

      const bool ph2 = (ref_ali.size() > AlignmentPI::phase1size);

      // check that the geomtery is a tracker one
      const char *path_toTopologyXML = nullptr;
      if (ph2) {
        if (AlignmentPI::isReorderedTFPXTEPX(ref_ali) && AlignmentPI::isReorderedTFPXTEPX(target_ali)) {
          edm::LogPrint("TrackerAlignment_PayloadInspector")
              << "Both reference and target alignments are reordered. Using the trackerParameters for the Reordered "
                 "TFPX,TEPX.";
          path_toTopologyXML = "Geometry/TrackerCommonData/data/PhaseII/TFPXTEPXReordered/trackerParameters.xml";
        } else if (!AlignmentPI::isReorderedTFPXTEPX(ref_ali) && !AlignmentPI::isReorderedTFPXTEPX(target_ali)) {
          edm::LogPrint("TrackerAlignment_PayloadInspector")
              << "Neither reference nor target alignments are reordered. Using the standard trackerParameters.";
          path_toTopologyXML = "Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml";
        } else {
          if (cat == RegionCategory::ALL || cat == RegionCategory::INNER) {
            // Emit warning and exit false if alignments are mismatched
            edm::LogWarning("TrackerAlignment_PayloadInspector")
                << "Mismatched alignments detected. One is reordered while the other is not. Unable to proceed.";
            return false;
          } else {
            edm::LogWarning("TrackerAlignment_PayloadInspector")
                << "Mismatched inner tracks alignments detected. One is reordered while the other is not. Ignoring as "
                   "OT only comparison requested.";
            path_toTopologyXML = "Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml";
          }
        }
      } else {
        path_toTopologyXML = (ref_ali.size() == AlignmentPI::phase0size)
                                 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                 : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      }

      // Use remove_if along with a lambda expression to remove elements based on the condition (subid > 2)
      if (cat != RegionCategory::ALL) {
        ref_ali.erase(std::remove_if(ref_ali.begin(),
                                     ref_ali.end(),
                                     [](const AlignTransform &transform) {
                                       int subid = DetId(transform.rawId()).subdetId();
                                       return (cat == RegionCategory::INNER) ? (subid > 2) : (subid <= 2);
                                     }),
                      ref_ali.end());

        target_ali.erase(std::remove_if(target_ali.begin(),
                                        target_ali.end(),
                                        [](const AlignTransform &transform) {
                                          int subid = DetId(transform.rawId()).subdetId();
                                          return (cat == RegionCategory::INNER) ? (subid > 2) : (subid <= 2);
                                        }),
                         target_ali.end());
      }
      TCanvas canvas("Alignment Comparison", "Alignment Comparison", 2000, 1200);
      canvas.Divide(3, 2);

      if (ref_ali.size() != target_ali.size()) {
        edm::LogError("TrackerAlignment_PayloadInspector")
            << "the size of the reference alignment (" << ref_ali.size()
            << ") is different from the one of the target (" << target_ali.size()
            << ")! You are probably trying to compare different underlying geometries. Exiting";
        return false;
      }

      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      for (const auto &ali : ref_ali) {
        auto mydetid = ali.rawId();
        if (DetId(mydetid).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignment_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << DetId(mydetid).rawId() << " (" << DetId(mydetid).det()
              << ") is different from " << DetId::Tracker << " (is DoubleSide: " << tTopo.tidIsDoubleSide(mydetid)
              << "); subdetId " << DetId(mydetid).subdetId() << " - terminating ";
          return false;
        }
      }

      const std::vector<AlignmentPI::coordinate> coords = {AlignmentPI::t_x,
                                                           AlignmentPI::t_y,
                                                           AlignmentPI::t_z,
                                                           AlignmentPI::rot_alpha,
                                                           AlignmentPI::rot_beta,
                                                           AlignmentPI::rot_gamma};

      std::unordered_map<AlignmentPI::coordinate, std::unique_ptr<TH1F>> diffs;

      // generate the map of histograms
      for (const auto &coord : coords) {
        auto s_coord = AlignmentPI::getStringFromCoordinate(coord);
        std::string unit =
            (coord == AlignmentPI::t_x || coord == AlignmentPI::t_y || coord == AlignmentPI::t_z) ? "[#mum]" : "[mrad]";

        diffs[coord] = std::make_unique<TH1F>(Form("comparison_%s", s_coord.c_str()),
                                              Form(";Detector Id index; #Delta%s %s", s_coord.c_str(), unit.c_str()),
                                              ref_ali.size(),
                                              -0.5,
                                              ref_ali.size() - 0.5);
      }

      // fill all the histograms together
      std::map<int, AlignmentPI::partitions> boundaries;
      if (cat < RegionCategory::OUTER) {
        boundaries.insert({0, AlignmentPI::BPix});  // always start with BPix, not filled in the loop
      }
      AlignmentPI::fillComparisonHistograms(boundaries, ref_ali, target_ali, diffs);

      unsigned int subpad{1};
      TLegend legend = TLegend(0.17, 0.84, 0.95, 0.94);
      legend.SetTextSize(0.023);
      for (const auto &coord : coords) {
        canvas.cd(subpad);
        canvas.cd(subpad)->SetTopMargin(0.06);
        canvas.cd(subpad)->SetLeftMargin(0.17);
        canvas.cd(subpad)->SetRightMargin(0.05);
        canvas.cd(subpad)->SetBottomMargin(0.15);
        AlignmentPI::makeNicePlotStyle(diffs[coord].get(), kBlack);
        auto max = diffs[coord]->GetMaximum();
        auto min = diffs[coord]->GetMinimum();
        auto range = std::abs(max) > std::abs(min) ? std::abs(max) : std::abs(min);
        if (range == 0.f)
          range = 0.1;
        //auto newMax = (max > 0.) ? max*1.2 : max*0.8;

        diffs[coord]->GetYaxis()->SetRangeUser(-range * 1.5, range * 1.5);
        diffs[coord]->GetYaxis()->SetTitleOffset(1.5);
        diffs[coord]->SetMarkerStyle(20);
        diffs[coord]->SetMarkerSize(0.5);
        diffs[coord]->Draw("P");

        if (subpad == 1) { /* fill the legend only at the first pass */
          if (this->m_plotAnnotations.ntags == 2) {
            legend.SetHeader("#bf{Two Tags Comparison}", "C");  // option "C" allows to center the header
            legend.AddEntry(
                diffs[coord].get(),
                ("#splitline{" + tagname1 + " : " + firstIOVsince + "}{" + tagname2 + " : " + lastIOVsince + "}")
                    .c_str(),
                "PL");
          } else {
            legend.SetHeader(("tag: #bf{" + tagname1 + "}").c_str(), "C");  // option "C" allows to center the header
            legend.AddEntry(diffs[coord].get(),
                            ("#splitline{IOV since: " + firstIOVsince + "}{IOV since: " + lastIOVsince + "}").c_str(),
                            "PL");
          }
        }
        subpad++;
      }

      canvas.Update();
      canvas.cd();
      canvas.Modified();

      bool doOnlyPixel = (cat == RegionCategory::INNER);

      TLine l[6][boundaries.size()];
      TLatex tSubdet[6];
      for (unsigned int i = 0; i < 6; i++) {
        tSubdet[i].SetTextColor(kRed);
        tSubdet[i].SetNDC();
        tSubdet[i].SetTextAlign(21);
        tSubdet[i].SetTextSize(doOnlyPixel ? 0.05 : 0.03);
        tSubdet[i].SetTextAngle(90);
      }

      subpad = 0;
      for (const auto &coord : coords) {
        auto s_coord = AlignmentPI::getStringFromCoordinate(coord);
        canvas.cd(subpad + 1);
        for (const auto &line : boundaries | boost::adaptors::indexed(0)) {
          const auto &index = line.index();
          const auto value = line.value();
          l[subpad][index] = TLine(diffs[coord]->GetBinLowEdge(value.first),
                                   canvas.cd(subpad + 1)->GetUymin(),
                                   diffs[coord]->GetBinLowEdge(value.first),
                                   canvas.cd(subpad + 1)->GetUymax() * 0.84);
          l[subpad][index].SetLineWidth(1);
          l[subpad][index].SetLineStyle(9);
          l[subpad][index].SetLineColor(2);
          l[subpad][index].Draw("same");
        }

        for (const auto &elem : boundaries | boost::adaptors::indexed(0)) {
          const auto &lm = canvas.cd(subpad + 1)->GetLeftMargin();
          const auto &rm = 1 - canvas.cd(subpad + 1)->GetRightMargin();
          const auto &frac = float(elem.value().first) / ref_ali.size();

          LogDebug("TrackerAlignmentCompareAll")
              << __PRETTY_FUNCTION__ << " left margin:  " << lm << " right margin: " << rm << " fraction: " << frac;

          float theX_ = lm + (rm - lm) * frac + ((elem.index() > 0 || doOnlyPixel) ? 0.025 : 0.01);

          tSubdet[subpad].DrawLatex(
              theX_, 0.23, Form("%s", AlignmentPI::getStringFromPart(elem.value().second, /*is phase2?*/ ph2).c_str()));
        }

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.042);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(canvas.cd(subpad + 1)->GetLeftMargin(),
                         1 - canvas.cd(subpad + 1)->GetTopMargin() + 0.01,
                         ("Tracker Alignment Compare : #color[4]{" + s_coord + "}").c_str());
        legend.Draw("same");
        subpad++;
      }  // loop on the coordinates

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      //canvas.SaveAs("out.root");

      return true;
    }
  };

  typedef TrackerAlignmentCompareAll<1, MULTI_IOV, RegionCategory::ALL> TrackerAlignmentComparatorSingleTag;
  typedef TrackerAlignmentCompareAll<2, SINGLE_IOV, RegionCategory::ALL> TrackerAlignmentComparatorTwoTags;

  typedef TrackerAlignmentCompareAll<1, MULTI_IOV, RegionCategory::INNER> PixelAlignmentComparatorSingleTag;
  typedef TrackerAlignmentCompareAll<2, SINGLE_IOV, RegionCategory::INNER> PixelAlignmentComparatorTwoTags;

  typedef TrackerAlignmentCompareAll<1, MULTI_IOV, RegionCategory::OUTER> OTAlignmentComparatorSingleTag;
  typedef TrackerAlignmentCompareAll<2, SINGLE_IOV, RegionCategory::OUTER> OTAlignmentComparatorTwoTags;

  //*******************************************/
  // Size of the movement over all partitions,
  // one at a time (in cylindrical coordinates)
  //******************************************//

  template <int ntags, IOVMultiplicity nIOVs, RegionCategory cat>
  class TrackerAlignmentCompareCylindricalBase : public PlotImage<Alignments, nIOVs, ntags> {
    enum coordinate {
      t_r = 1,
      t_phi = 2,
      t_z = 3,
    };

  public:
    TrackerAlignmentCompareCylindricalBase()
        : PlotImage<Alignments, nIOVs, ntags>("comparison of cylindrical coordinates between two geometries") {}

    bool fill() override {
      TGaxis::SetExponentOffset(-0.12, 0.01, "y");  // Y offset

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<AlignTransform> ref_ali = first_payload->m_align;
      std::vector<AlignTransform> target_ali = last_payload->m_align;

      TCanvas canvas("Alignment Comparison", "", 2000, 600);
      canvas.Divide(3, 1);

      const bool ph2 = (ref_ali.size() > AlignmentPI::phase1size);

      const std::vector<coordinate> coords = {t_r, t_phi, t_z};
      std::unordered_map<coordinate, std::unique_ptr<TH1F>> diffs;

      // Use remove_if along with a lambda expression to remove elements based on the condition (subid > 2)
      if (cat != RegionCategory::ALL) {
        ref_ali.erase(std::remove_if(ref_ali.begin(),
                                     ref_ali.end(),
                                     [](const AlignTransform &transform) {
                                       int subid = DetId(transform.rawId()).subdetId();
                                       return (cat == RegionCategory::INNER) ? (subid > 2) : (subid <= 2);
                                     }),
                      ref_ali.end());

        target_ali.erase(std::remove_if(target_ali.begin(),
                                        target_ali.end(),
                                        [](const AlignTransform &transform) {
                                          int subid = DetId(transform.rawId()).subdetId();
                                          return (cat == RegionCategory::INNER) ? (subid > 2) : (subid <= 2);
                                        }),
                         target_ali.end());
      }

      auto h_deltaR = std::make_unique<TH1F>(
          "deltaR", Form(";Detector Id index; #DeltaR [#mum]"), ref_ali.size(), -0.5, ref_ali.size() - 0.5);
      auto h_deltaPhi = std::make_unique<TH1F>(
          "deltaPhi", Form(";Detector Id index; #Delta#phi [mrad]"), ref_ali.size(), -0.5, ref_ali.size() - 0.5);
      auto h_deltaZ = std::make_unique<TH1F>(
          "deltaZ", Form(";Detector Id index; #DeltaZ [#mum]"), ref_ali.size(), -0.5, ref_ali.size() - 0.5);

      std::map<int, AlignmentPI::partitions> boundaries;
      if (cat < RegionCategory::OUTER) {
        boundaries.insert({0, AlignmentPI::BPix});  // always start with BPix, not filled in the loop
      }

      int counter = 0; /* start the counter */
      AlignmentPI::partitions currentPart = AlignmentPI::BPix;
      for (unsigned int i = 0; i < ref_ali.size(); i++) {
        if (ref_ali[i].rawId() == target_ali[i].rawId()) {
          counter++;
          int subid = DetId(ref_ali[i].rawId()).subdetId();
          auto thePart = static_cast<AlignmentPI::partitions>(subid);

          if (thePart != currentPart) {
            currentPart = thePart;
            boundaries.insert({counter, thePart});
          }

          const auto &deltaTrans = target_ali[i].translation() - ref_ali[i].translation();
          double dPhi = target_ali[i].translation().phi() - ref_ali[i].translation().phi();
          if (dPhi > M_PI) {
            dPhi -= 2.0 * M_PI;
          }
          if (dPhi < -M_PI) {
            dPhi += 2.0 * M_PI;
          }

          h_deltaR->SetBinContent(i + 1, deltaTrans.perp() * AlignmentPI::cmToUm);
          h_deltaPhi->SetBinContent(i + 1, dPhi * AlignmentPI::tomRad);
          h_deltaZ->SetBinContent(i + 1, deltaTrans.z() * AlignmentPI::cmToUm);
        }
      }

      diffs[t_r] = std::move(h_deltaR);
      diffs[t_phi] = std::move(h_deltaPhi);
      diffs[t_z] = std::move(h_deltaZ);

      unsigned int subpad{1};
      TLegend legend = TLegend(0.17, 0.84, 0.95, 0.94);
      legend.SetTextSize(0.023);
      for (const auto &coord : coords) {
        canvas.cd(subpad);
        canvas.cd(subpad)->SetTopMargin(0.06);
        canvas.cd(subpad)->SetLeftMargin(0.17);
        canvas.cd(subpad)->SetRightMargin(0.05);
        canvas.cd(subpad)->SetBottomMargin(0.15);
        AlignmentPI::makeNicePlotStyle(diffs[coord].get(), kBlack);
        auto max = diffs[coord]->GetMaximum();
        auto min = diffs[coord]->GetMinimum();
        auto range = std::abs(max) > std::abs(min) ? std::abs(max) : std::abs(min);
        if (range == 0.f) {
          range = 0.1;
        }

        // no negative radii differnces
        if (coord != t_r) {
          diffs[coord]->GetYaxis()->SetRangeUser(-range * 1.5, range * 1.5);
        } else {
          diffs[coord]->GetYaxis()->SetRangeUser(0., range * 1.5);
        }

        diffs[coord]->GetYaxis()->SetTitleOffset(1.5);
        diffs[coord]->SetMarkerStyle(20);
        diffs[coord]->SetMarkerSize(0.5);
        diffs[coord]->Draw("P");

        if (subpad == 1) { /* fill the legend only at the first pass */
          if (this->m_plotAnnotations.ntags == 2) {
            legend.SetHeader("#bf{Two Tags Comparison}", "C");  // option "C" allows to center the header
            legend.AddEntry(
                diffs[coord].get(),
                ("#splitline{" + tagname1 + " : " + firstIOVsince + "}{" + tagname2 + " : " + lastIOVsince + "}")
                    .c_str(),
                "PL");
          } else {
            legend.SetHeader(("tag: #bf{" + tagname1 + "}").c_str(), "C");  // option "C" allows to center the header
            legend.AddEntry(diffs[coord].get(),
                            ("#splitline{IOV since: " + firstIOVsince + "}{IOV since: " + lastIOVsince + "}").c_str(),
                            "PL");
          }
        }
        subpad++;
      }

      canvas.Update();
      canvas.cd();
      canvas.Modified();

      bool doOnlyPixel = (cat == RegionCategory::INNER);

      TLine l[6][boundaries.size()];
      TLatex tSubdet[6];
      for (unsigned int i = 0; i < 6; i++) {
        tSubdet[i].SetTextColor(kRed);
        tSubdet[i].SetNDC();
        tSubdet[i].SetTextAlign(21);
        tSubdet[i].SetTextSize(doOnlyPixel ? 0.05 : 0.03);
        tSubdet[i].SetTextAngle(90);
      }

      subpad = 0;
      for (const auto &coord : coords) {
        auto s_coord = getStringFromCoordinate(coord);
        canvas.cd(subpad + 1);
        for (const auto &line : boundaries | boost::adaptors::indexed(0)) {
          const auto &index = line.index();
          const auto value = line.value();
          l[subpad][index] = TLine(diffs[coord]->GetBinLowEdge(value.first),
                                   canvas.cd(subpad + 1)->GetUymin(),
                                   diffs[coord]->GetBinLowEdge(value.first),
                                   canvas.cd(subpad + 1)->GetUymax() * 0.84);
          l[subpad][index].SetLineWidth(1);
          l[subpad][index].SetLineStyle(9);
          l[subpad][index].SetLineColor(2);
          l[subpad][index].Draw("same");
        }

        for (const auto &elem : boundaries | boost::adaptors::indexed(0)) {
          const auto &lm = canvas.cd(subpad + 1)->GetLeftMargin();
          const auto &rm = 1 - canvas.cd(subpad + 1)->GetRightMargin();
          const auto &frac = float(elem.value().first) / ref_ali.size();

          LogDebug("TrackerAlignmentCompareCylindricalBase")
              << __PRETTY_FUNCTION__ << " left margin:  " << lm << " right margin: " << rm << " fraction: " << frac;

          float theX_ = lm + (rm - lm) * frac + ((elem.index() > 0 || doOnlyPixel) ? 0.025 : 0.01);

          tSubdet[subpad].DrawLatex(
              theX_, 0.23, Form("%s", AlignmentPI::getStringFromPart(elem.value().second, /*is phase2?*/ ph2).c_str()));
        }

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.042);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(canvas.cd(subpad + 1)->GetLeftMargin(),
                         1 - canvas.cd(subpad + 1)->GetTopMargin() + 0.01,
                         ("Tracker Alignment Compare : #color[4]{" + s_coord + "}").c_str());
        legend.Draw("same");
        subpad++;
      }  // loop on the coordinates

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    /*--------------------------------------------------------------------*/
    inline std::string getStringFromCoordinate(coordinate coord)
    /*--------------------------------------------------------------------*/
    {
      switch (coord) {
        case t_r:
          return "r-translation";
        case t_phi:
          return "#phi-rotation";
        case t_z:
          return "z-translation";
        default:
          return "should never be here!";
      }
    }
  };

  typedef TrackerAlignmentCompareCylindricalBase<1, MULTI_IOV, RegionCategory::ALL>
      TrackerAlignmentCompareRPhiZSingleTag;
  typedef TrackerAlignmentCompareCylindricalBase<2, SINGLE_IOV, RegionCategory::ALL> TrackerAlignmentCompareRPhiZTwoTags;

  typedef TrackerAlignmentCompareCylindricalBase<1, MULTI_IOV, RegionCategory::INNER>
      PixelAlignmentCompareRPhiZSingleTag;
  typedef TrackerAlignmentCompareCylindricalBase<2, SINGLE_IOV, RegionCategory::INNER> PixelAlignmentCompareRPhiZTwoTags;

  typedef TrackerAlignmentCompareCylindricalBase<1, MULTI_IOV, RegionCategory::OUTER> OTAlignmentCompareRPhiZSingleTag;
  typedef TrackerAlignmentCompareCylindricalBase<2, SINGLE_IOV, RegionCategory::OUTER> OTAlignmentCompareRPhiZTwoTags;

  //*******************************************//
  // Size of the movement over all partitions,
  // one coordinate (x,y,z,...) at a time
  //******************************************//

  template <AlignmentPI::coordinate coord, int ntags, IOVMultiplicity nIOVs>
  class TrackerAlignmentComparatorBase : public PlotImage<Alignments, nIOVs, ntags> {
  public:
    TrackerAlignmentComparatorBase()
        : PlotImage<Alignments, nIOVs, ntags>("comparison of " + AlignmentPI::getStringFromCoordinate(coord) +
                                              " coordinate between two geometries") {}

    bool fill() override {
      TGaxis::SetExponentOffset(-0.12, 0.01, "y");  // Y offset

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<AlignTransform> ref_ali = first_payload->m_align;
      std::vector<AlignTransform> target_ali = last_payload->m_align;

      TCanvas canvas("Alignment Comparison", "Alignment Comparison", 1200, 1200);

      if (ref_ali.size() != target_ali.size()) {
        edm::LogError("TrackerAlignment_PayloadInspector")
            << "the size of the reference alignment (" << ref_ali.size()
            << ") is different from the one of the target (" << target_ali.size()
            << ")! You are probably trying to compare different underlying geometries. Exiting";
        return false;
      }

      const bool ph2 = (ref_ali.size() > AlignmentPI::phase1size);

      // check that the geomtery is a tracker one
      const char *path_toTopologyXML = nullptr;
      if (ph2) {
        if (AlignmentPI::isReorderedTFPXTEPX(ref_ali) && AlignmentPI::isReorderedTFPXTEPX(target_ali)) {
          edm::LogPrint("TrackerAlignment_PayloadInspector")
              << "Both reference and target alignments are reordered. Using the trackerParameters for the Reordered "
                 "TFPX,TEPX.";
          path_toTopologyXML = "Geometry/TrackerCommonData/data/PhaseII/TFPXTEPXReordered/trackerParameters.xml";
        } else if (!AlignmentPI::isReorderedTFPXTEPX(ref_ali) && !AlignmentPI::isReorderedTFPXTEPX(target_ali)) {
          edm::LogPrint("TrackerAlignment_PayloadInspector")
              << "Neither reference nor target alignments are reordered. Using the standard trackerParameters.";
          path_toTopologyXML = "Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml";
        } else {
          // Emit warning and exit false if alignments are mismatched
          edm::LogWarning("TrackerAlignment_PayloadInspector")
              << "Mismatched alignments detected. One is reordered while the other is not. Unable to proceed.";
          return false;
        }
      } else {
        path_toTopologyXML = (ref_ali.size() == AlignmentPI::phase0size)
                                 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                 : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      }

      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      for (const auto &ali : ref_ali) {
        auto mydetid = ali.rawId();
        if (DetId(mydetid).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignment_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << DetId(mydetid).rawId() << " (" << DetId(mydetid).det()
              << ") is different from " << DetId::Tracker << " (is DoubleSide: " << tTopo.tidIsDoubleSide(mydetid)
              << "); subdetId " << DetId(mydetid).subdetId() << " - terminating ";
          return false;
        }
      }

      auto s_coord = AlignmentPI::getStringFromCoordinate(coord);
      std::string unit =
          (coord == AlignmentPI::t_x || coord == AlignmentPI::t_y || coord == AlignmentPI::t_z) ? "[#mum]" : "[mrad]";

      //std::unique_ptr<TH1F> compare = std::unique_ptr<TH1F>(new TH1F("comparison",Form("Comparison of %s;DetId index; #Delta%s %s",s_coord.c_str(),s_coord.c_str(),unit.c_str()),ref_ali.size(),-0.5,ref_ali.size()-0.5));
      std::unique_ptr<TH1F> compare =
          std::make_unique<TH1F>("comparison",
                                 Form(";Detector Id index; #Delta%s %s", s_coord.c_str(), unit.c_str()),
                                 ref_ali.size(),
                                 -0.5,
                                 ref_ali.size() - 0.5);

      // fill the histograms
      std::map<int, AlignmentPI::partitions> boundaries;
      boundaries.insert({0, AlignmentPI::BPix});  // always start with BPix, not filled in the loop
      AlignmentPI::fillComparisonHistogram(coord, boundaries, ref_ali, target_ali, compare);

      canvas.cd();

      canvas.SetTopMargin(0.06);
      canvas.SetLeftMargin(0.17);
      canvas.SetRightMargin(0.05);
      canvas.SetBottomMargin(0.15);
      AlignmentPI::makeNicePlotStyle(compare.get(), kBlack);
      auto max = compare->GetMaximum();
      auto min = compare->GetMinimum();
      auto range = std::abs(max) > std::abs(min) ? std::abs(max) : std::abs(min);
      if (range == 0.f)
        range = 0.1;
      //auto newMax = (max > 0.) ? max*1.2 : max*0.8;

      compare->GetYaxis()->SetRangeUser(-range * 1.5, range * 1.5);
      compare->GetYaxis()->SetTitleOffset(1.5);
      compare->SetMarkerStyle(20);
      compare->SetMarkerSize(0.5);
      compare->Draw("P");

      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      for (const auto &line : boundaries | boost::adaptors::indexed(0)) {
        const auto &index = line.index();
        const auto value = line.value();
        l[index] = TLine(compare->GetBinLowEdge(value.first),
                         canvas.cd()->GetUymin(),
                         compare->GetBinLowEdge(value.first),
                         canvas.cd()->GetUymax());
        l[index].SetLineWidth(1);
        l[index].SetLineStyle(9);
        l[index].SetLineColor(2);
        l[index].Draw("same");
      }

      TLatex tSubdet;
      tSubdet.SetNDC();
      tSubdet.SetTextAlign(21);
      tSubdet.SetTextSize(0.027);
      tSubdet.SetTextAngle(90);

      for (const auto &elem : boundaries) {
        tSubdet.SetTextColor(kRed);
        auto myPair = AlignmentPI::calculatePosition(gPad, compare->GetBinLowEdge(elem.first));
        float theX_ = elem.first != 0 ? myPair.first + 0.025 : myPair.first + 0.01;
        const bool isPhase2 = (ref_ali.size() > AlignmentPI::phase1size);
        tSubdet.DrawLatex(theX_, 0.20, Form("%s", AlignmentPI::getStringFromPart(elem.second, isPhase2).c_str()));
      }

      TLegend legend = TLegend(0.17, 0.86, 0.95, 0.94);
      if (this->m_plotAnnotations.ntags == 2) {
        legend.SetHeader("#bf{Two Tags Comparison}", "C");  // option "C" allows to center the header
        legend.AddEntry(
            compare.get(),
            ("#splitline{" + tagname1 + " : " + firstIOVsince + "}{" + tagname2 + " : " + lastIOVsince + "}").c_str(),
            "PL");
      } else {
        legend.SetHeader(("tag: #bf{" + tagname1 + "}").c_str(), "C");  // option "C" allows to center the header
        legend.AddEntry(compare.get(),
                        ("#splitline{IOV since: " + firstIOVsince + "}{IOV since: " + lastIOVsince + "}").c_str(),
                        "PL");
      }
      legend.SetTextSize(0.020);
      legend.Draw("same");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.042);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("Tracker Alignment Compare :#color[4]{" + s_coord + "}").c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  template <AlignmentPI::coordinate coord>
  using TrackerAlignmentCompare = TrackerAlignmentComparatorBase<coord, 1, MULTI_IOV>;

  template <AlignmentPI::coordinate coord>
  using TrackerAlignmentCompareTwoTags = TrackerAlignmentComparatorBase<coord, 2, SINGLE_IOV>;

  typedef TrackerAlignmentCompare<AlignmentPI::t_x> TrackerAlignmentCompareX;
  typedef TrackerAlignmentCompare<AlignmentPI::t_y> TrackerAlignmentCompareY;
  typedef TrackerAlignmentCompare<AlignmentPI::t_z> TrackerAlignmentCompareZ;

  typedef TrackerAlignmentCompare<AlignmentPI::rot_alpha> TrackerAlignmentCompareAlpha;
  typedef TrackerAlignmentCompare<AlignmentPI::rot_beta> TrackerAlignmentCompareBeta;
  typedef TrackerAlignmentCompare<AlignmentPI::rot_gamma> TrackerAlignmentCompareGamma;

  typedef TrackerAlignmentCompareTwoTags<AlignmentPI::t_x> TrackerAlignmentCompareXTwoTags;
  typedef TrackerAlignmentCompareTwoTags<AlignmentPI::t_y> TrackerAlignmentCompareYTwoTags;
  typedef TrackerAlignmentCompareTwoTags<AlignmentPI::t_z> TrackerAlignmentCompareZTwoTags;

  typedef TrackerAlignmentCompareTwoTags<AlignmentPI::rot_alpha> TrackerAlignmentCompareAlphaTwoTags;
  typedef TrackerAlignmentCompareTwoTags<AlignmentPI::rot_beta> TrackerAlignmentCompareBetaTwoTags;
  typedef TrackerAlignmentCompareTwoTags<AlignmentPI::rot_gamma> TrackerAlignmentCompareGammaTwoTags;

  //*******************************************//
  // Summary canvas per subdetector
  //******************************************//

  template <int ntags, IOVMultiplicity nIOVs, AlignmentPI::partitions q>
  class TrackerAlignmentSummaryBase : public PlotImage<Alignments, nIOVs, ntags> {
  public:
    TrackerAlignmentSummaryBase()
        : PlotImage<Alignments, nIOVs, ntags>("Comparison of all coordinates between two geometries for " +
                                              getStringFromPart(q)) {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<AlignTransform> ref_ali = first_payload->m_align;
      std::vector<AlignTransform> target_ali = last_payload->m_align;

      if (ref_ali.size() != target_ali.size()) {
        edm::LogError("TrackerAlignment_PayloadInspector")
            << "the size of the reference alignment (" << ref_ali.size()
            << ") is different from the one of the target (" << target_ali.size()
            << ")! You are probably trying to compare different underlying geometries. Exiting";
        return false;
      }

      // check that the geomtery is a tracker one
      const char *path_toTopologyXML = (ref_ali.size() == AlignmentPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      for (const auto &ali : ref_ali) {
        auto mydetid = ali.rawId();
        if (DetId(mydetid).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignment_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << DetId(mydetid).rawId() << " (" << DetId(mydetid).det()
              << ") is different from " << DetId::Tracker << " (is DoubleSide: " << tTopo.tidIsDoubleSide(mydetid)
              << "); subdetId " << DetId(mydetid).subdetId() << " - terminating ";
          return false;
        }
      }

      TCanvas canvas("Alignment Comparison", "Alignment Comparison", 1800, 1200);
      canvas.Divide(3, 2);

      std::unordered_map<AlignmentPI::coordinate, std::unique_ptr<TH1F>> diffs;
      std::vector<AlignmentPI::coordinate> coords = {AlignmentPI::t_x,
                                                     AlignmentPI::t_y,
                                                     AlignmentPI::t_z,
                                                     AlignmentPI::rot_alpha,
                                                     AlignmentPI::rot_beta,
                                                     AlignmentPI::rot_gamma};

      for (const auto &coord : coords) {
        auto s_coord = AlignmentPI::getStringFromCoordinate(coord);
        std::string unit =
            (coord == AlignmentPI::t_x || coord == AlignmentPI::t_y || coord == AlignmentPI::t_z) ? "[#mum]" : "[mrad]";

        diffs[coord] = std::make_unique<TH1F>(Form("hDiff_%s", s_coord.c_str()),
                                              Form(";#Delta%s %s;n. of modules", s_coord.c_str(), unit.c_str()),
                                              1001,
                                              -500.5,
                                              500.5);
      }

      // fill the comparison histograms
      std::map<int, AlignmentPI::partitions> boundaries;
      AlignmentPI::fillComparisonHistograms(boundaries, ref_ali, target_ali, diffs, true, q);

      int c_index = 1;

      //TLegend (Double_t x1, Double_t y1, Double_t x2, Double_t y2, const char *header="", Option_t *option="brNDC")
      auto legend = std::make_unique<TLegend>(0.14, 0.88, 0.96, 0.99);
      if (this->m_plotAnnotations.ntags == 2) {
        legend->SetHeader("#bf{Two Tags Comparison}", "C");  // option "C" allows to center the header
        legend->AddEntry(
            diffs[AlignmentPI::t_x].get(),
            ("#splitline{" + tagname1 + " : " + firstIOVsince + "}{" + tagname2 + " : " + lastIOVsince + "}").c_str(),
            "PL");
      } else {
        legend->SetHeader(("tag: #bf{" + tagname1 + "}").c_str(), "C");  // option "C" allows to center the header
        legend->AddEntry(diffs[AlignmentPI::t_x].get(),
                         ("#splitline{IOV since: " + firstIOVsince + "}{IOV since: " + lastIOVsince + "}").c_str(),
                         "PL");
      }
      legend->SetTextSize(0.025);

      for (const auto &coord : coords) {
        canvas.cd(c_index)->SetLogy();
        canvas.cd(c_index)->SetTopMargin(0.01);
        canvas.cd(c_index)->SetBottomMargin(0.15);
        canvas.cd(c_index)->SetLeftMargin(0.14);
        canvas.cd(c_index)->SetRightMargin(0.04);
        diffs[coord]->SetLineWidth(2);
        AlignmentPI::makeNicePlotStyle(diffs[coord].get(), kBlack);

        //float x_max = diffs[coord]->GetXaxis()->GetBinCenter(diffs[coord]->FindLastBinAbove(0.));
        //float x_min = diffs[coord]->GetXaxis()->GetBinCenter(diffs[coord]->FindFirstBinAbove(0.));
        //float extremum = std::abs(x_max) > std::abs(x_min) ? std::abs(x_max) : std::abs(x_min);
        //diffs[coord]->GetXaxis()->SetRangeUser(-extremum*2,extremum*2);

        int i_max = diffs[coord]->FindLastBinAbove(0.);
        int i_min = diffs[coord]->FindFirstBinAbove(0.);
        diffs[coord]->GetXaxis()->SetRange(std::max(1, i_min - 10), std::min(i_max + 10, diffs[coord]->GetNbinsX()));
        diffs[coord]->SetMaximum(diffs[coord]->GetMaximum() * 5);
        diffs[coord]->Draw("HIST");
        AlignmentPI::makeNiceStats(diffs[coord].get(), q, kBlack);

        legend->Draw("same");

        c_index++;
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef TrackerAlignmentSummaryBase<1, MULTI_IOV, AlignmentPI::BPix> TrackerAlignmentSummaryBPix;
  typedef TrackerAlignmentSummaryBase<1, MULTI_IOV, AlignmentPI::FPix> TrackerAlignmentSummaryFPix;
  typedef TrackerAlignmentSummaryBase<1, MULTI_IOV, AlignmentPI::TIB> TrackerAlignmentSummaryTIB;

  typedef TrackerAlignmentSummaryBase<1, MULTI_IOV, AlignmentPI::TID> TrackerAlignmentSummaryTID;
  typedef TrackerAlignmentSummaryBase<1, MULTI_IOV, AlignmentPI::TOB> TrackerAlignmentSummaryTOB;
  typedef TrackerAlignmentSummaryBase<1, MULTI_IOV, AlignmentPI::TEC> TrackerAlignmentSummaryTEC;

  typedef TrackerAlignmentSummaryBase<2, SINGLE_IOV, AlignmentPI::BPix> TrackerAlignmentSummaryBPixTwoTags;
  typedef TrackerAlignmentSummaryBase<2, SINGLE_IOV, AlignmentPI::FPix> TrackerAlignmentSummaryFPixTwoTags;
  typedef TrackerAlignmentSummaryBase<2, SINGLE_IOV, AlignmentPI::TIB> TrackerAlignmentSummaryTIBTwoTags;

  typedef TrackerAlignmentSummaryBase<2, SINGLE_IOV, AlignmentPI::TID> TrackerAlignmentSummaryTIDTwoTags;
  typedef TrackerAlignmentSummaryBase<2, SINGLE_IOV, AlignmentPI::TOB> TrackerAlignmentSummaryTOBTwoTags;
  typedef TrackerAlignmentSummaryBase<2, SINGLE_IOV, AlignmentPI::TEC> TrackerAlignmentSummaryTECTwoTags;

  /************************************************
   Full Pixel Tracker Map Comparison of coordinates
  *************************************************/
  template <AlignmentPI::coordinate coord, int ntags, IOVMultiplicity nIOVs>
  class PixelAlignmentComparisonMapBase : public PlotImage<Alignments, nIOVs, ntags> {
  public:
    PixelAlignmentComparisonMapBase()
        : PlotImage<Alignments, nIOVs, ntags>("SiPixel Comparison Map of " +
                                              AlignmentPI::getStringFromCoordinate(coord)) {
      label_ = "PixelAlignmentComparisonMap" + AlignmentPI::getStringFromCoordinate(coord);
      payloadString = "Tracker Alignment";
    }

    bool fill() override {
      gStyle->SetPalette(1);

      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      const std::vector<AlignTransform> &ref_ali = first_payload->m_align;
      const std::vector<AlignTransform> &target_ali = last_payload->m_align;

      if (last_payload.get() && first_payload.get()) {
        Phase1PixelSummaryMap fullMap(
            "",
            fmt::sprintf("%s %s", payloadString, AlignmentPI::getStringFromCoordinate(coord)),
            fmt::sprintf(
                "#Delta %s [%s]", AlignmentPI::getStringFromCoordinate(coord), (coord <= 3) ? "#mum" : "mrad"));
        fullMap.createTrackerBaseMap();

        if (this->isPhase0(ref_ali) || this->isPhase0(target_ali)) {
          edm::LogError(label_) << "Pixel Tracker Alignment maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          AlignmentPI::displayNotSupported(canvas, 0);
          std::string fileName(this->m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        }

        // fill the map of differences
        std::map<uint32_t, double> diffPerDetid;
        this->fillPerDetIdDiff(coord, ref_ali, target_ali, diffPerDetid);

        // now fill the tracker map
        for (const auto &elem : diffPerDetid) {
          // reject Strips
          int subid = DetId(elem.first).subdetId();
          if (subid > 2) {
            continue;
          }
          fullMap.fillTrackerMap(elem.first, elem.second);
        }

        // limit the axis range (in case of need)
        //fullMap.setZAxisRange(-50.f,50.f);

        TCanvas canvas("Canv", "Canv", 3000, 2000);
        fullMap.printTrackerMap(canvas);

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.025);
        ltx.SetTextAlign(11);

        ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                         gPad->GetBottomMargin() + 0.01,
                         ("#color[4]{" + tagname1 + "}, IOV: #color[4]{" + firstIOVsince + "} vs #color[4]{" +
                          tagname2 + "}, IOV: #color[4]{" + lastIOVsince + "}")
                             .c_str());

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

  protected:
    std::string payloadString;
    std::string label_;

  private:
    //_________________________________________________
    bool isPhase0(std::vector<AlignTransform> theAlis) {
      SiPixelDetInfoFileReader reader =
          SiPixelDetInfoFileReader(edm::FileInPath(SiPixelDetInfoFileReader::kPh0DefaultFile).fullPath());
      const auto &p0detIds = reader.getAllDetIds();

      std::vector<uint32_t> ownDetIds;
      std::transform(theAlis.begin(), theAlis.end(), std::back_inserter(ownDetIds), [](AlignTransform ali) -> uint32_t {
        return ali.rawId();
      });

      for (const auto &det : ownDetIds) {
        // if found at least one phase-0 detId early return
        if (std::find(p0detIds.begin(), p0detIds.end(), det) != p0detIds.end()) {
          return true;
        }
      }
      return false;
    }

    /*--------------------------------------------------------------------*/
    void fillPerDetIdDiff(const AlignmentPI::coordinate &myCoord,
                          const std::vector<AlignTransform> &ref_ali,
                          const std::vector<AlignTransform> &target_ali,
                          std::map<uint32_t, double> &diff)
    /*--------------------------------------------------------------------*/
    {
      for (unsigned int i = 0; i < ref_ali.size(); i++) {
        uint32_t detid = ref_ali[i].rawId();
        if (ref_ali[i].rawId() == target_ali[i].rawId()) {
          CLHEP::HepRotation target_rot(target_ali[i].rotation());
          CLHEP::HepRotation ref_rot(ref_ali[i].rotation());

          align::RotationType target_ROT(target_rot.xx(),
                                         target_rot.xy(),
                                         target_rot.xz(),
                                         target_rot.yx(),
                                         target_rot.yy(),
                                         target_rot.yz(),
                                         target_rot.zx(),
                                         target_rot.zy(),
                                         target_rot.zz());

          align::RotationType ref_ROT(ref_rot.xx(),
                                      ref_rot.xy(),
                                      ref_rot.xz(),
                                      ref_rot.yx(),
                                      ref_rot.yy(),
                                      ref_rot.yz(),
                                      ref_rot.zx(),
                                      ref_rot.zy(),
                                      ref_rot.zz());

          const std::vector<double> deltaRot = {
              ::deltaPhi(align::toAngles(target_ROT)[0], align::toAngles(ref_ROT)[0]),
              ::deltaPhi(align::toAngles(target_ROT)[1], align::toAngles(ref_ROT)[1]),
              ::deltaPhi(align::toAngles(target_ROT)[2], align::toAngles(ref_ROT)[2])};

          const auto &deltaTrans = target_ali[i].translation() - ref_ali[i].translation();

          switch (myCoord) {
            case AlignmentPI::t_x:
              diff.insert({detid, deltaTrans.x() * AlignmentPI::cmToUm});
              break;
            case AlignmentPI::t_y:
              diff.insert({detid, deltaTrans.y() * AlignmentPI::cmToUm});
              break;
            case AlignmentPI::t_z:
              diff.insert({detid, deltaTrans.z() * AlignmentPI::cmToUm});
              break;
            case AlignmentPI::rot_alpha:
              diff.insert({detid, deltaRot[0] * AlignmentPI::tomRad});
              break;
            case AlignmentPI::rot_beta:
              diff.insert({detid, deltaRot[1] * AlignmentPI::tomRad});
              break;
            case AlignmentPI::rot_gamma:
              diff.insert({detid, deltaRot[2] * AlignmentPI::tomRad});
              break;
            default:
              edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << myCoord << std::endl;
              break;
          }  // switch on the coordinate
        }    // check on the same detID
      }      // loop on the components
    }
  };

  template <AlignmentPI::coordinate coord>
  using PixelAlignmentCompareMap = PixelAlignmentComparisonMapBase<coord, 1, MULTI_IOV>;

  template <AlignmentPI::coordinate coord>
  using PixelAlignmentCompareMapTwoTags = PixelAlignmentComparisonMapBase<coord, 2, SINGLE_IOV>;

  typedef PixelAlignmentCompareMap<AlignmentPI::t_x> PixelAlignmentCompareMapX;
  typedef PixelAlignmentCompareMap<AlignmentPI::t_y> PixelAlignmentCompareMapY;
  typedef PixelAlignmentCompareMap<AlignmentPI::t_z> PixelAlignmentCompareMapZ;

  typedef PixelAlignmentCompareMap<AlignmentPI::rot_alpha> PixelAlignmentCompareMapAlpha;
  typedef PixelAlignmentCompareMap<AlignmentPI::rot_beta> PixelAlignmentCompareMapBeta;
  typedef PixelAlignmentCompareMap<AlignmentPI::rot_gamma> PixelAlignmentCompareMapGamma;

  typedef PixelAlignmentCompareMapTwoTags<AlignmentPI::t_x> PixelAlignmentCompareMapXTwoTags;
  typedef PixelAlignmentCompareMapTwoTags<AlignmentPI::t_y> PixelAlignmentCompareMapYTwoTags;
  typedef PixelAlignmentCompareMapTwoTags<AlignmentPI::t_z> PixelAlignmentCompareMapZTwoTags;

  typedef PixelAlignmentCompareMapTwoTags<AlignmentPI::rot_alpha> PixelAlignmentCompareMapAlphaTwoTags;
  typedef PixelAlignmentCompareMapTwoTags<AlignmentPI::rot_beta> PixelAlignmentCompareMapBetaTwoTags;
  typedef PixelAlignmentCompareMapTwoTags<AlignmentPI::rot_gamma> PixelAlignmentCompareMapGammaTwoTags;

  //*******************************************//
  // History of the position of the BPix Barycenter
  //******************************************//

  template <AlignmentPI::coordinate coord>
  class BPixBarycenterHistory : public HistoryPlot<Alignments, float> {
  public:
    BPixBarycenterHistory()
        : HistoryPlot<Alignments, float>(
              " Barrel Pixel " + AlignmentPI::getStringFromCoordinate(coord) + " positions vs time",
              AlignmentPI::getStringFromCoordinate(coord) + " position [cm]") {}
    ~BPixBarycenterHistory() override = default;

    float getFromPayload(Alignments &payload) override {
      std::vector<AlignTransform> alignments = payload.m_align;

      float barycenter = 0.;
      float nmodules(0.);
      for (const auto &ali : alignments) {
        if (DetId(ali.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignment_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << ali.rawId() << " " << DetId(ali.rawId()).det()
              << " is different from " << DetId::Tracker << "  - terminating ";
          return false;
        }

        int subid = DetId(ali.rawId()).subdetId();
        if (subid != PixelSubdetector::PixelBarrel)
          continue;

        nmodules++;
        switch (coord) {
          case AlignmentPI::t_x:
            barycenter += (ali.translation().x());
            break;
          case AlignmentPI::t_y:
            barycenter += (ali.translation().y());
            break;
          case AlignmentPI::t_z:
            barycenter += (ali.translation().z());
            break;
          default:
            edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << coord << std::endl;
            break;
        }  // switch on the coordinate (only X,Y,Z are interesting)
      }    // ends loop on the alignments

      edm::LogInfo("TrackerAlignment_PayloadInspector") << "barycenter (" << barycenter << ")/n. modules (" << nmodules
                                                        << ") =  " << barycenter / nmodules << std::endl;

      // take the mean
      barycenter /= nmodules;

      // applied GPR correction to move barycenter to global CMS coordinates
      barycenter += hardcodeGPR.at(coord);

      return barycenter;

    }  // payload
  };

  typedef BPixBarycenterHistory<AlignmentPI::t_x> X_BPixBarycenterHistory;
  typedef BPixBarycenterHistory<AlignmentPI::t_y> Y_BPixBarycenterHistory;
  typedef BPixBarycenterHistory<AlignmentPI::t_z> Z_BPixBarycenterHistory;

  /************************************************
    Display of Tracker Detector barycenters
  *************************************************/
  class TrackerAlignmentBarycenters : public PlotImage<Alignments, SINGLE_IOV> {
  public:
    TrackerAlignmentBarycenters() : PlotImage<Alignments, SINGLE_IOV>("Display of Tracker Alignment Barycenters") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      const auto &tagname = PlotBase::getTag<0>().name;
      std::shared_ptr<Alignments> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);

      TCanvas canvas("Tracker Alignment Barycenter Summary", "Tracker Alignment Barycenter summary", 1600, 1000);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.15);
      canvas.SetRightMargin(0.03);
      canvas.Modified();
      canvas.SetGrid();

      auto h2_BarycenterParameters =
          std::make_unique<TH2F>("Parameters", "SubDetector Barycenter summary", 6, 0.0, 6.0, 6, 0, 6.);

      auto h2_uncBarycenterParameters =
          std::make_unique<TH2F>("Parameters2", "SubDetector Barycenter summary", 6, 0.0, 6.0, 6, 0, 6.);

      h2_BarycenterParameters->SetStats(false);
      h2_BarycenterParameters->SetTitle(nullptr);
      h2_uncBarycenterParameters->SetStats(false);
      h2_uncBarycenterParameters->SetTitle(nullptr);

      std::vector<AlignTransform> alignments = payload->m_align;

      isPhase0 = (alignments.size() == AlignmentPI::phase0size) ? true : false;

      // check that the geomtery is a tracker one
      const char *path_toTopologyXML = isPhase0 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                                : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";

      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      AlignmentPI::TkAlBarycenters barycenters;
      // compute uncorrected barycenter
      barycenters.computeBarycenters(
          alignments, tTopo, {{AlignmentPI::t_x, 0.0}, {AlignmentPI::t_y, 0.0}, {AlignmentPI::t_z, 0.0}});

      auto Xbarycenters = barycenters.getX();
      auto Ybarycenters = barycenters.getY();
      auto Zbarycenters = barycenters.getZ();

      // compute barycenter corrected for the GPR
      barycenters.computeBarycenters(alignments, tTopo, hardcodeGPR);

      auto c_Xbarycenters = barycenters.getX();
      auto c_Ybarycenters = barycenters.getY();
      auto c_Zbarycenters = barycenters.getZ();

      h2_BarycenterParameters->GetXaxis()->SetBinLabel(1, "X [cm]");
      h2_BarycenterParameters->GetXaxis()->SetBinLabel(2, "Y [cm]");
      h2_BarycenterParameters->GetXaxis()->SetBinLabel(3, "Z [cm]");
      h2_BarycenterParameters->GetXaxis()->SetBinLabel(4, "X_{no GPR} [cm]");
      h2_BarycenterParameters->GetXaxis()->SetBinLabel(5, "Y_{no GPR} [cm]");
      h2_BarycenterParameters->GetXaxis()->SetBinLabel(6, "Z_{no GPR} [cm]");

      bool isLikelyMC(false);
      int checkX =
          std::count_if(Xbarycenters.begin(), Xbarycenters.begin() + 2, [](float a) { return (std::abs(a) >= 1.e-4); });
      int checkY =
          std::count_if(Ybarycenters.begin(), Ybarycenters.begin() + 2, [](float a) { return (std::abs(a) >= 1.e-4); });
      int checkZ =
          std::count_if(Zbarycenters.begin(), Zbarycenters.begin() + 2, [](float a) { return (std::abs(a) >= 1.e-4); });

      // if all the coordinate barycenters for both BPix and FPix are below 10um
      // this is very likely a MC payload
      if ((checkX + checkY + checkZ) == 0 && run == 1)
        isLikelyMC = true;

      unsigned int yBin = 6;
      for (unsigned int i = 0; i < 6; i++) {
        auto thePart = static_cast<AlignmentPI::partitions>(i + 1);
        std::string theLabel = getStringFromPart(thePart);
        h2_BarycenterParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        if (!isLikelyMC) {
          h2_BarycenterParameters->SetBinContent(1, yBin, c_Xbarycenters[i]);
          h2_BarycenterParameters->SetBinContent(2, yBin, c_Ybarycenters[i]);
          h2_BarycenterParameters->SetBinContent(3, yBin, c_Zbarycenters[i]);
        }

        h2_uncBarycenterParameters->SetBinContent(4, yBin, Xbarycenters[i]);
        h2_uncBarycenterParameters->SetBinContent(5, yBin, Ybarycenters[i]);
        h2_uncBarycenterParameters->SetBinContent(6, yBin, Zbarycenters[i]);
        yBin--;
      }

      h2_BarycenterParameters->GetXaxis()->LabelsOption("h");
      h2_BarycenterParameters->GetYaxis()->SetLabelSize(0.05);
      h2_BarycenterParameters->GetXaxis()->SetLabelSize(0.05);
      h2_BarycenterParameters->SetMarkerSize(1.5);
      h2_BarycenterParameters->Draw("TEXT");

      h2_uncBarycenterParameters->SetMarkerSize(1.5);
      h2_uncBarycenterParameters->SetMarkerColor(kRed);
      h2_uncBarycenterParameters->Draw("TEXTsame");

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.045);
      t1.DrawLatex(0.5, 0.96, Form("TkAl Barycenters, Tag: #color[4]{%s}, IOV #color[4]{%i}", tagname.c_str(), run));
      t1.SetTextSize(0.025);

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    bool isPhase0;
  };

  /************************************************
    Comparator of Tracker Detector barycenters
  *************************************************/
  template <int ntags, IOVMultiplicity nIOVs>
  class TrackerAlignmentBarycentersComparatorBase : public PlotImage<Alignments, nIOVs, ntags> {
  public:
    TrackerAlignmentBarycentersComparatorBase()
        : PlotImage<Alignments, nIOVs, ntags>("Comparison of Tracker Alignment Barycenters") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      unsigned int first_run = std::get<0>(firstiov);
      unsigned int last_run = std::get<0>(lastiov);

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::vector<AlignTransform> last_alignments = last_payload->m_align;

      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));
      std::vector<AlignTransform> first_alignments = first_payload->m_align;

      isInitialPhase0 = (first_alignments.size() == AlignmentPI::phase0size) ? true : false;
      isFinalPhase0 = (last_alignments.size() == AlignmentPI::phase0size) ? true : false;

      // check that the geomtery is a tracker one
      const char *path_toTopologyXML = isInitialPhase0 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                                       : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";

      TrackerTopology tTopo_f =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      path_toTopologyXML = isFinalPhase0 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                         : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";

      TrackerTopology tTopo_l =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      TCanvas canvas("Tracker Alignment Barycenter Summary", "Tracker Alignment Barycenter summary", 1200, 800);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.15);
      canvas.SetRightMargin(0.03);
      canvas.Modified();
      canvas.SetGrid();

      auto h2_BarycenterDiff =
          std::make_unique<TH2F>("Parameters diff", "SubDetector Barycenter Difference", 3, 0.0, 3.0, 6, 0, 6.);

      h2_BarycenterDiff->SetStats(false);
      h2_BarycenterDiff->SetTitle(nullptr);
      h2_BarycenterDiff->GetXaxis()->SetBinLabel(1, "X [#mum]");
      h2_BarycenterDiff->GetXaxis()->SetBinLabel(2, "Y [#mum]");
      h2_BarycenterDiff->GetXaxis()->SetBinLabel(3, "Z [#mum]");

      AlignmentPI::TkAlBarycenters l_barycenters;
      l_barycenters.computeBarycenters(last_alignments, tTopo_l, hardcodeGPR);

      AlignmentPI::TkAlBarycenters f_barycenters;
      f_barycenters.computeBarycenters(first_alignments, tTopo_f, hardcodeGPR);

      unsigned int yBin = 6;
      for (unsigned int i = 0; i < 6; i++) {
        auto thePart = static_cast<AlignmentPI::partitions>(i + 1);
        std::string theLabel = getStringFromPart(thePart);
        h2_BarycenterDiff->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_BarycenterDiff->SetBinContent(
            1, yBin, (l_barycenters.getX()[i] - f_barycenters.getX()[i]) * AlignmentPI::cmToUm);
        h2_BarycenterDiff->SetBinContent(
            2, yBin, (l_barycenters.getY()[i] - f_barycenters.getY()[i]) * AlignmentPI::cmToUm);
        h2_BarycenterDiff->SetBinContent(
            3, yBin, (l_barycenters.getZ()[i] - f_barycenters.getZ()[i]) * AlignmentPI::cmToUm);
        yBin--;
      }

      h2_BarycenterDiff->GetXaxis()->LabelsOption("h");
      h2_BarycenterDiff->GetYaxis()->SetLabelSize(0.05);
      h2_BarycenterDiff->GetXaxis()->SetLabelSize(0.05);
      h2_BarycenterDiff->SetMarkerSize(1.5);
      h2_BarycenterDiff->SetMarkerColor(kRed);
      h2_BarycenterDiff->Draw("TEXT");

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Tracker Alignment Barycenters Diff, IOV %i - IOV %i", last_run, first_run));
      t1.SetTextSize(0.025);

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    bool isInitialPhase0;
    bool isFinalPhase0;
  };

  using TrackerAlignmentBarycentersCompare = TrackerAlignmentBarycentersComparatorBase<1, MULTI_IOV>;
  using TrackerAlignmentBarycentersCompareTwoTags = TrackerAlignmentBarycentersComparatorBase<2, SINGLE_IOV>;

  /************************************************
    Comparator of Pixel Tracker Detector barycenters
  *************************************************/
  template <int ntags, IOVMultiplicity nIOVs>
  class PixelBarycentersComparatorBase : public PlotImage<Alignments, nIOVs, ntags> {
  public:
    PixelBarycentersComparatorBase() : PlotImage<Alignments, nIOVs, ntags>("Comparison of Pixel Barycenters") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = PlotBase::getTag<0>().iovs;
      auto tagname1 = PlotBase::getTag<0>().name;
      std::string tagname2 = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = PlotBase::getTag<1>().iovs;
        tagname2 = PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      unsigned int first_run = std::get<0>(firstiov);
      unsigned int last_run = std::get<0>(lastiov);

      std::shared_ptr<Alignments> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::vector<AlignTransform> last_alignments = last_payload->m_align;

      std::shared_ptr<Alignments> first_payload = this->fetchPayload(std::get<1>(firstiov));
      std::vector<AlignTransform> first_alignments = first_payload->m_align;

      TCanvas canvas("Pixel Barycenter Summary", "Pixel Barycenter summary", 1200, 1200);
      canvas.Divide(2, 2);
      canvas.cd();

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.5,
                   0.97,
                   ("Pixel Barycenters comparison, IOV: #color[2]{" + std::to_string(first_run) +
                    "} vs IOV: #color[4]{" + std::to_string(last_run) + "}")
                       .c_str());
      t1.SetTextSize(0.025);

      for (unsigned int c = 1; c <= 4; c++) {
        canvas.cd(c)->SetTopMargin(0.07);
        canvas.cd(c)->SetBottomMargin(0.12);
        canvas.cd(c)->SetLeftMargin(0.15);
        canvas.cd(c)->SetRightMargin(0.03);
        canvas.cd(c)->Modified();
        canvas.cd(c)->SetGrid();
      }

      std::array<std::string, 3> structures = {{"FPIX-", "BPIX", "FPIX+"}};
      std::array<std::unique_ptr<TH2F>, 3> histos;

      isInitialPhase0 = (first_alignments.size() == AlignmentPI::phase0size) ? true : false;
      isFinalPhase0 = (last_alignments.size() == AlignmentPI::phase0size) ? true : false;

      // check that the geomtery is a tracker one
      const char *path_toTopologyXML = isInitialPhase0 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                                       : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";

      TrackerTopology tTopo_f =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      AlignmentPI::TkAlBarycenters myInitialBarycenters;
      //myInitialBarycenters.computeBarycenters(first_alignments,tTopo_f,hardcodeGPR);
      myInitialBarycenters.computeBarycenters(
          first_alignments, tTopo_f, {{AlignmentPI::t_x, 0.0}, {AlignmentPI::t_y, 0.0}, {AlignmentPI::t_z, 0.0}});

      path_toTopologyXML = isFinalPhase0 ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                         : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";

      TrackerTopology tTopo_l =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      AlignmentPI::TkAlBarycenters myFinalBarycenters;
      //myFinalBarycenters.computeBarycenters(last_alignments,tTopo_l,hardcodeGPR);
      myFinalBarycenters.computeBarycenters(
          last_alignments, tTopo_l, {{AlignmentPI::t_x, 0.0}, {AlignmentPI::t_y, 0.0}, {AlignmentPI::t_z, 0.0}});

      if (isFinalPhase0 != isInitialPhase0) {
        edm::LogWarning("TrackerAlignment_PayloadInspector")
            << "the size of the reference alignment (" << first_alignments.size()
            << ") is different from the one of the target (" << last_alignments.size()
            << ")! You are probably trying to compare different underlying geometries.";
      }

      unsigned int index(0);
      for (const auto &piece : structures) {
        const char *name = piece.c_str();
        histos[index] = std::make_unique<TH2F>(
            name,
            Form("%s x-y Barycenter Difference;x_{%s}-x_{TOB} [mm];y_{%s}-y_{TOB} [mm]", name, name, name),
            100,
            -3.,
            3.,
            100,
            -3.,
            3.);

        histos[index]->SetStats(false);
        histos[index]->SetTitle(nullptr);
        histos[index]->GetYaxis()->SetLabelSize(0.05);
        histos[index]->GetXaxis()->SetLabelSize(0.05);
        histos[index]->GetYaxis()->SetTitleSize(0.06);
        histos[index]->GetXaxis()->SetTitleSize(0.06);
        histos[index]->GetYaxis()->CenterTitle();
        histos[index]->GetXaxis()->CenterTitle();
        histos[index]->GetXaxis()->SetTitleOffset(0.9);
        index++;
      }

      auto h2_ZBarycenterDiff = std::make_unique<TH2F>(
          "Pixel_z_diff", "Pixel z-Barycenter Difference;; z_{Pixel-Ideal} -z_{TOB} [mm]", 3, -0.5, 2.5, 100, -10., 10.);
      h2_ZBarycenterDiff->SetStats(false);
      h2_ZBarycenterDiff->SetTitle(nullptr);
      h2_ZBarycenterDiff->GetXaxis()->SetBinLabel(1, "FPIX -");
      h2_ZBarycenterDiff->GetXaxis()->SetBinLabel(2, "BPIX");
      h2_ZBarycenterDiff->GetXaxis()->SetBinLabel(3, "FPIX +");
      h2_ZBarycenterDiff->GetYaxis()->SetLabelSize(0.05);
      h2_ZBarycenterDiff->GetXaxis()->SetLabelSize(0.07);
      h2_ZBarycenterDiff->GetYaxis()->SetTitleSize(0.06);
      h2_ZBarycenterDiff->GetXaxis()->SetTitleSize(0.06);
      h2_ZBarycenterDiff->GetYaxis()->CenterTitle();
      h2_ZBarycenterDiff->GetXaxis()->CenterTitle();
      h2_ZBarycenterDiff->GetYaxis()->SetTitleOffset(1.1);

      std::function<GlobalPoint(int)> cutFunctorInitial = [&myInitialBarycenters](int index) {
        switch (index) {
          case 1:
            return myInitialBarycenters.getPartitionAvg(AlignmentPI::PARTITION::FPIXm);
          case 2:
            return myInitialBarycenters.getPartitionAvg(AlignmentPI::PARTITION::BPIX);
          case 3:
            return myInitialBarycenters.getPartitionAvg(AlignmentPI::PARTITION::FPIXp);
          default:
            return GlobalPoint(0, 0, 0);
        }
      };

      std::function<GlobalPoint(int)> cutFunctorFinal = [&myFinalBarycenters](int index) {
        switch (index) {
          case 1:
            return myFinalBarycenters.getPartitionAvg(AlignmentPI::PARTITION::FPIXm);
          case 2:
            return myFinalBarycenters.getPartitionAvg(AlignmentPI::PARTITION::BPIX);
          case 3:
            return myFinalBarycenters.getPartitionAvg(AlignmentPI::PARTITION::FPIXp);
          default:
            return GlobalPoint(0, 0, 0);
        }
      };

      float x0i, x0f, y0i, y0f;

      t1.SetNDC(kFALSE);
      t1.SetTextSize(0.047);
      for (unsigned int c = 1; c <= 3; c++) {
        x0i = cutFunctorInitial(c).x() * 10;  // transform cm to mm (x10)
        x0f = cutFunctorFinal(c).x() * 10;
        y0i = cutFunctorInitial(c).y() * 10;
        y0f = cutFunctorFinal(c).y() * 10;

        canvas.cd(c);
        histos[c - 1]->Draw();

        COUT << "initial x,y " << std::left << std::setw(7) << structures[c - 1] << " (" << x0i << "," << y0i << ") mm"
             << std::endl;
        COUT << "final   x,y " << std::left << std::setw(7) << structures[c - 1] << " (" << x0f << "," << y0f << ") mm"
             << std::endl;

        TMarker *initial = new TMarker(x0i, y0i, 21);
        TMarker *final = new TMarker(x0f, y0f, 20);

        initial->SetMarkerColor(kRed);
        final->SetMarkerColor(kBlue);
        initial->SetMarkerSize(2.5);
        final->SetMarkerSize(2.5);
        t1.SetTextColor(kRed);
        initial->Draw();
        t1.DrawLatex(x0i, y0i + (y0i > y0f ? 0.3 : -0.5), Form("(%.2f,%.2f)", x0i, y0i));
        final->Draw("same");
        t1.SetTextColor(kBlue);
        t1.DrawLatex(x0f, y0f + (y0i > y0f ? -0.5 : 0.3), Form("(%.2f,%.2f)", x0f, y0f));
      }

      // fourth pad is a special case for the z coordinate
      canvas.cd(4);
      h2_ZBarycenterDiff->Draw();
      float z0i, z0f;

      // numbers do agree with https://twiki.cern.ch/twiki/bin/view/CMSPublic/TkAlignmentPerformancePhaseIStartUp17#Pixel_Barycentre_Positions

      std::array<double, 3> hardcodeIdealZPhase0 = {{-41.94909, 0., 41.94909}};  // units are cm
      std::array<double, 3> hardcodeIdealZPhase1 = {{-39.82911, 0., 39.82911}};  // units are cm

      for (unsigned int c = 1; c <= 3; c++) {
        // less than pretty but needed to remove the z position of the FPix barycenters != 0

        z0i =
            (cutFunctorInitial(c).z() - (isInitialPhase0 ? hardcodeIdealZPhase0[c - 1] : hardcodeIdealZPhase1[c - 1])) *
            10;  // convert to mm
        z0f =
            (cutFunctorFinal(c).z() - (isFinalPhase0 ? hardcodeIdealZPhase0[c - 1] : hardcodeIdealZPhase1[c - 1])) * 10;

        TMarker *initial = new TMarker(c - 1, z0i, 21);
        TMarker *final = new TMarker(c - 1, z0f, 20);

        COUT << "initial   z " << std::left << std::setw(7) << structures[c - 1] << " " << z0i << " mm" << std::endl;
        COUT << "final     z " << std::left << std::setw(7) << structures[c - 1] << " " << z0f << " mm" << std::endl;

        initial->SetMarkerColor(kRed);
        final->SetMarkerColor(kBlue);
        initial->SetMarkerSize(2.5);
        final->SetMarkerSize(2.5);
        initial->Draw();
        t1.SetTextColor(kRed);
        t1.DrawLatex(c - 1, z0i + (z0i > z0f ? 1. : -1.5), Form("(%.2f)", z0i));
        final->Draw("same");
        t1.SetTextColor(kBlue);
        t1.DrawLatex(c - 1, z0f + (z0i > z0f ? -1.5 : 1), Form("(%.2f)", z0f));
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  private:
    bool isInitialPhase0;
    bool isFinalPhase0;
  };

  using PixelBarycentersCompare = PixelBarycentersComparatorBase<1, MULTI_IOV>;
  using PixelBarycentersCompareTwoTags = PixelBarycentersComparatorBase<2, SINGLE_IOV>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(TrackerAlignment) {
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentComparatorSingleTag);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(OTAlignmentComparatorSingleTag);
  PAYLOAD_INSPECTOR_CLASS(OTAlignmentComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentComparatorSingleTag);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareRPhiZSingleTag);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareRPhiZTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareRPhiZSingleTag);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareRPhiZTwoTags);
  PAYLOAD_INSPECTOR_CLASS(OTAlignmentCompareRPhiZSingleTag);
  PAYLOAD_INSPECTOR_CLASS(OTAlignmentCompareRPhiZTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareX);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareY);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareZ);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareAlpha);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareBeta);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareGamma);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareXTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareYTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareZTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareAlphaTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareBetaTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareGammaTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapX);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapY);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapZ);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapAlpha);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapBeta);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapGamma);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapXTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapYTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapZTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapAlphaTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapBetaTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelAlignmentCompareMapGammaTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryBPix);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryFPix);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTIB);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTID);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTOB);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTEC);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryBPixTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryFPixTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTIBTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTIDTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTOBTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTECTwoTags);
  PAYLOAD_INSPECTOR_CLASS(X_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(Y_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(Z_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentBarycenters);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentBarycentersCompare);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentBarycentersCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelBarycentersCompare);
  PAYLOAD_INSPECTOR_CLASS(PixelBarycentersCompareTwoTags);
}
