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

#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>  // std::setprecision

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

  //*******************************************//
  // Size of the movement over all partitions
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

      int counter = 0;
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

      std::vector<int> boundaries;
      AlignmentPI::partitions currentPart = AlignmentPI::BPix;
      for (unsigned int i = 0; i < ref_ali.size(); i++) {
        if (ref_ali[i].rawId() == target_ali[i].rawId()) {
          counter++;
          int subid = DetId(ref_ali[i].rawId()).subdetId();

          auto thePart = static_cast<AlignmentPI::partitions>(subid);
          if (thePart != currentPart) {
            currentPart = thePart;
            boundaries.push_back(counter);
          }

          CLHEP::HepRotation target_rot(target_ali[i].rotation());
          CLHEP::HepRotation ref_rot(ref_ali[i].rotation());

          align::RotationType target_rotation(target_rot.xx(),
                                              target_rot.xy(),
                                              target_rot.xz(),
                                              target_rot.yx(),
                                              target_rot.yy(),
                                              target_rot.yz(),
                                              target_rot.zx(),
                                              target_rot.zy(),
                                              target_rot.zz());

          align::RotationType ref_rotation(ref_rot.xx(),
                                           ref_rot.xy(),
                                           ref_rot.xz(),
                                           ref_rot.yx(),
                                           ref_rot.yy(),
                                           ref_rot.yz(),
                                           ref_rot.zx(),
                                           ref_rot.zy(),
                                           ref_rot.zz());

          align::EulerAngles target_eulerAngles = align::toAngles(target_rotation);
          align::EulerAngles ref_eulerAngles = align::toAngles(ref_rotation);

          switch (coord) {
            case AlignmentPI::t_x:
              compare->SetBinContent(
                  i + 1, (target_ali[i].translation().x() - ref_ali[i].translation().x()) * AlignmentPI::cmToUm);
              break;
            case AlignmentPI::t_y:
              compare->SetBinContent(
                  i + 1, (target_ali[i].translation().y() - ref_ali[i].translation().y()) * AlignmentPI::cmToUm);
              break;
            case AlignmentPI::t_z:
              compare->SetBinContent(
                  i + 1, (target_ali[i].translation().z() - ref_ali[i].translation().z()) * AlignmentPI::cmToUm);
              break;
            case AlignmentPI::rot_alpha: {
              auto deltaRot = target_eulerAngles[0] - ref_eulerAngles[0];
              compare->SetBinContent(i + 1, AlignmentPI::returnZeroIfNear2PI(deltaRot) * AlignmentPI::tomRad);
              break;
            }
            case AlignmentPI::rot_beta: {
              auto deltaRot = target_eulerAngles[1] - ref_eulerAngles[1];
              compare->SetBinContent(i + 1, AlignmentPI::returnZeroIfNear2PI(deltaRot) * AlignmentPI::tomRad);
              break;
            }
            case AlignmentPI::rot_gamma: {
              auto deltaRot = target_eulerAngles[2] - ref_eulerAngles[2];
              compare->SetBinContent(i + 1, AlignmentPI::returnZeroIfNear2PI(deltaRot) * AlignmentPI::tomRad);
              break;
            }
            default:
              edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << coord << std::endl;
              break;
          }  // switch on the coordinate
        }    // check on the same detID
      }      // loop on the components

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
      unsigned int i = 0;
      for (const auto &line : boundaries) {
        l[i] = TLine(compare->GetBinLowEdge(line),
                     canvas.cd()->GetUymin(),
                     compare->GetBinLowEdge(line),
                     canvas.cd()->GetUymax());
        l[i].SetLineWidth(1);
        l[i].SetLineStyle(9);
        l[i].SetLineColor(2);
        l[i].Draw("same");
        i++;
      }

      TLatex tSubdet;
      tSubdet.SetNDC();
      tSubdet.SetTextAlign(21);
      tSubdet.SetTextSize(0.027);
      tSubdet.SetTextAngle(90);
      for (unsigned int j = 1; j <= 6; j++) {
        auto thePart = static_cast<AlignmentPI::partitions>(j);
        tSubdet.SetTextColor(kRed);
        auto myPair = (j > 1) ? AlignmentPI::calculatePosition(gPad, compare->GetBinLowEdge(boundaries[j - 2]))
                              : AlignmentPI::calculatePosition(gPad, compare->GetBinLowEdge(0));
        float theX_ = myPair.first + 0.025;
        tSubdet.DrawLatex(theX_, 0.20, Form("%s", (AlignmentPI::getStringFromPart(thePart)).c_str()));
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

  template <AlignmentPI::partitions q>
  class TrackerAlignmentSummary : public PlotImage<Alignments, MULTI_IOV> {
  public:
    TrackerAlignmentSummary()
        : PlotImage<Alignments, MULTI_IOV>("Comparison of all coordinates between two geometries for " +
                                           getStringFromPart(q)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto sorted_iovs = tag.iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<Alignments> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<Alignments> first_payload = fetchPayload(std::get<1>(firstiov));

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

      std::unordered_map<AlignmentPI::coordinate, std::unique_ptr<TH1F> > diffs;
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
                                              1000,
                                              -500.,
                                              500.);
      }

      int loopedComponents(0);
      for (unsigned int i = 0; i < ref_ali.size(); i++) {
        if (ref_ali[i].rawId() == target_ali[i].rawId()) {
          loopedComponents++;
          int subid = DetId(ref_ali[i].rawId()).subdetId();
          auto thePart = static_cast<AlignmentPI::partitions>(subid);
          if (thePart != q)
            continue;

          CLHEP::HepRotation target_rot(target_ali[i].rotation());
          CLHEP::HepRotation ref_rot(ref_ali[i].rotation());

          align::RotationType target_rotation(target_rot.xx(),
                                              target_rot.xy(),
                                              target_rot.xz(),
                                              target_rot.yx(),
                                              target_rot.yy(),
                                              target_rot.yz(),
                                              target_rot.zx(),
                                              target_rot.zy(),
                                              target_rot.zz());

          align::RotationType ref_rotation(ref_rot.xx(),
                                           ref_rot.xy(),
                                           ref_rot.xz(),
                                           ref_rot.yx(),
                                           ref_rot.yy(),
                                           ref_rot.yz(),
                                           ref_rot.zx(),
                                           ref_rot.zy(),
                                           ref_rot.zz());

          align::EulerAngles target_eulerAngles = align::toAngles(target_rotation);
          align::EulerAngles ref_eulerAngles = align::toAngles(ref_rotation);

          for (const auto &coord : coords) {
            switch (coord) {
              case AlignmentPI::t_x:
                diffs[coord]->Fill((target_ali[i].translation().x() - ref_ali[i].translation().x()) *
                                   AlignmentPI::cmToUm);
                break;
              case AlignmentPI::t_y:
                diffs[coord]->Fill((target_ali[i].translation().y() - ref_ali[i].translation().y()) *
                                   AlignmentPI::cmToUm);
                break;
              case AlignmentPI::t_z:
                diffs[coord]->Fill((target_ali[i].translation().z() - ref_ali[i].translation().z()) *
                                   AlignmentPI::cmToUm);
                break;
              case AlignmentPI::rot_alpha: {
                auto deltaRot = target_eulerAngles[0] - ref_eulerAngles[0];
                diffs[coord]->Fill(AlignmentPI::returnZeroIfNear2PI(deltaRot) * AlignmentPI::tomRad);
                break;
              }
              case AlignmentPI::rot_beta: {
                auto deltaRot = target_eulerAngles[1] - ref_eulerAngles[1];
                diffs[coord]->Fill(AlignmentPI::returnZeroIfNear2PI(deltaRot) * AlignmentPI::tomRad);
                break;
              }
              case AlignmentPI::rot_gamma: {
                auto deltaRot = target_eulerAngles[2] - ref_eulerAngles[2];
                diffs[coord]->Fill(AlignmentPI::returnZeroIfNear2PI(deltaRot) * AlignmentPI::tomRad);
                break;
              }
              default:
                edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << coord << std::endl;
                break;
            }  // switch on the coordinate
          }
        }  // check on the same detID
      }    // loop on the components

      int c_index = 1;

      auto legend = std::make_unique<TLegend>(0.14, 0.93, 0.55, 0.98);
      legend->AddEntry(
          diffs[AlignmentPI::t_x].get(),
          ("#DeltaIOV: " + std::to_string(std::get<0>(lastiov)) + "-" + std::to_string(std::get<0>(firstiov))).c_str(),
          "L");
      legend->SetTextSize(0.03);

      for (const auto &coord : coords) {
        canvas.cd(c_index)->SetLogy();
        canvas.cd(c_index)->SetTopMargin(0.02);
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
        diffs[coord]->Draw("HIST");
        AlignmentPI::makeNiceStats(diffs[coord].get(), q, kBlack);

        legend->Draw("same");

        c_index++;
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef TrackerAlignmentSummary<AlignmentPI::BPix> TrackerAlignmentSummaryBPix;
  typedef TrackerAlignmentSummary<AlignmentPI::FPix> TrackerAlignmentSummaryFPix;
  typedef TrackerAlignmentSummary<AlignmentPI::TIB> TrackerAlignmentSummaryTIB;

  typedef TrackerAlignmentSummary<AlignmentPI::TID> TrackerAlignmentSummaryTID;
  typedef TrackerAlignmentSummary<AlignmentPI::TOB> TrackerAlignmentSummaryTOB;
  typedef TrackerAlignmentSummary<AlignmentPI::TEC> TrackerAlignmentSummaryTEC;

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
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Tracker Alignment Barycenters, IOV %i", run));
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
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryBPix);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryFPix);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTIB);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTID);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTOB);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTEC);
  PAYLOAD_INSPECTOR_CLASS(X_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(Y_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(Z_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentBarycenters);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentBarycentersCompare);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentBarycentersCompareTwoTags);
  PAYLOAD_INSPECTOR_CLASS(PixelBarycentersCompare);
  PAYLOAD_INSPECTOR_CLASS(PixelBarycentersCompareTwoTags);
}
