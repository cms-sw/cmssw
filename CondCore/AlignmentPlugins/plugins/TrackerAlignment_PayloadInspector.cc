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
#include "DataFormats/DetId/interface/DetId.h"

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

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

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

  template <AlignmentPI::coordinate coord>
  class TrackerAlignmentCompare : public cond::payloadInspector::PlotImage<Alignments> {
  public:
    TrackerAlignmentCompare()
        : cond::payloadInspector::PlotImage<Alignments>("comparison of " + AlignmentPI::getStringFromCoordinate(coord) +
                                                        " coordinate between two geometries") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      std::vector<std::tuple<cond::Time_t, cond::Hash> > sorted_iovs = iovs;

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
          std::unique_ptr<TH1F>(new TH1F("comparison",
                                         Form(";Detector Id index; #Delta%s %s", s_coord.c_str(), unit.c_str()),
                                         ref_ali.size(),
                                         -0.5,
                                         ref_ali.size() - 0.5));

      std::vector<int> boundaries;
      AlignmentPI::partitions currentPart = AlignmentPI::BPix;
      for (unsigned int i = 0; i <= ref_ali.size(); i++) {
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
            case AlignmentPI::rot_alpha:
              compare->SetBinContent(i + 1, (target_eulerAngles[0] - ref_eulerAngles[0]) * 1000.);
              break;
            case AlignmentPI::rot_beta:
              compare->SetBinContent(i + 1, (target_eulerAngles[1] - ref_eulerAngles[1]) * 1000.);
              break;
            case AlignmentPI::rot_gamma:
              compare->SetBinContent(i + 1, (target_eulerAngles[2] - ref_eulerAngles[2]) * 1000.);
              break;
            default:
              edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << coord << std::endl;
              break;
          }  // switch on the coordinate
        }    // check on the same detID
      }      // loop on the components

      canvas.cd();

      canvas.SetLeftMargin(0.17);
      canvas.SetRightMargin(0.05);
      canvas.SetBottomMargin(0.15);
      AlignmentPI::makeNicePlotStyle(compare.get(), kBlack);
      auto max = compare->GetMaximum();
      auto min = compare->GetMinimum();
      auto range = std::abs(max) > std::abs(min) ? std::abs(max) : std::abs(min);
      //auto newMax = (max > 0.) ? max*1.2 : max*0.8;
      compare->GetYaxis()->SetRangeUser(-range * 1.3, range * 1.2);
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

      TLegend legend = TLegend(0.58, 0.82, 0.95, 0.9);
      legend.SetTextSize(0.03);
      legend.SetHeader("Alignment comparison", "C");  // option "C" allows to center the header
      legend.AddEntry(
          compare.get(),
          ("IOV:" + std::to_string(std::get<0>(lastiov)) + "-" + std::to_string(std::get<0>(firstiov))).c_str(),
          "PL");
      legend.Draw("same");

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(21);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.2, 0.93, Form("%s", s_coord.c_str()));
      t1.SetTextColor(kBlue);
      t1.DrawLatex(0.6, 0.93, Form("IOV %s - %s ", lastIOVsince.c_str(), firstIOVsince.c_str()));

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef TrackerAlignmentCompare<AlignmentPI::t_x> TrackerAlignmentCompareX;
  typedef TrackerAlignmentCompare<AlignmentPI::t_y> TrackerAlignmentCompareY;
  typedef TrackerAlignmentCompare<AlignmentPI::t_z> TrackerAlignmentCompareZ;

  typedef TrackerAlignmentCompare<AlignmentPI::rot_alpha> TrackerAlignmentCompareAlpha;
  typedef TrackerAlignmentCompare<AlignmentPI::rot_beta> TrackerAlignmentCompareBeta;
  typedef TrackerAlignmentCompare<AlignmentPI::rot_gamma> TrackerAlignmentCompareGamma;

  //*******************************************//
  // Summary canvas per subdetector
  //******************************************//

  template <AlignmentPI::partitions q>
  class TrackerAlignmentSummary : public cond::payloadInspector::PlotImage<Alignments> {
  public:
    TrackerAlignmentSummary()
        : cond::payloadInspector::PlotImage<Alignments>("Comparison of all coordinates between two geometries for " +
                                                        getStringFromPart(q)) {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      std::vector<std::tuple<cond::Time_t, cond::Hash> > sorted_iovs = iovs;

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
      for (unsigned int i = 0; i <= ref_ali.size(); i++) {
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
              case AlignmentPI::rot_alpha:
                diffs[coord]->Fill((target_eulerAngles[0] - ref_eulerAngles[0]) * 1000.);
                break;
              case AlignmentPI::rot_beta:
                diffs[coord]->Fill((target_eulerAngles[1] - ref_eulerAngles[1]) * 1000.);
                break;
              case AlignmentPI::rot_gamma:
                diffs[coord]->Fill((target_eulerAngles[2] - ref_eulerAngles[2]) * 1000.);
                break;
              default:
                edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << coord << std::endl;
                break;
            }  // switch on the coordinate
          }
        }  // check on the same detID
      }    // loop on the components

      int c_index = 1;

      auto legend = std::unique_ptr<TLegend>(new TLegend(0.14, 0.93, 0.55, 0.98));
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
  class BPixBarycenterHistory : public cond::payloadInspector::HistoryPlot<Alignments, float> {
  public:
    BPixBarycenterHistory()
        : cond::payloadInspector::HistoryPlot<Alignments, float>(
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

}  // namespace

PAYLOAD_INSPECTOR_MODULE(TrackerAlignment) {
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareX);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareY);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareZ);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareAlpha);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareBeta);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentCompareGamma);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryBPix);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryFPix);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTIB);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTID);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTOB);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentSummaryTEC);
  PAYLOAD_INSPECTOR_CLASS(X_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(Y_BPixBarycenterHistory);
  PAYLOAD_INSPECTOR_CLASS(Z_BPixBarycenterHistory);
}
