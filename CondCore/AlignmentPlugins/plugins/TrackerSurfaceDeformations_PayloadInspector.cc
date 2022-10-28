/*!
  \file TrackerSurfaceDeformations_PayloadInspector
  \Payload Inspector Plugin for Tracker Surface Deformations
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/02/01 15:57:24 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/Alignment/interface/Definitions.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// needed for mapping
#include "CondCore/AlignmentPlugins/interface/AlignmentPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// for the pixel map
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TGaxis.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  using namespace cond::payloadInspector;

  class TrackerSurfaceDeformationsTest : public Histogram1D<AlignmentSurfaceDeformations, SINGLE_IOV> {
  public:
    TrackerSurfaceDeformationsTest()
        : Histogram1D<AlignmentSurfaceDeformations, SINGLE_IOV>(
              "TrackerSurfaceDeformationsTest", "TrackerSurfaceDeformationsTest", 2, 0.0, 2.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const &iov : tag.iovs) {
        std::shared_ptr<AlignmentSurfaceDeformations> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          int i = 0;
          auto listOfItems = payload->items();
          COUT << "items size:" << listOfItems.size() << std::endl;

          for (const auto &item : listOfItems) {
            COUT << i << " " << item.m_rawId << " Det: " << DetId(item.m_rawId).subdetId() << " " << item.m_index
                 << std::endl;
            const auto beginEndPair = payload->parameters(i);
            std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);
            COUT << "params.size()" << params.size() << std::endl;
            for (const auto &par : params) {
              COUT << par << std::endl;
            }
            i++;
          }

        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  //*******************************************************
  // Summary of the parameters for each partition for 1 IOV
  //*******************************************************

  template <AlignmentPI::partitions q>
  class TrackerAlignmentSurfaceDeformationsSummary : public PlotImage<AlignmentSurfaceDeformations, SINGLE_IOV> {
  public:
    TrackerAlignmentSurfaceDeformationsSummary()
        : PlotImage<AlignmentSurfaceDeformations, SINGLE_IOV>("Details for " + AlignmentPI::getStringFromPart(q)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<AlignmentSurfaceDeformations> payload = fetchPayload(std::get<1>(iov));
      auto listOfItems = payload->items();

      int canvas_w = (q <= 4) ? 1200 : 1800;
      std::pair<int, int> divisions = (q <= 4) ? std::make_pair(3, 1) : std::make_pair(7, 2);

      TCanvas canvas("Summary", "Summary", canvas_w, 600);
      canvas.Divide(divisions.first, divisions.second);

      std::array<std::unique_ptr<TH1F>, 14> summaries;
      for (int nPar = 0; nPar < 14; nPar++) {
        summaries[nPar] =
            std::make_unique<TH1F>(Form("h_summary_%i", nPar),
                                   Form("Surface Deformation parameter %i;parameter %i size;# modules", nPar, nPar),
                                   100,
                                   -0.1,
                                   0.1);
      }

      int i = 0;
      for (const auto &item : listOfItems) {
        int subid = DetId(item.m_rawId).subdetId();
        auto thePart = static_cast<AlignmentPI::partitions>(subid);

        const auto beginEndPair = payload->parameters(i);
        std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);

        // increase the counter before continuing if partition doesn't match,
        // otherwise the cound of the parameter count gets altered
        i++;

        // return if not the right partition
        if (thePart != q)
          continue;
        int nPar = 0;

        for (const auto &par : params) {
          summaries[nPar]->Fill(par);
          nPar++;
        }  // ends loop on the parameters
      }    // ends loop on the item vector

      TLatex t1;

      for (int c = 1; c <= (divisions.first * divisions.second); c++) {
        canvas.cd(c)->SetLogy();
        canvas.cd(c)->SetTopMargin(0.02);
        canvas.cd(c)->SetBottomMargin(0.15);
        canvas.cd(c)->SetLeftMargin(0.14);
        canvas.cd(c)->SetRightMargin(0.03);

        summaries[c - 1]->SetLineWidth(2);
        AlignmentPI::makeNicePlotStyle(summaries[c - 1].get(), kBlack);
        summaries[c - 1]->Draw("same");
        summaries[c - 1]->SetTitle("");

        AlignmentPI::makeNiceStats(summaries[c - 1].get(), q, kBlack);

        canvas.cd(c);

        t1.SetTextAlign(21);
        t1.SetTextSize(0.06);
        t1.SetTextColor(kBlue);
        t1.DrawLatexNDC(0.32, 0.95, Form("IOV: %s ", std::to_string(std::get<0>(iov)).c_str()));
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::BPix> BPixSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::FPix> FPixSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TIB> TIBSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TID> TIDSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TOB> TOBSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TEC> TECSurfaceDeformationsSummary;

  //*******************************************************
  // Comparison of the parameters for each partition for 1 IOV
  //*******************************************************

  template <AlignmentPI::partitions q>
  class TrackerAlignmentSurfaceDeformationsComparison : public PlotImage<AlignmentSurfaceDeformations, MULTI_IOV> {
  public:
    TrackerAlignmentSurfaceDeformationsComparison()
        : PlotImage<AlignmentSurfaceDeformations, MULTI_IOV>("Details for " + AlignmentPI::getStringFromPart(q)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto sorted_iovs = tag.iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<AlignmentSurfaceDeformations> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlignmentSurfaceDeformations> first_payload = fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      auto first_listOfItems = first_payload->items();
      auto last_listOfItems = last_payload->items();

      int canvas_w = (q <= 4) ? 1600 : 1800;
      std::pair<int, int> divisions = (q <= 4) ? std::make_pair(3, 1) : std::make_pair(7, 2);

      TCanvas canvas("Comparison", "Comparison", canvas_w, 600);
      canvas.Divide(divisions.first, divisions.second);

      std::array<std::unique_ptr<TH1F>, 14> deltas;
      for (int nPar = 0; nPar < 14; nPar++) {
        deltas[nPar] =
            std::make_unique<TH1F>(Form("h_summary_%i", nPar),
                                   Form("Surface Deformation #Delta parameter %i;#Deltapar_{%i};# modules", nPar, nPar),
                                   100,
                                   -0.05,
                                   0.05);
      }

      //assert(first_listOfItems.size() == last_listOfItems.size());

      for (unsigned int i = 0; i < first_listOfItems.size(); i++) {
        auto first_id = first_listOfItems[i].m_rawId;
        int subid = DetId(first_listOfItems[i].m_rawId).subdetId();
        auto thePart = static_cast<AlignmentPI::partitions>(subid);
        if (thePart != q)
          continue;

        const auto f_beginEndPair = first_payload->parameters(i);
        std::vector<align::Scalar> first_params(f_beginEndPair.first, f_beginEndPair.second);

        for (unsigned int j = 0; j < last_listOfItems.size(); j++) {
          auto last_id = last_listOfItems[j].m_rawId;
          if (first_id == last_id) {
            const auto l_beginEndPair = last_payload->parameters(j);
            std::vector<align::Scalar> last_params(l_beginEndPair.first, l_beginEndPair.second);

            assert(first_params.size() == last_params.size());

            for (unsigned int nPar = 0; nPar < first_params.size(); nPar++) {
              deltas[nPar]->Fill(last_params[nPar] - first_params[nPar]);
            }
            break;
          }
        }

      }  // ends loop on the item vector

      TLatex t1;

      for (int c = 1; c <= (divisions.first * divisions.second); c++) {
        canvas.cd(c)->SetLogy();
        canvas.cd(c)->SetTopMargin(0.015);
        canvas.cd(c)->SetBottomMargin(0.13);
        canvas.cd(c)->SetLeftMargin(0.14);
        canvas.cd(c)->SetRightMargin(0.03);

        deltas[c - 1]->SetLineWidth(2);
        AlignmentPI::makeNicePlotStyle(deltas[c - 1].get(), kBlack);
        deltas[c - 1]->Draw("same");
        deltas[c - 1]->SetTitle("");

        AlignmentPI::makeNiceStats(deltas[c - 1].get(), q, kBlack);

        canvas.cd(c);
        t1.SetTextAlign(21);
        t1.SetTextSize(0.045);
        t1.SetTextColor(kBlue);
        t1.DrawLatexNDC(0.4, 0.95, Form("#DeltaIOV: %s - %s ", lastIOVsince.c_str(), firstIOVsince.c_str()));

        if (deltas[c - 1]->GetEntries() == 0) {
          TLatex t2;
          t2.SetTextAlign(21);
          t2.SetTextSize(0.1);
          t2.SetTextAngle(45);
          t2.SetTextColor(kRed);
          t2.DrawLatexNDC(0.6, 0.50, "NO COMMON DETIDS");
        }
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::BPix> BPixSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::FPix> FPixSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TIB> TIBSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TID> TIDSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TOB> TOBSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TEC> TECSurfaceDeformationsComparison;

  // /************************************************
  //   TrackerMap of single parameter
  // *************************************************/
  template <unsigned int par>
  class SurfaceDeformationTrackerMap : public PlotImage<AlignmentSurfaceDeformations, SINGLE_IOV> {
  public:
    SurfaceDeformationTrackerMap()
        : PlotImage<AlignmentSurfaceDeformations, SINGLE_IOV>(
              "Tracker Map of Tracker Surface deformations - parameter: " + std::to_string(par)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<AlignmentSurfaceDeformations> payload = fetchPayload(std::get<1>(iov));
      auto listOfItems = payload->items();

      std::string titleMap =
          "Surface deformation parameter " + std::to_string(par) + " value (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("Surface Deformations");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::map<unsigned int, float> surfDefMap;

      bool isPhase0(false);
      if (listOfItems.size() == AlignmentPI::phase0size)
        isPhase0 = true;

      int iDet = 0;
      for (const auto &item : listOfItems) {
        // fill the tracker map
        int subid = DetId(item.m_rawId).subdetId();

        if (DetId(item.m_rawId).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << item.m_rawId << " - terminating ";
          return false;
        }

        const auto beginEndPair = payload->parameters(iDet);
        std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);

        iDet++;
        // protect against exceeding the vector of parameter size
        if (par >= params.size())
          continue;

        if (isPhase0) {
          tmap->addPixel(true);
          tmap->fill(item.m_rawId, params.at(par));
          surfDefMap[item.m_rawId] = params.at(par);
        } else {
          if (subid != 1 && subid != 2) {
            tmap->fill(item.m_rawId, params.at(par));
            surfDefMap[item.m_rawId] = params.at(par);
          }
        }
      }  // loop over detIds

      //=========================

      // saturate at 1.5sigma
      auto autoRange = AlignmentPI::getTheRange(surfDefMap, 1.5);  //tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);
      // protect against uniform values (Surface deformations are defined positive)
      if (autoRange.first != autoRange.second) {
        tmap->save(true, autoRange.first, autoRange.second, fileName);
      } else {
        if (autoRange.first == 0.) {
          tmap->save(true, 0., 1., fileName);
        } else {
          tmap->save(true, autoRange.first, autoRange.second, fileName);
        }
      }

      return true;
    }
  };

  typedef SurfaceDeformationTrackerMap<0> SurfaceDeformationParameter0TrackerMap;
  typedef SurfaceDeformationTrackerMap<1> SurfaceDeformationParameter1TrackerMap;
  typedef SurfaceDeformationTrackerMap<2> SurfaceDeformationParameter2TrackerMap;
  typedef SurfaceDeformationTrackerMap<3> SurfaceDeformationParameter3TrackerMap;
  typedef SurfaceDeformationTrackerMap<4> SurfaceDeformationParameter4TrackerMap;
  typedef SurfaceDeformationTrackerMap<5> SurfaceDeformationParameter5TrackerMap;
  typedef SurfaceDeformationTrackerMap<6> SurfaceDeformationParameter6TrackerMap;
  typedef SurfaceDeformationTrackerMap<7> SurfaceDeformationParameter7TrackerMap;
  typedef SurfaceDeformationTrackerMap<8> SurfaceDeformationParameter8TrackerMap;
  typedef SurfaceDeformationTrackerMap<9> SurfaceDeformationParameter9TrackerMap;
  typedef SurfaceDeformationTrackerMap<10> SurfaceDeformationParameter10TrackerMap;
  typedef SurfaceDeformationTrackerMap<11> SurfaceDeformationParameter11TrackerMap;
  typedef SurfaceDeformationTrackerMap<12> SurfaceDeformationParameter12TrackerMap;

  // /************************************************
  //   TrackerMap of single parameter
  // *************************************************/
  template <unsigned int par>
  class SurfaceDeformationPixelMap : public PlotImage<AlignmentSurfaceDeformations, SINGLE_IOV> {
  public:
    SurfaceDeformationPixelMap()
        : PlotImage<AlignmentSurfaceDeformations, SINGLE_IOV>(
              "Tracker Map of Tracker Surface deformations - parameter: " + std::to_string(par)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<AlignmentSurfaceDeformations> payload = fetchPayload(std::get<1>(iov));
      auto listOfItems = payload->items();

      TCanvas canvas("Canv", "Canv", 1400, 1000);
      Phase1PixelSummaryMap fullMap("", "Surface deformation parameter " + std::to_string(par), "");
      fullMap.createTrackerBaseMap();

      std::map<unsigned int, float> surfDefMap;

      bool isPhase0(false);
      if (listOfItems.size() == AlignmentPI::phase0size) {
        isPhase0 = true;
      }

      int iDet = 0;
      for (const auto &item : listOfItems) {
        // fill the tracker map
        int subid = DetId(item.m_rawId).subdetId();

        if (DetId(item.m_rawId).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << item.m_rawId << " - terminating ";
          return false;
        }

        const auto beginEndPair = payload->parameters(iDet);
        std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);

        iDet++;
        // protect against exceeding the vector of parameter size
        if (par >= params.size())
          continue;

        if (isPhase0) {
          surfDefMap[item.m_rawId] = params.at(par);
        } else {
          if (subid == 1 || subid == 2) {
            fullMap.fillTrackerMap(item.m_rawId, params.at(par));
            surfDefMap[item.m_rawId] = params.at(par);
          }
        }
      }  // loop over detIds

      //=========================

      if (surfDefMap.empty()) {
        edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector") << "No common DetIds have been found!!! ";
      }

      // protect against uniform values (Surface deformations are defined positive)
      fullMap.printTrackerMap(canvas);

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef SurfaceDeformationPixelMap<0> SurfaceDeformationParameter0PixelMap;
  typedef SurfaceDeformationPixelMap<1> SurfaceDeformationParameter1PixelMap;
  typedef SurfaceDeformationPixelMap<2> SurfaceDeformationParameter2PixelMap;
  typedef SurfaceDeformationPixelMap<3> SurfaceDeformationParameter3PixelMap;
  typedef SurfaceDeformationPixelMap<4> SurfaceDeformationParameter4PixelMap;
  typedef SurfaceDeformationPixelMap<5> SurfaceDeformationParameter5PixelMap;
  typedef SurfaceDeformationPixelMap<6> SurfaceDeformationParameter6PixelMap;
  typedef SurfaceDeformationPixelMap<7> SurfaceDeformationParameter7PixelMap;
  typedef SurfaceDeformationPixelMap<8> SurfaceDeformationParameter8PixelMap;
  typedef SurfaceDeformationPixelMap<9> SurfaceDeformationParameter9PixelMap;
  typedef SurfaceDeformationPixelMap<10> SurfaceDeformationParameter10PixelMap;
  typedef SurfaceDeformationPixelMap<11> SurfaceDeformationParameter11PixelMap;
  typedef SurfaceDeformationPixelMap<12> SurfaceDeformationParameter12PixelMap;

  // /************************************************
  //   TrackerMap of delta for a single parameter
  // *************************************************/
  template <unsigned int m_par>
  class SurfaceDeformationsTkMapDelta : public PlotImage<AlignmentSurfaceDeformations, MULTI_IOV> {
  public:
    SurfaceDeformationsTkMapDelta()
        : PlotImage<AlignmentSurfaceDeformations, MULTI_IOV>(
              "Tracker Map of Tracker Surface deformations differences - parameter: " + std::to_string(m_par)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto sorted_iovs = tag.iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<AlignmentSurfaceDeformations> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlignmentSurfaceDeformations> first_payload = fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      auto first_listOfItems = first_payload->items();
      auto last_listOfItems = last_payload->items();

      std::string titleMap = "#Delta Surface deformation parameter " + std::to_string(m_par) +
                             " (IOV : " + std::to_string(std::get<0>(lastiov)) + "- " +
                             std::to_string(std::get<0>(firstiov)) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("Surface Deformations #Delta");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::map<unsigned int, float> f_paramsMap;
      std::map<unsigned int, float> l_paramsMap;
      std::map<unsigned int, float> surfDefMap;

      // check the payload sizes are matched
      if (first_listOfItems.size() != last_listOfItems.size()) {
        edm::LogInfo("TrackerSurfaceDeformations_PayloadInspector")
            << "(" << firstIOVsince << ") has " << first_listOfItems.size() << " DetIds - (" << lastIOVsince << ") has "
            << last_listOfItems.size() << " DetIds" << std::endl;
      };

      bool isPhase0(false);
      if (first_listOfItems.size() <= AlignmentPI::phase0size)
        isPhase0 = true;
      if (isPhase0)
        tmap->addPixel(true);

      // loop on the first payload
      int iDet = 0;
      for (const auto &f_item : first_listOfItems) {
        auto first_id = f_item.m_rawId;

        if (DetId(first_id).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << first_id << " - terminating ";
          return false;
        }

        const auto f_beginEndPair = first_payload->parameters(iDet);
        std::vector<align::Scalar> first_params(f_beginEndPair.first, f_beginEndPair.second);

        iDet++;
        // protect against exceeding the vector of parameter size
        if (m_par >= first_params.size())
          continue;

        f_paramsMap[first_id] = first_params.at(m_par);
      }

      // loop on the second payload
      int jDet = 0;
      for (const auto &l_item : last_listOfItems) {
        auto last_id = l_item.m_rawId;

        if (DetId(last_id).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << last_id << " - terminating ";
          return false;
        }

        const auto l_beginEndPair = last_payload->parameters(jDet);
        std::vector<align::Scalar> last_params(l_beginEndPair.first, l_beginEndPair.second);

        jDet++;
        // protect against exceeding the vector of parameter size
        if (m_par >= last_params.size())
          continue;

        l_paramsMap[last_id] = last_params.at(m_par);
      }

      // fill the tk map
      for (const auto &f_entry : f_paramsMap) {
        for (const auto &l_entry : l_paramsMap) {
          if (f_entry.first != l_entry.first)
            continue;

          int subid = DetId(f_entry.first).subdetId();

          float delta = (l_entry.second - f_entry.second);

          //COUT<<" match! subid:" << subid << " rawId:" << f_entry.first << " delta:"<< delta << std::endl;

          if (isPhase0) {
            tmap->addPixel(true);
            tmap->fill(f_entry.first, delta);
            surfDefMap[f_entry.first] = delta;
          } else {
            // fill pixel map only for phase-0 (in lack of a dedicate phase-I map)
            if (subid != 1 && subid != 2) {
              tmap->fill(f_entry.first, delta);
              surfDefMap[f_entry.first] = delta;
            }
          }  // if not phase-0
        }    // loop on the last payload map
      }      // loop on the first payload map

      //=========================

      if (surfDefMap.empty()) {
        edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector") << "No common DetIds have been found!!! ";
        tmap->fillc_all_blank();
        tmap->setTitle("NO COMMON DETIDS (IOV : " + std::to_string(std::get<0>(lastiov)) + "- " +
                       std::to_string(std::get<0>(firstiov)) + ")");
      }

      // saturate at 1.5sigma
      auto autoRange = AlignmentPI::getTheRange(surfDefMap, 1.5);  //tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);
      // protect against uniform values (Surface deformations are defined positive)
      if (autoRange.first != autoRange.second) {
        tmap->save(true, autoRange.first, autoRange.second, fileName);
      } else {
        if (autoRange.first == 0.) {
          tmap->save(true, 0., 1., fileName);
        } else {
          tmap->save(true, autoRange.first, autoRange.second, fileName);
        }
      }

      return true;
    }
  };

  typedef SurfaceDeformationsTkMapDelta<0> SurfaceDeformationParameter0TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<1> SurfaceDeformationParameter1TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<2> SurfaceDeformationParameter2TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<3> SurfaceDeformationParameter3TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<4> SurfaceDeformationParameter4TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<5> SurfaceDeformationParameter5TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<6> SurfaceDeformationParameter6TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<7> SurfaceDeformationParameter7TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<8> SurfaceDeformationParameter8TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<9> SurfaceDeformationParameter9TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<10> SurfaceDeformationParameter10TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<11> SurfaceDeformationParameter11TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<12> SurfaceDeformationParameter12TkMapDelta;

  // /************************************************
  //   TrackerMap of delta for a single parameter
  // *************************************************/
  template <unsigned int m_par>
  class SurfaceDeformationsPXMapDelta : public PlotImage<AlignmentSurfaceDeformations, MULTI_IOV> {
  public:
    SurfaceDeformationsPXMapDelta()
        : PlotImage<AlignmentSurfaceDeformations, MULTI_IOV>(
              "Tracker Map of Tracker Surface deformations differences - parameter: " + std::to_string(m_par)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto sorted_iovs = tag.iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<AlignmentSurfaceDeformations> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlignmentSurfaceDeformations> first_payload = fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      auto first_listOfItems = first_payload->items();
      auto last_listOfItems = last_payload->items();

      std::string titleMap = "#Delta Surface deformation parameter " + std::to_string(m_par) +
                             " (IOV : " + std::to_string(std::get<0>(lastiov)) + "- " +
                             std::to_string(std::get<0>(firstiov)) + ")";

      TCanvas canvas("Canv", "Canv", 1400, 1000);
      Phase1PixelSummaryMap fullMap("", "#Delta Surface deformation parameter " + std::to_string(m_par), "");
      fullMap.createTrackerBaseMap();

      std::map<unsigned int, float> f_paramsMap;
      std::map<unsigned int, float> l_paramsMap;
      std::map<unsigned int, float> surfDefMap;

      // check the payload sizes are matched
      if (first_listOfItems.size() != last_listOfItems.size()) {
        edm::LogInfo("TrackerSurfaceDeformations_PayloadInspector")
            << "(" << firstIOVsince << ") has " << first_listOfItems.size() << " DetIds - (" << lastIOVsince << ") has "
            << last_listOfItems.size() << " DetIds" << std::endl;
      };

      bool isPhase0(false);
      if (first_listOfItems.size() <= AlignmentPI::phase0size) {
        isPhase0 = true;
      }

      // loop on the first payload
      int iDet = 0;
      for (const auto &f_item : first_listOfItems) {
        auto first_id = f_item.m_rawId;

        if (DetId(first_id).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << first_id << " - terminating ";
          return false;
        }

        const auto f_beginEndPair = first_payload->parameters(iDet);
        std::vector<align::Scalar> first_params(f_beginEndPair.first, f_beginEndPair.second);

        iDet++;
        // protect against exceeding the vector of parameter size
        if (m_par >= first_params.size())
          continue;

        f_paramsMap[first_id] = first_params.at(m_par);
      }

      // loop on the second payload
      int jDet = 0;
      for (const auto &l_item : last_listOfItems) {
        auto last_id = l_item.m_rawId;

        if (DetId(last_id).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << last_id << " - terminating ";
          return false;
        }

        const auto l_beginEndPair = last_payload->parameters(jDet);
        std::vector<align::Scalar> last_params(l_beginEndPair.first, l_beginEndPair.second);

        jDet++;
        // protect against exceeding the vector of parameter size
        if (m_par >= last_params.size())
          continue;

        l_paramsMap[last_id] = last_params.at(m_par);
      }

      if (f_paramsMap.empty() || l_paramsMap.empty()) {
        edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
            << " One or more of the paylods is null \n"
            << " Cannot perform the comparison." << std::endl;
        return false;
      }

      // fill the tk map
      for (const auto &f_entry : f_paramsMap) {
        for (const auto &l_entry : l_paramsMap) {
          if (f_entry.first != l_entry.first)
            continue;

          int subid = DetId(f_entry.first).subdetId();
          float delta = (l_entry.second - f_entry.second);

          if (isPhase0) {
            surfDefMap[f_entry.first] = delta;
          } else {
            // fill pixel map only for phase-0 (in lack of a dedicate phase-I map)
            if (subid == 1 || subid == 2) {
              fullMap.fillTrackerMap(f_entry.first, delta);
              surfDefMap[f_entry.first] = delta;
            }
          }  // if not phase-0
        }    // loop on the last payload map
      }      // loop on the first payload map

      //=========================

      if (surfDefMap.empty()) {
        edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector") << "No common DetIds have been found!!! ";
      }

      fullMap.printTrackerMap(canvas);

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef SurfaceDeformationsPXMapDelta<0> SurfaceDeformationParameter0PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<1> SurfaceDeformationParameter1PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<2> SurfaceDeformationParameter2PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<3> SurfaceDeformationParameter3PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<4> SurfaceDeformationParameter4PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<5> SurfaceDeformationParameter5PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<6> SurfaceDeformationParameter6PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<7> SurfaceDeformationParameter7PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<8> SurfaceDeformationParameter8PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<9> SurfaceDeformationParameter9PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<10> SurfaceDeformationParameter10PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<11> SurfaceDeformationParameter11PXMapDelta;
  typedef SurfaceDeformationsPXMapDelta<12> SurfaceDeformationParameter12PXMapDelta;

  // /************************************************
  //  Tracker Surface Deformations grand summary comparison of 2 IOVs
  // *************************************************/

  template <unsigned int m_par>
  class TrackerSurfaceDeformationsComparator : public PlotImage<AlignmentSurfaceDeformations, MULTI_IOV> {
  public:
    TrackerSurfaceDeformationsComparator()
        : PlotImage<AlignmentSurfaceDeformations, MULTI_IOV>("Summary per Tracker region of parameter " +
                                                             std::to_string(m_par) + " of Surface Deformations") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto sorted_iovs = tag.iovs;

      TGaxis::SetMaxDigits(3);
      gStyle->SetPaintTextFormat(".1f");

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
        return std::get<0>(t1) < std::get<0>(t2);
      });

      auto firstiov = sorted_iovs.front();
      auto lastiov = sorted_iovs.back();

      std::shared_ptr<AlignmentSurfaceDeformations> last_payload = fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlignmentSurfaceDeformations> first_payload = fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      auto first_listOfItems = first_payload->items();
      auto last_listOfItems = last_payload->items();

      TCanvas canvas("Comparison", "Comparison", 1600, 800);

      std::map<AlignmentPI::regions, std::shared_ptr<TH1F> > FirstSurfDef_spectraByRegion;
      std::map<AlignmentPI::regions, std::shared_ptr<TH1F> > LastSurfDef_spectraByRegion;
      std::shared_ptr<TH1F> summaryFirst;
      std::shared_ptr<TH1F> summaryLast;

      // book the intermediate histograms
      for (int r = AlignmentPI::BPixL1o; r != AlignmentPI::StripDoubleSide; r++) {
        AlignmentPI::regions part = static_cast<AlignmentPI::regions>(r);
        std::string s_part = AlignmentPI::getStringFromRegionEnum(part);

        FirstSurfDef_spectraByRegion[part] =
            std::make_shared<TH1F>(Form("hfirstSurfDef_%i_%s", m_par, s_part.c_str()),
                                   Form(";%s SurfDef parameter %i;n. of modules", s_part.c_str(), m_par),
                                   10000,
                                   -1,
                                   1.);
        LastSurfDef_spectraByRegion[part] =
            std::make_shared<TH1F>(Form("hlastSurfDef_%i_%s", m_par, s_part.c_str()),
                                   Form(";%s SurfDef parameter %i;n. of modules", s_part.c_str(), m_par),
                                   10000,
                                   -1.,
                                   1.);
      }

      summaryFirst = std::make_shared<TH1F>(
          Form("first Summary_%i", m_par),
          Form("Summary for parameter %i Surface Deformation;;Surface Deformation paramter %i", m_par, m_par),
          FirstSurfDef_spectraByRegion.size(),
          0,
          FirstSurfDef_spectraByRegion.size());
      summaryLast = std::make_shared<TH1F>(
          Form("last Summary_%i", m_par),
          Form("Summary for parameter %i Surface Deformation;;Surface Deformation paramter %i", m_par, m_par),
          LastSurfDef_spectraByRegion.size(),
          0,
          LastSurfDef_spectraByRegion.size());

      // N.B. <= and not == because the list of surface deformations might not include all tracker modules
      const char *path_toTopologyXML = (first_listOfItems.size() <= AlignmentPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      const char *alternative_path_toTopologyXML = (first_listOfItems.size() <= AlignmentPI::phase0size)
                                                       ? "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml"
                                                       : "Geometry/TrackerCommonData/data/trackerParameters.xml";
      TrackerTopology f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());
      TrackerTopology af_tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
          edm::FileInPath(alternative_path_toTopologyXML).fullPath());

      bool isPhase0(false);
      if (first_listOfItems.size() <= AlignmentPI::phase0size)
        isPhase0 = true;

      // -------------------------------------------------------------------
      // loop on the first vector of errors
      // -------------------------------------------------------------------
      int iDet = 0;
      for (const auto &it : first_listOfItems) {
        if (DetId(it.m_rawId).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.m_rawId << " - terminating ";
          return false;
        }

        AlignmentPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.m_rawId);
        t_info_fromXML.fillGeometryInfo(detid, f_tTopo, isPhase0);

        //COUT<<"sanityCheck: "<< t_info_fromXML.sanityCheck() << std::endl;

        if (!t_info_fromXML.sanityCheck()) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Wrong choice of Tracker Topology encountered for DetId:" << it.m_rawId << " ---> changing";
          t_info_fromXML.init();
          t_info_fromXML.fillGeometryInfo(detid, af_tTopo, !isPhase0);
        }

        //t_info_fromXML.printAll();

        AlignmentPI::regions thePart = t_info_fromXML.filterThePartition();

        // skip the glued detector detIds
        if (thePart == AlignmentPI::StripDoubleSide)
          continue;

        const auto f_beginEndPair = first_payload->parameters(iDet);
        std::vector<align::Scalar> first_params(f_beginEndPair.first, f_beginEndPair.second);

        iDet++;
        // protect against exceeding the vector of parameter size
        if (m_par >= first_params.size())
          continue;

        FirstSurfDef_spectraByRegion[thePart]->Fill(first_params.at(m_par));
        //COUT<<  getStringFromRegionEnum(thePart) << " first payload: "<< first_params.at(m_par) << std::endl;

      }  // ends loop on the vector of error transforms

      // N.B. <= and not == because the list of surface deformations might not include all tracker modules
      path_toTopologyXML = (last_listOfItems.size() <= AlignmentPI::phase0size)
                               ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                               : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      alternative_path_toTopologyXML = (first_listOfItems.size() <= AlignmentPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/trackerParameters.xml";
      TrackerTopology l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());
      TrackerTopology al_tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(
          edm::FileInPath(alternative_path_toTopologyXML).fullPath());

      if (last_listOfItems.size() <= AlignmentPI::phase0size)
        isPhase0 = true;

      // -------------------------------------------------------------------
      // loop on the second vector of errors
      // -------------------------------------------------------------------
      int jDet = 0;
      for (const auto &it : last_listOfItems) {
        if (DetId(it.m_rawId).det() != DetId::Tracker) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.m_rawId << " - terminating ";
          return false;
        }

        AlignmentPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.m_rawId);
        t_info_fromXML.fillGeometryInfo(detid, l_tTopo, isPhase0);

        //COUT<<"sanityCheck: "<< t_info_fromXML.sanityCheck() << std::endl;

        if (!t_info_fromXML.sanityCheck()) {
          edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector")
              << "Wrong choice of Tracker Topology encountered for DetId:" << it.m_rawId << " ---> changing";
          t_info_fromXML.init();
          isPhase0 = !isPhase0;
          t_info_fromXML.fillGeometryInfo(detid, al_tTopo, !isPhase0);
        }

        //t_info_fromXML.printAll();

        AlignmentPI::regions thePart = t_info_fromXML.filterThePartition();

        // skip the glued detector detIds
        if (thePart == AlignmentPI::StripDoubleSide)
          continue;

        const auto l_beginEndPair = last_payload->parameters(jDet);
        std::vector<align::Scalar> last_params(l_beginEndPair.first, l_beginEndPair.second);

        jDet++;
        // protect against exceeding the vector of parameter size
        if (m_par >= last_params.size())
          continue;

        LastSurfDef_spectraByRegion[thePart]->Fill(last_params.at(m_par));
        //COUT<< getStringFromRegionEnum(thePart) <<  " last payload: "<< last_params.at(m_par) << std::endl;

      }  // ends loop on the vector of error transforms

      // fill the summary plots
      int bin = 1;
      for (int r = AlignmentPI::BPixL1o; r != AlignmentPI::StripDoubleSide; r++) {
        AlignmentPI::regions part = static_cast<AlignmentPI::regions>(r);

        summaryFirst->GetXaxis()->SetBinLabel(bin, AlignmentPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float f_mean = FirstSurfDef_spectraByRegion[part]->GetMean();
        summaryFirst->SetBinContent(bin, f_mean);
        //summaryFirst->SetBinError(bin, FirstSurfDef_spectraByRegion[part]->GetRMS());

        summaryLast->GetXaxis()->SetBinLabel(bin, AlignmentPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float l_mean = LastSurfDef_spectraByRegion[part]->GetMean();
        summaryLast->SetBinContent(bin, l_mean);
        //summaryLast->SetBinError(bin,LastSurfDef_spectraByRegion[part]->GetRMS());
        bin++;
      }

      AlignmentPI::makeNicePlotStyle(summaryFirst.get(), kBlue);
      summaryFirst->SetMarkerColor(kBlue);
      summaryFirst->GetXaxis()->LabelsOption("v");
      summaryFirst->GetXaxis()->SetLabelSize(0.05);
      summaryFirst->GetYaxis()->SetTitleOffset(0.9);

      AlignmentPI::makeNicePlotStyle(summaryLast.get(), kRed);
      summaryLast->SetMarkerColor(kRed);
      summaryLast->GetYaxis()->SetTitleOffset(0.9);
      summaryLast->GetXaxis()->LabelsOption("v");
      summaryLast->GetXaxis()->SetLabelSize(0.05);

      canvas.cd()->SetGridy();

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.02);
      canvas.Modified();

      summaryFirst->SetFillColor(kBlue);
      summaryLast->SetFillColor(kRed);

      summaryFirst->SetBarWidth(0.45);
      summaryFirst->SetBarOffset(0.1);

      summaryLast->SetBarWidth(0.4);
      summaryLast->SetBarOffset(0.55);

      float max = (summaryFirst->GetMaximum() > summaryLast->GetMaximum()) ? summaryFirst->GetMaximum()
                                                                           : summaryLast->GetMaximum();
      float min = (summaryFirst->GetMinimum() < summaryLast->GetMinimum()) ? summaryFirst->GetMinimum()
                                                                           : summaryLast->GetMinimum();

      summaryFirst->GetYaxis()->SetRangeUser(min * 1.20, max * 1.40);
      summaryFirst->Draw("b");
      //summaryFirst->Draw("text90same");
      summaryLast->Draw("b,same");
      //summaryLast->Draw("text180same");

      TLegend legend = TLegend(0.52, 0.82, 0.98, 0.9);
      legend.SetHeader(("Surface Deformation par " + std::to_string(m_par) + " comparison").c_str(),
                       "C");  // option "C" allows to center the header
      legend.AddEntry(
          summaryLast.get(),
          ("IOV:  #scale[1.2]{" + std::to_string(std::get<0>(lastiov)) + "} | #color[2]{" + std::get<1>(lastiov) + "}")
              .c_str(),
          "F");
      legend.AddEntry(summaryFirst.get(),
                      ("IOV:  #scale[1.2]{" + std::to_string(std::get<0>(firstiov)) + "} | #color[4]{" +
                       std::get<1>(firstiov) + "}")
                          .c_str(),
                      "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;

    }  // ends fill method
  };

  typedef TrackerSurfaceDeformationsComparator<0> TrackerSurfaceDeformationsPar0Comparator;
  typedef TrackerSurfaceDeformationsComparator<1> TrackerSurfaceDeformationsPar1Comparator;
  typedef TrackerSurfaceDeformationsComparator<2> TrackerSurfaceDeformationsPar2Comparator;
  typedef TrackerSurfaceDeformationsComparator<3> TrackerSurfaceDeformationsPar3Comparator;
  typedef TrackerSurfaceDeformationsComparator<4> TrackerSurfaceDeformationsPar4Comparator;
  typedef TrackerSurfaceDeformationsComparator<5> TrackerSurfaceDeformationsPar5Comparator;
  typedef TrackerSurfaceDeformationsComparator<6> TrackerSurfaceDeformationsPar6Comparator;
  typedef TrackerSurfaceDeformationsComparator<7> TrackerSurfaceDeformationsPar7Comparator;
  typedef TrackerSurfaceDeformationsComparator<8> TrackerSurfaceDeformationsPar8Comparator;
  typedef TrackerSurfaceDeformationsComparator<9> TrackerSurfaceDeformationsPar9Comparator;
  typedef TrackerSurfaceDeformationsComparator<10> TrackerSurfaceDeformationsPar10Comparator;
  typedef TrackerSurfaceDeformationsComparator<11> TrackerSurfaceDeformationsPar11Comparator;
  typedef TrackerSurfaceDeformationsComparator<12> TrackerSurfaceDeformationsPar12Comparator;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(TrackerSurfaceDeformations) {
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsTest);
  PAYLOAD_INSPECTOR_CLASS(BPixSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(FPixSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(TIBSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(TIDSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(TOBSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(TECSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(BPixSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(FPixSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(TIBSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(TIDSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(TOBSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(TECSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter0TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter1TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter2TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter3TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter4TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter5TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter6TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter7TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter8TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter9TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter10TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter11TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter12TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter0PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter1PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter2PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter3PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter4PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter5PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter6PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter7PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter8PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter9PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter10PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter11PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter12PixelMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter0TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter1TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter2TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter3TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter4TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter5TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter6TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter7TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter8TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter9TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter10TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter11TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter12TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter0PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter1PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter2PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter3PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter4PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter5PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter6PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter7PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter8PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter9PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter10PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter11PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter12PXMapDelta);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar0Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar1Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar2Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar3Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar4Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar5Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar6Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar7Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar8Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar9Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar10Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar11Comparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsPar12Comparator);
}
