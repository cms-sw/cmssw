/*!
  \file TrackerAlignmentErrorExtended_PayloadInspector
  \Payload Inspector Plugin for Tracker Alignment Errors (APE)
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/07/10 10:59:24 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

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

  using namespace cond::payloadInspector;

  const std::map<AlignmentPI::partitions, std::pair<AlignmentPI::regions, AlignmentPI::regions> > partLimits = {
      {AlignmentPI::BPix, std::make_pair(AlignmentPI::BPixL1o, AlignmentPI::BPixL4i)},
      {AlignmentPI::FPix, std::make_pair(AlignmentPI::FPixmL1, AlignmentPI::FPixpL3)},
      {AlignmentPI::TIB, std::make_pair(AlignmentPI::TIBL1Ro, AlignmentPI::TIBL4i)},
      {AlignmentPI::TOB, std::make_pair(AlignmentPI::TOBL1Ro, AlignmentPI::TOBL6i)},
      {AlignmentPI::TID, std::make_pair(AlignmentPI::TIDmR1R, AlignmentPI::TIDpR3)},
      {AlignmentPI::TEC, std::make_pair(AlignmentPI::TECmR1R, AlignmentPI::TECpR7)}};

  /************************************************
    1d histogram of sqrt(d_ii) of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  template <AlignmentPI::index i>
  class TrackerAlignmentErrorExtendedValue : public Histogram1D<AlignmentErrorsExtended, SINGLE_IOV> {
  public:
    TrackerAlignmentErrorExtendedValue()
        : Histogram1D<AlignmentErrorsExtended, SINGLE_IOV>(
              "TrackerAlignmentErrorExtendedValue",
              "TrackerAlignmentErrorExtendedValue sqrt(d_{" + getStringFromIndex(i) + "})",
              500,
              0.0,
              500.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<AlignmentErrorsExtended> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;
          auto indices = AlignmentPI::getIndices(i);

          for (const auto& it : alignErrors) {
            CLHEP::HepSymMatrix errMatrix = it.matrix();

            if (DetId(it.rawId()).det() != DetId::Tracker) {
              edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector")
                  << "Encountered invalid Tracker DetId:" << it.rawId() << " - terminating ";
              return false;
            }

            // to be used to fill the histogram
            fillWithValue(sqrt(errMatrix[indices.first][indices.second]) * AlignmentPI::cmToUm);
          }  // loop on the vector of modules
        }    // payload
      }      // iovs
      return true;
    }  // fill
  };

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXValue;
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYValue;
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZValue;

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYValue;
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZValue;
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZValue;

  // /************************************************
  //   Summary spectra of sqrt(d_ii) of 1 IOV
  // *************************************************/

  template <AlignmentPI::index i>
  class TrackerAlignmentErrorExtendedSummary : public PlotImage<AlignmentErrorsExtended, SINGLE_IOV> {
  public:
    TrackerAlignmentErrorExtendedSummary()
        : PlotImage<AlignmentErrorsExtended, SINGLE_IOV>("Summary per Tracker Partition of sqrt(d_{" +
                                                         getStringFromIndex(i) + "}) of APE matrix") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<AlignmentErrorsExtended> payload = fetchPayload(std::get<1>(iov));
      std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;

      const char* path_toTopologyXML = (alignErrors.size() == AlignmentPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      auto indices = AlignmentPI::getIndices(i);

      TCanvas canvas("Partion summary", "partition summary", 1200, 1000);
      canvas.Divide(3, 2);
      std::map<AlignmentPI::partitions, int> colormap;
      colormap[AlignmentPI::BPix] = kBlue;
      colormap[AlignmentPI::FPix] = kBlue + 2;
      colormap[AlignmentPI::TIB] = kRed;
      colormap[AlignmentPI::TOB] = kRed + 2;
      colormap[AlignmentPI::TID] = kRed + 4;
      colormap[AlignmentPI::TEC] = kRed + 6;

      std::map<AlignmentPI::partitions, std::shared_ptr<TH1F> > APE_spectra;
      std::vector<AlignmentPI::partitions> parts = {
          AlignmentPI::BPix, AlignmentPI::FPix, AlignmentPI::TIB, AlignmentPI::TID, AlignmentPI::TOB, AlignmentPI::TEC};

      auto s_index = getStringFromIndex(i);

      for (const auto& part : parts) {
        std::string s_part = AlignmentPI::getStringFromPart(part);

        APE_spectra[part] =
            std::make_shared<TH1F>(Form("hAPE_%s", s_part.c_str()),
                                   Form(";%s APE #sqrt{d_{%s}} [#mum];n. of modules", s_part.c_str(), s_index.c_str()),
                                   200,
                                   -10.,
                                   200.);
      }

      for (const auto& it : alignErrors) {
        CLHEP::HepSymMatrix errMatrix = it.matrix();
        int subid = DetId(it.rawId()).subdetId();
        double matrixElement = sqrt(errMatrix[indices.first][indices.second]);

        if (DetId(it.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.rawId() << " - terminating ";
          return false;
        }

        switch (subid) {
          case 1:
            APE_spectra[AlignmentPI::BPix]->Fill(std::min(200., matrixElement) * AlignmentPI::cmToUm);
            break;
          case 2:
            APE_spectra[AlignmentPI::FPix]->Fill(std::min(200., matrixElement) * AlignmentPI::cmToUm);
            break;
          case 3:
            if (!tTopo.tibIsDoubleSide(it.rawId())) {  // no glued DetIds
              APE_spectra[AlignmentPI::TIB]->Fill(std::min(200., matrixElement) * AlignmentPI::cmToUm);
            }
            break;
          case 4:
            if (!tTopo.tidIsDoubleSide(it.rawId())) {  // no glued DetIds
              APE_spectra[AlignmentPI::TID]->Fill(std::min(200., matrixElement) * AlignmentPI::cmToUm);
            }
            break;
          case 5:
            if (!tTopo.tobIsDoubleSide(it.rawId())) {  // no glued DetIds
              APE_spectra[AlignmentPI::TOB]->Fill(std::min(200., matrixElement) * AlignmentPI::cmToUm);
            }
            break;
          case 6:
            if (!tTopo.tecIsDoubleSide(it.rawId())) {  // no glued DetIds
              APE_spectra[AlignmentPI::TEC]->Fill(std::min(200., matrixElement) * AlignmentPI::cmToUm);
            }
            break;
          default:
            COUT << "will do nothing" << std::endl;
            break;
        }
      }

      TLatex t1;
      t1.SetTextAlign(21);
      t1.SetTextSize(0.07);
      t1.SetTextColor(kBlue);

      int c_index = 1;
      for (const auto& part : parts) {
        canvas.cd(c_index)->SetLogy();
        canvas.cd(c_index)->SetTopMargin(0.02);
        canvas.cd(c_index)->SetBottomMargin(0.15);
        canvas.cd(c_index)->SetLeftMargin(0.14);
        canvas.cd(c_index)->SetRightMargin(0.04);
        APE_spectra[part]->SetLineWidth(2);
        AlignmentPI::makeNicePlotStyle(APE_spectra[part].get(), colormap[part]);
        APE_spectra[part]->Draw("HIST");
        AlignmentPI::makeNiceStats(APE_spectra[part].get(), part, colormap[part]);

        t1.DrawLatexNDC(0.35, 0.92, ("IOV: " + std::to_string(std::get<0>(iov))).c_str());

        c_index++;
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXSummary;
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYSummary;
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZSummary;

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYSummary;
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZSummary;
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZSummary;

  // /************************************************
  //   TrackerMap of sqrt(d_ii) of 1 IOV
  // *************************************************/
  template <AlignmentPI::index i>
  class TrackerAlignmentErrorExtendedTrackerMap : public PlotImage<AlignmentErrorsExtended, SINGLE_IOV> {
  public:
    TrackerAlignmentErrorExtendedTrackerMap()
        : PlotImage<AlignmentErrorsExtended, SINGLE_IOV>("Tracker Map of sqrt(d_{" + getStringFromIndex(i) +
                                                         "}) of APE matrix") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<AlignmentErrorsExtended> payload = fetchPayload(std::get<1>(iov));

      std::string titleMap = "APE #sqrt{d_{" + getStringFromIndex(i) + "}} value (payload : " + std::get<1>(iov) + ")";

      std::unique_ptr<TrackerMap> tmap = std::make_unique<TrackerMap>("APE_dii");
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;

      auto indices = AlignmentPI::getIndices(i);

      bool isPhase0(false);
      if (alignErrors.size() == AlignmentPI::phase0size)
        isPhase0 = true;

      for (const auto& it : alignErrors) {
        CLHEP::HepSymMatrix errMatrix = it.matrix();

        // fill the tracker map

        int subid = DetId(it.rawId()).subdetId();

        if (DetId(it.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.rawId() << " - terminating ";
          return false;
        }

        if (isPhase0) {
          tmap->addPixel(true);
          tmap->fill(it.rawId(), sqrt(errMatrix[indices.first][indices.second]) * AlignmentPI::cmToUm);
        } else {
          if (subid != 1 && subid != 2) {
            tmap->fill(it.rawId(), sqrt(errMatrix[indices.first][indices.second]) * AlignmentPI::cmToUm);
          }
        }
      }  // loop over detIds

      //=========================

      auto autoRange = tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);
      // protect against uniform values (APE are defined positive)
      if (autoRange.first != autoRange.second) {
        tmap->save(true, 0., autoRange.second, fileName);
      } else {
        if (autoRange.first != 0.)
          tmap->save(true, 0., autoRange.first * 1.05, fileName);
        else
          tmap->save(true, 0., 1., fileName);
      }

      return true;
    }
  };

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXTrackerMap;
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYTrackerMap;
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZTrackerMap;

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYTrackerMap;
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZTrackerMap;
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZTrackerMap;

  // /************************************************
  //  Partition details of 1 IOV
  // *************************************************/
  template <AlignmentPI::partitions q>
  class TrackerAlignmentErrorExtendedDetail : public PlotImage<AlignmentErrorsExtended, SINGLE_IOV> {
  public:
    TrackerAlignmentErrorExtendedDetail()
        : PlotImage<AlignmentErrorsExtended, SINGLE_IOV>("Details for " + AlignmentPI::getStringFromPart(q)) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      gStyle->SetPaintTextFormat(".1f");
      std::shared_ptr<AlignmentErrorsExtended> payload = fetchPayload(std::get<1>(iov));

      std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;

      const char* path_toTopologyXML = (alignErrors.size() == AlignmentPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      bool isPhase0(false);
      if (alignErrors.size() == AlignmentPI::phase0size)
        isPhase0 = true;

      TCanvas canvas("Summary", "Summary", 1600, 1200);
      canvas.Divide(3, 2);

      // define the paritions range to act upon
      auto begin = partLimits.at(q).first;
      auto end = partLimits.at(q).second;
      // dinamically defined range
      auto range = (end - begin) + 1;

      std::map<std::pair<AlignmentPI::index, AlignmentPI::regions>, std::shared_ptr<TH1F> > APE_spectraByRegion;
      std::map<AlignmentPI::index, std::shared_ptr<TH1F> > summaries;

      for (int k = AlignmentPI::XX; k <= AlignmentPI::ZZ; k++) {
        AlignmentPI::index coord = (AlignmentPI::index)k;
        std::string s_coord = AlignmentPI::getStringFromIndex(coord);

        summaries[coord] = std::make_shared<TH1F>(
            Form("Summary_%s", s_coord.c_str()),
            Form("Summary for #LT #sqrt{d_{%s}} #GT APE;;average APE #LT #sqrt{d_{%s}} #GT [#mum]",
                 s_coord.c_str(),
                 s_coord.c_str()),
            range,
            0,
            range);

        //COUT<<"begin ( "<< begin << "): " << AlignmentPI::getStringFromRegionEnum(begin) << " end ( " << end << "): " <<  AlignmentPI::getStringFromRegionEnum(end) <<" | range = "<< range << std::endl;

        for (int j = begin; j <= end; j++) {
          AlignmentPI::regions part = (AlignmentPI::regions)j;

          // dont' book region that don't exist
          if (isPhase0 && (part == AlignmentPI::BPixL4o || part == AlignmentPI::BPixL4i ||
                           part == AlignmentPI::FPixmL3 || part == AlignmentPI::FPixpL3))
            continue;

          std::string s_part = AlignmentPI::getStringFromRegionEnum(part);

          auto hash = std::make_pair(coord, part);

          APE_spectraByRegion[hash] = std::make_shared<TH1F>(
              Form("hAPE_%s_%s", s_coord.c_str(), s_part.c_str()),
              Form(";%s APE #sqrt{d_{%s}} [#mum];n. of modules", s_part.c_str(), s_coord.c_str()),
              1000,
              0.,
              1000.);
        }
      }

      // loop on the vector of errors
      for (const auto& it : alignErrors) {
        CLHEP::HepSymMatrix errMatrix = it.matrix();

        if (DetId(it.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.rawId() << " - terminating ";
          return false;
        }

        int subid = DetId(it.rawId()).subdetId();
        if (subid != q)
          continue;

        // fill the struct
        AlignmentPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.rawId());
        t_info_fromXML.fillGeometryInfo(detid, tTopo, isPhase0);
        //t_info_fromXML.printAll();

        AlignmentPI::regions thePart = t_info_fromXML.filterThePartition();

        // skip the glued detector detIds
        if (thePart == AlignmentPI::StripDoubleSide)
          continue;

        for (int k = AlignmentPI::XX; k <= AlignmentPI::ZZ; k++) {
          AlignmentPI::index coord = (AlignmentPI::index)k;
          auto indices = AlignmentPI::getIndices(coord);
          auto hash = std::make_pair(coord, thePart);

          APE_spectraByRegion[hash]->Fill(sqrt(errMatrix[indices.first][indices.second]) * AlignmentPI::cmToUm);

        }  // loop on the coordinate indices
      }    // loop over detIds

      // plotting section

      TLegend legend = TLegend(0.15, 0.85, 0.99, 0.92);
      legend.AddEntry(summaries[AlignmentPI::XX].get(),
                      ("#splitline{IOV: " + std::to_string(std::get<0>(iov)) + "}{" + std::get<1>(iov) + "}").c_str(),
                      "F");
      legend.SetTextSize(0.030);

      for (int k = AlignmentPI::XX; k <= AlignmentPI::ZZ; k++) {
        AlignmentPI::index coord = (AlignmentPI::index)k;

        for (int j = begin; j <= end; j++) {
          AlignmentPI::regions part = (AlignmentPI::regions)j;

          // don't fill regions that do not exist
          if (isPhase0 && (part == AlignmentPI::BPixL4o || part == AlignmentPI::BPixL4i ||
                           part == AlignmentPI::FPixmL3 || part == AlignmentPI::FPixpL3))
            continue;

          auto hash = std::make_pair(coord, part);
          summaries[coord]->GetXaxis()->SetBinLabel((j - begin) + 1,
                                                    AlignmentPI::getStringFromRegionEnum(part).c_str());

          // avoid filling the histogram with numerical noise
          float mean = APE_spectraByRegion[hash]->GetMean() > 10.e-6 ? APE_spectraByRegion[hash]->GetMean() : 10.e-6;
          summaries[coord]->SetBinContent((j - begin) + 1, mean);
          //summaries[coord]->SetBinError((j-begin)+1,APE_spectraByRegion[hash]->GetRMS());
          AlignmentPI::makeNicePlotStyle(summaries[coord].get(), kBlue);
          summaries[coord]->GetXaxis()->LabelsOption("v");
          summaries[coord]->GetXaxis()->SetLabelSize(0.06);

        }  // loop over the detector regions

        canvas.cd(k);
        canvas.cd(k)->SetTopMargin(0.08);
        canvas.cd(k)->SetBottomMargin(0.15);
        canvas.cd(k)->SetLeftMargin(0.15);
        canvas.cd(k)->SetRightMargin(0.01);
        //summaries[coord]->SetTitleOffset(0.06);
        summaries[coord]->SetFillColorAlpha(kRed, 0.35);
        summaries[coord]->SetMarkerSize(2.5);
        summaries[coord]->SetMarkerColor(kRed);

        // to ensure 0. is actually displayed as 0.
        summaries[coord]->GetYaxis()->SetRangeUser(0., std::max(1., summaries[coord]->GetMaximum() * 1.50));
        summaries[coord]->Draw("text90");
        summaries[coord]->Draw("HISTsame");

        legend.Draw("same");

      }  // loop over the matrix elements

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  typedef TrackerAlignmentErrorExtendedDetail<AlignmentPI::BPix> TrackerAlignmentErrorExtendedBPixDetail;
  typedef TrackerAlignmentErrorExtendedDetail<AlignmentPI::FPix> TrackerAlignmentErrorExtendedFPixDetail;
  typedef TrackerAlignmentErrorExtendedDetail<AlignmentPI::TIB> TrackerAlignmentErrorExtendedTIBDetail;
  typedef TrackerAlignmentErrorExtendedDetail<AlignmentPI::TOB> TrackerAlignmentErrorExtendedTOBDetail;
  typedef TrackerAlignmentErrorExtendedDetail<AlignmentPI::TID> TrackerAlignmentErrorExtendedTIDDetail;
  typedef TrackerAlignmentErrorExtendedDetail<AlignmentPI::TEC> TrackerAlignmentErrorExtendedTECDetail;

  // /************************************************
  //  Tracker Aligment Extended Errors grand summary comparison of 2 IOVs
  // *************************************************/

  template <AlignmentPI::index i, int ntags, IOVMultiplicity nIOVs>
  class TrackerAlignmentErrorExtendedComparatorBase : public PlotImage<AlignmentErrorsExtended, nIOVs, ntags> {
  public:
    TrackerAlignmentErrorExtendedComparatorBase()
        : PlotImage<AlignmentErrorsExtended, nIOVs, ntags>("Summary per Tracker region of sqrt(d_{" +
                                                           getStringFromIndex(i) + "}) of APE matrix") {}

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

      gStyle->SetPaintTextFormat(".1f");

      std::shared_ptr<AlignmentErrorsExtended> last_payload = this->fetchPayload(std::get<1>(lastiov));
      std::shared_ptr<AlignmentErrorsExtended> first_payload = this->fetchPayload(std::get<1>(firstiov));

      std::vector<AlignTransformErrorExtended> f_alignErrors = first_payload->m_alignError;
      std::vector<AlignTransformErrorExtended> l_alignErrors = last_payload->m_alignError;

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Comparison", "Comparison", 1600, 800);

      std::map<AlignmentPI::regions, std::shared_ptr<TH1F> > FirstAPE_spectraByRegion;
      std::map<AlignmentPI::regions, std::shared_ptr<TH1F> > LastAPE_spectraByRegion;
      std::shared_ptr<TH1F> summaryFirst;
      std::shared_ptr<TH1F> summaryLast;

      // get the name of the index
      std::string s_coord = AlignmentPI::getStringFromIndex(i);

      // book the intermediate histograms
      for (int r = AlignmentPI::BPixL1o; r != AlignmentPI::StripDoubleSide; r++) {
        AlignmentPI::regions part = static_cast<AlignmentPI::regions>(r);
        std::string s_part = AlignmentPI::getStringFromRegionEnum(part);

        FirstAPE_spectraByRegion[part] =
            std::make_shared<TH1F>(Form("hfirstAPE_%s_%s", s_coord.c_str(), s_part.c_str()),
                                   Form(";%s APE #sqrt{d_{%s}} [#mum];n. of modules", s_part.c_str(), s_coord.c_str()),
                                   1000,
                                   0.,
                                   1000.);
        LastAPE_spectraByRegion[part] =
            std::make_shared<TH1F>(Form("hlastAPE_%s_%s", s_coord.c_str(), s_part.c_str()),
                                   Form(";%s APE #sqrt{d_{%s}} [#mum];n. of modules", s_part.c_str(), s_coord.c_str()),
                                   1000,
                                   0.,
                                   1000.);
      }

      summaryFirst =
          std::make_shared<TH1F>(Form("first Summary_%s", s_coord.c_str()),
                                 Form("Summary for #LT #sqrt{d_{%s}} #GT APE;;average APE #LT #sqrt{d_{%s}} #GT [#mum]",
                                      s_coord.c_str(),
                                      s_coord.c_str()),
                                 FirstAPE_spectraByRegion.size(),
                                 0,
                                 FirstAPE_spectraByRegion.size());
      summaryLast =
          std::make_shared<TH1F>(Form("last Summary_%s", s_coord.c_str()),
                                 Form("Summary for #LT #sqrt{d_{%s}} #GT APE;;average APE #LT #sqrt{d_{%s}} #GT [#mum]",
                                      s_coord.c_str(),
                                      s_coord.c_str()),
                                 LastAPE_spectraByRegion.size(),
                                 0,
                                 LastAPE_spectraByRegion.size());

      const char* path_toTopologyXML = (f_alignErrors.size() == AlignmentPI::phase0size)
                                           ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                                           : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology f_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      bool isPhase0(false);
      if (f_alignErrors.size() == AlignmentPI::phase0size)
        isPhase0 = true;

      // -------------------------------------------------------------------
      // loop on the first vector of errors
      // -------------------------------------------------------------------
      for (const auto& it : f_alignErrors) {
        CLHEP::HepSymMatrix errMatrix = it.matrix();

        if (DetId(it.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.rawId() << " - terminating ";
          return false;
        }

        AlignmentPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.rawId());
        t_info_fromXML.fillGeometryInfo(detid, f_tTopo, isPhase0);

        AlignmentPI::regions thePart = t_info_fromXML.filterThePartition();

        // skip the glued detector detIds
        if (thePart == AlignmentPI::StripDoubleSide)
          continue;

        auto indices = AlignmentPI::getIndices(i);
        FirstAPE_spectraByRegion[thePart]->Fill(sqrt(errMatrix[indices.first][indices.second]) * AlignmentPI::cmToUm);
      }  // ends loop on the vector of error transforms

      path_toTopologyXML = (l_alignErrors.size() == AlignmentPI::phase0size)
                               ? "Geometry/TrackerCommonData/data/trackerParameters.xml"
                               : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology l_tTopo =
          StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

      if (l_alignErrors.size() == AlignmentPI::phase0size)
        isPhase0 = true;

      // -------------------------------------------------------------------
      // loop on the second vector of errors
      // -------------------------------------------------------------------
      for (const auto& it : l_alignErrors) {
        CLHEP::HepSymMatrix errMatrix = it.matrix();

        if (DetId(it.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector")
              << "Encountered invalid Tracker DetId:" << it.rawId() << " - terminating ";
          return false;
        }

        AlignmentPI::topolInfo t_info_fromXML;
        t_info_fromXML.init();
        DetId detid(it.rawId());
        t_info_fromXML.fillGeometryInfo(detid, l_tTopo, isPhase0);

        AlignmentPI::regions thePart = t_info_fromXML.filterThePartition();

        // skip the glued detector detIds
        if (thePart == AlignmentPI::StripDoubleSide)
          continue;

        auto indices = AlignmentPI::getIndices(i);
        LastAPE_spectraByRegion[thePart]->Fill(sqrt(errMatrix[indices.first][indices.second]) * AlignmentPI::cmToUm);
      }  // ends loop on the vector of error transforms

      // fill the summary plots
      int bin = 1;
      for (int r = AlignmentPI::BPixL1o; r != AlignmentPI::StripDoubleSide; r++) {
        AlignmentPI::regions part = static_cast<AlignmentPI::regions>(r);

        summaryFirst->GetXaxis()->SetBinLabel(bin, AlignmentPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float f_mean =
            FirstAPE_spectraByRegion[part]->GetMean() > 10.e-6 ? FirstAPE_spectraByRegion[part]->GetMean() : 10.e-6;
        summaryFirst->SetBinContent(bin, f_mean);
        //summaryFirst->SetBinError(bin,APE_spectraByRegion[hash]->GetRMS());

        summaryLast->GetXaxis()->SetBinLabel(bin, AlignmentPI::getStringFromRegionEnum(part).c_str());
        // avoid filling the histogram with numerical noise
        float l_mean =
            LastAPE_spectraByRegion[part]->GetMean() > 10.e-6 ? LastAPE_spectraByRegion[part]->GetMean() : 10.e-6;
        summaryLast->SetBinContent(bin, l_mean);
        //summaryLast->SetBinError(bin,APE_spectraByRegion[hash]->GetRMS());
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

      summaryFirst->GetYaxis()->SetRangeUser(0., std::max(0., max * 1.20));

      summaryFirst->Draw("bar2");
      //summaryFirst->Draw("text90same");
      summaryLast->Draw("bar2,same");
      //summaryLast->Draw("text180same");

      TLegend legend = TLegend(0.52, 0.82, 0.98, 0.9);
      legend.SetHeader((getStringFromIndex(i) + " APE value comparison").c_str(),
                       "C");  // option "C" allows to center the header
      legend.AddEntry(
          summaryLast.get(),
          ("IOV: #scale[1.2]{" + std::to_string(std::get<0>(lastiov)) + "} | #color[2]{" + std::get<1>(lastiov) + "}")
              .c_str(),
          "F");
      legend.AddEntry(
          summaryFirst.get(),
          ("IOV: #scale[1.2]{" + std::to_string(std::get<0>(firstiov)) + "} | #color[4]{" + std::get<1>(firstiov) + "}")
              .c_str(),
          "F");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;

    }  // ends fill method
  };

  template <AlignmentPI::index i>
  using TrackerAlignmentErrorExtendedComparator = TrackerAlignmentErrorExtendedComparatorBase<i, 1, MULTI_IOV>;

  template <AlignmentPI::index i>
  using TrackerAlignmentErrorExtendedComparatorTwoTags = TrackerAlignmentErrorExtendedComparatorBase<i, 2, SINGLE_IOV>;

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedComparator<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXComparator;
  typedef TrackerAlignmentErrorExtendedComparator<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYComparator;
  typedef TrackerAlignmentErrorExtendedComparator<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZComparator;

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedComparator<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYComparator;
  typedef TrackerAlignmentErrorExtendedComparator<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZComparator;
  typedef TrackerAlignmentErrorExtendedComparator<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZComparator;

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedComparatorTwoTags<AlignmentPI::XX>
      TrackerAlignmentErrorExtendedXXComparatorTwoTags;
  typedef TrackerAlignmentErrorExtendedComparatorTwoTags<AlignmentPI::YY>
      TrackerAlignmentErrorExtendedYYComparatorTwoTags;
  typedef TrackerAlignmentErrorExtendedComparatorTwoTags<AlignmentPI::ZZ>
      TrackerAlignmentErrorExtendedZZComparatorTwoTags;

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedComparatorTwoTags<AlignmentPI::XY>
      TrackerAlignmentErrorExtendedXYComparatorTwoTags;
  typedef TrackerAlignmentErrorExtendedComparatorTwoTags<AlignmentPI::XZ>
      TrackerAlignmentErrorExtendedXZComparatorTwoTags;
  typedef TrackerAlignmentErrorExtendedComparatorTwoTags<AlignmentPI::YZ>
      TrackerAlignmentErrorExtendedYZComparatorTwoTags;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(TrackerAlignmentErrorExtended) {
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedBPixDetail);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedFPixDetail);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedTIBDetail);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedTOBDetail);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedTIDDetail);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedTECDetail);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXComparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYComparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZComparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYComparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZComparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZComparator);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZComparatorTwoTags);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZComparatorTwoTags);
}
