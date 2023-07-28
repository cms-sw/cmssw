#ifndef CONDCORE_BEAMSPOTPLUGINS_BEAMSPOTPAYLOADINSPECTORHELPER_H
#define CONDCORE_BEAMSPOTPLUGINS_BEAMSPOTPAYLOADINSPECTORHELPER_H

// User includes
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// system includes
#include <ctime>
#include <fmt/printf.h>
#include <memory>
#include <sstream>
#include <regex>

// ROOT includes
#include "TCanvas.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TStyle.h"

//#define MMDEBUG  /* to make it verbose */

namespace beamSpotPI {

  inline std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
    auto kLowMask = 0XFFFFFFFF;
    auto run = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run, lumi);
  }

  enum parameters {
    X = 0,                // 0  regular BS methods
    Y = 1,                // 1
    Z = 2,                // 2
    sigmaX = 3,           // 3
    sigmaY = 4,           // 4
    sigmaZ = 5,           // 5
    dxdz = 6,             // 6
    dydz = 7,             // 7
    lastLumi = 8,         // 8  additional int's
    lastRun = 9,          // 9
    lastFill = 10,        // 10
    nTracks = 11,         // 11
    nPVs = 12,            // 12
    nUsedEvents = 13,     // 13
    maxPVs = 14,          // 14
    meanPV = 15,          // 15 additional float's
    meanErrorPV = 16,     // 16
    rmsPV = 17,           // 17
    rmsErrorPV = 18,      // 18
    creationTime = 19,    // 19 additional cond::Time_t
    startTimeStamp = 20,  // 20
    endTimeStamp = 21,    // 21
    startTime = 22,       // 22 additional std::string
    endTime = 23,         // 23
    lumiRange = 24,       // 24
    END_OF_TYPES = 25,
  };

  /************************************************/
  // Function to convert cond::Time_t (in microseconds) to human-readable date string
  std::string convertTimeToDateString(cond::Time_t timeValue, bool hasMicros = false, bool toUTC = true) {
    // Convert microseconds to seconds
    std::time_t unixTime = static_cast<std::time_t>(hasMicros ? timeValue / 1000000 : timeValue);

    // Convert std::time_t to struct tm (to UTC, or not)
    std::tm* timeInfo = toUTC ? std::gmtime(&unixTime) : std::localtime(&unixTime);

    // Convert struct tm to human-readable string format
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeInfo);

    // Append microseconds to the string
    std::string dateString(buffer);
    //dateString += "." + std::to_string(timeValue % 1000000);

    return dateString;
  }

  /************************************************/
  inline std::string getStringFromParamEnum(const parameters& parameter,
                                            const bool addUnits = false /*not used by default*/) {
    switch (parameter) {
      case X:
        return (addUnits ? "X [cm]" : "X");
      case Y:
        return (addUnits ? "Y [cm]" : "Y");
      case Z:
        return (addUnits ? "Z [cm]" : "Z");
      case sigmaX:
        return (addUnits ? "#sigma_{X} [cm]" : "sigmaX");
      case sigmaY:
        return (addUnits ? "#sigma_{Y} [cm]" : "sigmaY");
      case sigmaZ:
        return (addUnits ? "#sigma_{Z} [cm]" : "sigmaZ");
      case dxdz:
        return (addUnits ? "#frac{dX}{dZ} [rad]" : "dx/dz");
      case dydz:
        return (addUnits ? "#frac{dY}{dZ} [rad]" : "dy/dz");
      default:
        return "should never be here";
    }
  }

  /**
   * Helper class for operations on the Beam Spot Parameters
   * It's a simplified representation of the beamspot
   * data used as the underlying type for data transfers and comparisons
   */
  template <class PayloadType>
  class BSParamsHelper {
    typedef std::array<double, parameters::lastLumi> bshelpdata;

  public:
    BSParamsHelper(const std::shared_ptr<PayloadType>& bs) {
      // fill in the central values
      m_values[parameters::X] = bs->x(), m_values[parameters::Y] = bs->y(), m_values[parameters::Z] = bs->z();
      m_values[parameters::sigmaX] = bs->beamWidthX(), m_values[parameters::sigmaY] = bs->beamWidthY(),
      m_values[parameters::sigmaZ] = bs->sigmaZ();
      m_values[parameters::dxdz] = bs->dxdz(), m_values[parameters::dydz] = bs->dydz();

      // fill in the errors
      m_errors[parameters::X] = bs->xError(), m_errors[parameters::Y] = bs->yError(),
      m_errors[parameters::Z] = bs->zError();
      m_errors[parameters::sigmaX] = bs->beamWidthXError(), m_errors[parameters::sigmaY] = bs->beamWidthYError(),
      m_errors[parameters::sigmaZ] = bs->sigmaZError();
      m_errors[parameters::dxdz] = bs->dxdzError(), m_errors[parameters::dydz] = bs->dydzError();
    }

    void printDebug(std::stringstream& ss) {
      ss << "Dumping BeamSpot parameters Data:" << std::endl;
      for (uint i = parameters::X; i <= parameters::dydz; i++) {
        parameters par = static_cast<parameters>(i);
        ss << getStringFromParamEnum(par) << " : " << m_values[i] << std::endl;
        ss << getStringFromParamEnum(par) << " error: " << m_errors[i] << std::endl;
        ss << std::endl;
      }
    }

    inline const bshelpdata centralValues() const { return m_values; }
    inline const bshelpdata errors() const { return m_errors; }

    // get the difference in values
    const bshelpdata diffCentralValues(const BSParamsHelper& bs2, const bool isPull = false) const {
      bshelpdata ret;
      for (uint i = parameters::X; i <= parameters::dydz; i++) {
        ret[i] = this->centralValues()[i] - bs2.centralValues()[i];
        if (isPull)
          (this->centralValues()[i] != 0.) ? ret[i] /= this->centralValues()[i] : 0.;
      }
      return ret;
    }

    // get the difference in errors
    const bshelpdata diffErrors(const BSParamsHelper& bs2, const bool isPull = false) const {
      bshelpdata ret;
      for (uint i = parameters::X; i <= parameters::dydz; i++) {
        ret[i] = this->errors()[i] - bs2.errors()[i];
        if (isPull)
          (this->errors()[i] != 0.) ? ret[i] /= this->errors()[i] : 0.;
      }
      return ret;
    }

  private:
    bshelpdata m_values; /* central values */
    bshelpdata m_errors; /* errors */
  };

  /************************************************
    template classes (history)
  *************************************************/

  template <parameters my_param, class PayloadType>
  class BeamSpot_history : public cond::payloadInspector::HistoryPlot<PayloadType, std::pair<double, double> > {
  public:
    BeamSpot_history()
        : cond::payloadInspector::HistoryPlot<PayloadType, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs run number", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(PayloadType& payload) override {
      auto ret = std::make_pair<double, double>(-9999., -9999.);

      switch (my_param) {
        case X:
          return std::make_pair<double, double>(payload.x(), payload.xError());
        case Y:
          return std::make_pair<double, double>(payload.y(), payload.yError());
        case Z:
          return std::make_pair<double, double>(payload.z(), payload.zError());
        case sigmaX:
          return std::make_pair<double, double>(payload.beamWidthX(), payload.beamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.beamWidthY(), payload.beamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.sigmaZ(), payload.sigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.dxdz(), payload.dxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.dydz(), payload.dydzError());
        case END_OF_TYPES:
          return ret;
        default:
          return ret;
      }
    }
  };

  /************************************************
    template classes (run history)
   *************************************************/

  template <parameters my_param, class PayloadType>
  class BeamSpot_runhistory : public cond::payloadInspector::RunHistoryPlot<PayloadType, std::pair<double, double> > {
  public:
    BeamSpot_runhistory()
        : cond::payloadInspector::RunHistoryPlot<PayloadType, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs run number", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(PayloadType& payload) override {
      auto ret = std::make_pair<double, double>(-9999., -9999.);

      switch (my_param) {
        case X:
          return std::make_pair<double, double>(payload.x(), payload.xError());
        case Y:
          return std::make_pair<double, double>(payload.y(), payload.yError());
        case Z:
          return std::make_pair<double, double>(payload.z(), payload.zError());
        case sigmaX:
          return std::make_pair<double, double>(payload.beamWidthX(), payload.beamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.beamWidthY(), payload.beamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.sigmaZ(), payload.sigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.dxdz(), payload.dxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.dydz(), payload.dydzError());
        case END_OF_TYPES:
          return ret;
        default:
          return ret;
      }
    }
  };

  /************************************************
    template classes (time history)
  *************************************************/

  template <parameters my_param, class PayloadType>
  class BeamSpot_timehistory : public cond::payloadInspector::TimeHistoryPlot<PayloadType, std::pair<double, double> > {
  public:
    BeamSpot_timehistory()
        : cond::payloadInspector::TimeHistoryPlot<PayloadType, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs time", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(PayloadType& payload) override {
      auto ret = std::make_pair<double, double>(-9999., -9999.);

      switch (my_param) {
        case X:
          return std::make_pair<double, double>(payload.x(), payload.xError());
        case Y:
          return std::make_pair<double, double>(payload.y(), payload.yError());
        case Z:
          return std::make_pair<double, double>(payload.z(), payload.zError());
        case sigmaX:
          return std::make_pair<double, double>(payload.beamWidthX(), payload.beamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.beamWidthY(), payload.beamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.sigmaZ(), payload.sigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.dxdz(), payload.dxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.dydz(), payload.dydzError());
        case END_OF_TYPES:
          return ret;
        default:
          return ret;
      }
    }
  };

  /************************************************
    X-Y correlation plot
  *************************************************/
  template <class PayloadType>
  class xyCorrelation : public cond::payloadInspector::ScatterPlot<PayloadType, double, double> {
  public:
    xyCorrelation() : cond::payloadInspector::ScatterPlot<PayloadType, double, double>("BeamSpot x vs y", "x", "y") {}

    std::tuple<double, double> getFromPayload(PayloadType& payload) override {
      return std::make_tuple(payload.x(), payload.y());
    }
  };

  /************************************************
    Display of Beam Spot parameters
  *************************************************/
  template <class PayloadType>
  class DisplayParameters : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    DisplayParameters()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "Display of BeamSpot parameters") {
      if constexpr (std::is_same_v<PayloadType, BeamSpotOnlineObjects>) {
        isOnline_ = true;
      } else {
        isOnline_ = false;
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      gStyle->SetHistMinimumZero(kTRUE);

      m_payload = this->fetchPayload(std::get<1>(iov));

      TCanvas canvas("Beam Spot Parameters Summary", "BeamSpot Parameters summary", isOnline_ ? 1500 : 1000, 1000);
      if (isOnline_) {
        canvas.Divide(2, 1);
      }
      canvas.cd(1);

      canvas.cd(1)->SetTopMargin(0.05);
      canvas.cd(1)->SetBottomMargin(0.06);
      canvas.cd(1)->SetLeftMargin(0.15);
      canvas.cd(1)->SetRightMargin(0.01);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      auto h2_BSParameters = std::make_unique<TH2F>("Parameters", "", 2, 0.0, 2.0, 8, 0, 8.);
      h2_BSParameters->SetStats(false);

      std::function<double(parameters, bool)> cutFunctor = [this](parameters my_param, bool isError) {
        double ret(-999.);
        if (!isError) {
          switch (my_param) {
            case X:
              return m_payload->x();
            case Y:
              return m_payload->y();
            case Z:
              return m_payload->z();
            case sigmaX:
              return m_payload->beamWidthX();
            case sigmaY:
              return m_payload->beamWidthY();
            case sigmaZ:
              return m_payload->sigmaZ();
            case dxdz:
              return m_payload->dxdz();
            case dydz:
              return m_payload->dydz();
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        } else {
          switch (my_param) {
            case X:
              return m_payload->xError();
            case Y:
              return m_payload->yError();
            case Z:
              return m_payload->zError();
            case sigmaX:
              return m_payload->beamWidthXError();
            case sigmaY:
              return m_payload->beamWidthYError();
            case sigmaZ:
              return m_payload->sigmaZError();
            case dxdz:
              return m_payload->dxdzError();
            case dydz:
              return m_payload->dydzError();
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        }
      };

      h2_BSParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_BSParameters->GetXaxis()->SetBinLabel(2, "Error");

      unsigned int yBin = 8;
      for (int foo = parameters::X; foo <= parameters::dydz; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = getStringFromTypeEnum(param);
        h2_BSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_BSParameters->SetBinContent(1, yBin, cutFunctor(param, false));
        h2_BSParameters->SetBinContent(2, yBin, cutFunctor(param, true));
        yBin--;
      }

      h2_BSParameters->GetXaxis()->LabelsOption("h");
      h2_BSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_BSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_BSParameters->SetMarkerSize(1.5);
      h2_BSParameters->Draw("TEXT");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      if (isOnline_) {
        ltx.SetTextSize(0.030);
      } else {
        ltx.SetTextSize(0.025);
      }
      ltx.SetTextAlign(11);

      auto runLS = beamSpotPI::unpack(std::get<0>(iov));

      ltx.DrawLatexNDC(
          gPad->GetLeftMargin(),
          1 - gPad->GetTopMargin() + 0.01,
          (tagname + " IOV: #color[4]{" + std::to_string(runLS.first) + "," + std::to_string(runLS.second) + "}")
              .c_str());

      if (isOnline_) {
        canvas.cd(2);
        canvas.cd(2)->SetTopMargin(0.05);
        canvas.cd(2)->SetBottomMargin(0.06);
        canvas.cd(2)->SetLeftMargin(0.25);
        canvas.cd(2)->SetRightMargin(0.01);
        canvas.cd(2)->Modified();
        canvas.cd(2)->SetGrid();

        auto extras = fillTheExtraHistogram();
        if (extras) {
          for (int bin = 1; bin <= extras->GetNbinsY(); bin++) {
            edm::LogVerbatim("BeamSpotPayloadInspectorHelper")
                << extras->GetYaxis()->GetBinLabel(bin) << ": " << extras->GetBinContent(1, bin) << "\n";
          }
        }
        extras->Draw("TEXT");

        ltx.DrawLatexNDC(
            gPad->GetLeftMargin(),
            1 - gPad->GetTopMargin() + 0.01,
            (tagname + " IOV: #color[4]{" + std::to_string(runLS.first) + "," + std::to_string(runLS.second) + "}")
                .c_str());

        if constexpr (std::is_same_v<PayloadType, BeamSpotOnlineObjects>) {
          // protections needed against old payload that do not have these data members persisted
          const auto& creationTime = test_<cond::Time_t, std::out_of_range>(
              [&]() {
                return m_payload->creationTime();
              },  // Lambda function capturing m_payload and calling creationTime
              better_error);

          const auto& startTime = test_<cond::Time_t, std::out_of_range>(
              [&]() {
                return m_payload->startTimeStamp();
              },  // Lambda function capturing m_payload and calling startTimeStamp
              better_error);

          const auto& endTime = test_<cond::Time_t, std::out_of_range>(
              [&]() {
                return m_payload->endTimeStamp();
              },  // Lambda function capturing m_payload and calling endTimeStamp
              better_error);
          canvas.cd(2);
          ltx.SetTextSize(0.025);
          ltx.DrawLatexNDC(
              gPad->GetLeftMargin() + 0.01,
              gPad->GetBottomMargin() + 0.15,
              ("#color[2]{(" + beamSpotPI::convertTimeToDateString(creationTime, /*has us*/ true) + ")}").c_str());

          ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                           gPad->GetBottomMargin() + 0.085,
                           ("#color[2]{(" + beamSpotPI::convertTimeToDateString(startTime) + ")}").c_str());

          ltx.DrawLatexNDC(gPad->GetLeftMargin() + 0.01,
                           gPad->GetBottomMargin() + 0.025,
                           ("#color[2]{(" + beamSpotPI::convertTimeToDateString(endTime) + ")}").c_str());

          ltx.DrawLatexNDC(
              gPad->GetLeftMargin(), gPad->GetBottomMargin() - 0.05, "#color[4]{N.B.} TimeStamps are in UTC");
        }

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

        return true;
      } else {
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

        return true;
      }
    }

  public:
    virtual std::shared_ptr<TH2F> fillTheExtraHistogram() const { return nullptr; }

  protected:
    bool isOnline_;
    std::shared_ptr<PayloadType> m_payload;

    /**
     * Can't use beamSpotPI::getStringFromParamEnum becasue it needs to be overridden
     * for the BeamSpotOnlineObjects case.
     */
    virtual std::string getStringFromTypeEnum(const parameters& parameter) const {
      switch (parameter) {
        case X:
          return "X [cm]";
        case Y:
          return "Y [cm]";
        case Z:
          return "Z [cm]";
        case sigmaX:
          return "#sigma_{X} [cm]";
        case sigmaY:
          return "#sigma_{Y} [cm]";
        case sigmaZ:
          return "#sigma_{Z} [cm]";
        case dxdz:
          return "#frac{dX}{dZ} [rad]";
        case dydz:
          return "#frac{dY}{dZ} [rad]";
        default:
          return "should never be here";
      }
    }

    // Slightly better error handler
    static void better_error(const std::exception& e) { edm::LogError("DisplayParameters") << e.what() << '\n'; }

    // Method to catch exceptions
    template <typename T, class Except, class Func, class Response>
    T test_(Func f, Response r) const {
      try {
        LogDebug("DisplayParameters") << "I have tried" << std::endl;
        return f();
      } catch (const Except& e) {
        LogDebug("DisplayParameters") << "I have caught!" << std::endl;
        r(e);
        return static_cast<T>(1);
      }
    }
  };

  /************************************************
    Display of Beam Spot parameters difference
  *************************************************/
  template <class PayloadType, cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class DisplayParametersDiff : public cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags> {
  public:
    DisplayParametersDiff()
        : cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags>("Display of BeamSpot parameters differences") {
      if constexpr (std::is_same_v<PayloadType, BeamSpotOnlineObjects>) {
        isOnline_ = true;
      } else {
        isOnline_ = false;
      }
    }

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      l_payload = this->fetchPayload(std::get<1>(lastiov));
      f_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Beam Spot Parameters Difference Summary", "Beam Spot Parameters Difference summary", 1000, 1000);
      canvas.cd(1);

      canvas.cd(1)->SetTopMargin(0.08);
      canvas.cd(1)->SetBottomMargin(0.06);
      canvas.cd(1)->SetLeftMargin(0.14);
      canvas.cd(1)->SetRightMargin(0.16);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      // for the "text"-filled histogram
      auto h2_BSParameters = std::make_unique<TH2F>("Parameters", "", 2, 0.0, 2.0, 8, 0, 8.);
      h2_BSParameters->SetStats(false);
      h2_BSParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_BSParameters->GetXaxis()->SetBinLabel(2, "Error");
      h2_BSParameters->GetXaxis()->LabelsOption("h");
      h2_BSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_BSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_BSParameters->SetMarkerSize(1.5);

      // prepare the arrays to fill the histogram
      beamSpotPI::BSParamsHelper fBS(f_payload);
      beamSpotPI::BSParamsHelper lBS(l_payload);

#ifdef MM_DEBUG
      std::stringstream ss1, ss2;
      edm::LogPrint("") << "**** first payload";
      fBS.printDebug(ss1);
      edm::LogPrint("") << ss1.str();
      edm::LogPrint("") << "**** last payload";
      lBS.printDebug(ss2);
      edm::LogPrint("") << ss2.str();
#endif

      const auto diffPars = fBS.diffCentralValues(lBS);
      const auto diffErrors = fBS.diffErrors(lBS);
      //const auto pullPars = fBS.diffCentralValues(lBS,true /*normalize*/);
      //const auto pullErrors = fBS.diffErrors(lBS,true /*normalize*/);

      unsigned int yBin = 8;
      for (int foo = parameters::X; foo <= parameters::dydz; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = beamSpotPI::getStringFromParamEnum(param, true /*use units*/);
        h2_BSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_BSParameters->SetBinContent(1, yBin, diffPars[foo]); /* profiting of the parameters enum indexing */
        h2_BSParameters->SetBinContent(2, yBin, diffErrors[foo]);
        yBin--;
      }

      // for the "colz"-filled histogram (clonde from the text-based one)
      auto h2_BSShadow = (TH2F*)(h2_BSParameters->Clone("shadow"));
      h2_BSShadow->GetZaxis()->SetTitle("#Delta Parameter(payload A - payload B)");
      h2_BSShadow->GetZaxis()->CenterTitle();
      h2_BSShadow->GetZaxis()->SetTitleOffset(1.5);

      // this is the fine gradient palette (blue to red)
      double max = h2_BSShadow->GetMaximum();
      double min = h2_BSShadow->GetMinimum();
      double val_white = 0.;
      double per_white = (max != min) ? ((val_white - min) / (max - min)) : 0.5;

      const int number = 3;
      double Red[number] = {0., 1., 1.};
      double Green[number] = {0., 1., 0.};
      double Blue[number] = {1., 1., 0.};
      double Stops[number] = {0., per_white, 1.};
      int nb = 256;
      h2_BSShadow->SetContour(nb);
      TColor::CreateGradientColorTable(number, Stops, Red, Green, Blue, nb);

      h2_BSShadow->Draw("colz");
      h2_BSParameters->Draw("TEXTsame");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.025);
      ltx.SetTextAlign(11);

      // compute the (run,LS) pairs
      auto l_runLS = beamSpotPI::unpack(std::get<0>(lastiov));
      std::string l_runLSs = "(" + std::to_string(l_runLS.first) + "," + std::to_string(l_runLS.second) + ")";
      auto f_runLS = beamSpotPI::unpack(std::get<0>(firstiov));
      std::string f_runLSs = "(" + std::to_string(f_runLS.first) + "," + std::to_string(f_runLS.second) + ")";

      if (this->m_plotAnnotations.ntags == 2) {
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin() - 0.1,
            1 - gPad->GetTopMargin() + 0.015,
            (fmt::sprintf(
                 "#splitline{A = #color[4]{%s}: %s}{B = #color[4]{%s}: %s}", f_tagname, f_runLSs, l_tagname, l_runLSs))
                .c_str());
      } else {
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin() - 0.1,
            1 - gPad->GetTopMargin() + 0.015,
            (fmt::sprintf("#splitline{#color[4]{%s}}{A = %s | B = %s}", f_tagname, l_runLSs, f_runLSs)).c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  public:
    /**
     * In case an extension to the BeamSpotOnlineObjects case will be needed in future
     */
    virtual std::shared_ptr<TH2F> fillTheExtraHistogram() const { return nullptr; }

  protected:
    bool isOnline_;
    std::shared_ptr<PayloadType> f_payload;
    std::shared_ptr<PayloadType> l_payload;
  };
}  // namespace beamSpotPI

// Similar namespace for SimBeamSpotObject
namespace simBeamSpotPI {

  enum parameters {
    X = 0,              // 0  - Positions
    Y = 1,              // 1
    Z = 2,              // 2
    sigmaZ = 3,         // 3  - Widths
    betaStar = 4,       // 4
    emittance = 5,      // 5
    expTransWidth = 6,  // 6  - from LPC-like calculation
    phi = 7,            // 7  - Additional parameters
    alpha = 8,          // 8
    timeOffset = 9,     // 9
    END_OF_TYPES = 10,
  };

  /************************************************/
  inline std::string getStringFromParamEnum(const parameters& parameter, const bool addUnits = false) {
    switch (parameter) {
      case X:
        return (addUnits ? "X [cm]" : "X");
      case Y:
        return (addUnits ? "Y [cm]" : "Y");
      case Z:
        return (addUnits ? "Z [cm]" : "Z");
      case sigmaZ:
        return (addUnits ? "#sigma_{Z} [cm]" : "sigmaZ");
      case betaStar:
        return (addUnits ? "#beta* [cm]" : "BetaStar");
      case emittance:
        return (addUnits ? "Emittance [cm]" : "Emittance");
      case expTransWidth:
        return (addUnits ? "#sigma^{trans}_{xy} [#mum]" : "Exp. trans width");
      case phi:
        return (addUnits ? "Phi [rad]" : "Phi");
      case alpha:
        return (addUnits ? "Alpha [rad]" : "Alpha");
      case timeOffset:
        return (addUnits ? "TimeOffset [ns]" : "TimeOffset");
      default:
        return "should never be here";
    }
  }

  /**
   * Helper class for operations on the Sim Beam Spot Parameters
   * It's a simplified representation of the beamspot
   * data used as the underlying type for data transfers and comparisons
   */
  template <class PayloadType>
  class SimBSParamsHelper {
    typedef std::array<double, parameters::END_OF_TYPES> bshelpdata;

  public:
    SimBSParamsHelper(const std::shared_ptr<PayloadType>& bs) {
      // fill in the values
      m_values[parameters::X] = bs->x(), m_values[parameters::Y] = bs->y(), m_values[parameters::Z] = bs->z();
      m_values[parameters::sigmaZ] = bs->sigmaZ(), m_values[parameters::betaStar] = bs->betaStar(),
      m_values[parameters::emittance] = bs->emittance();
      m_values[parameters::expTransWidth] = (1 / std::sqrt(2)) * std::sqrt(bs->emittance() * bs->betaStar()) * 10000.f;
      m_values[parameters::phi] = bs->phi(), m_values[parameters::alpha] = bs->alpha(),
      m_values[parameters::timeOffset] = bs->timeOffset();
    }

    void printDebug(std::stringstream& ss) {
      ss << "Dumping SimBeamSpot parameters Data:" << std::endl;
      for (uint i = parameters::X; i <= parameters::timeOffset; i++) {
        parameters par = static_cast<parameters>(i);
        ss << getStringFromParamEnum(par) << " : " << m_values[i] << std::endl;
        ss << std::endl;
      }
    }

    inline const bshelpdata centralValues() const { return m_values; }

    // get the difference in values
    const bshelpdata diffCentralValues(const SimBSParamsHelper& bs2, const bool isPull = false) const {
      bshelpdata ret;
      for (uint i = parameters::X; i <= parameters::timeOffset; i++) {
        ret[i] = this->centralValues()[i] - bs2.centralValues()[i];
        if (isPull)
          (this->centralValues()[i] != 0.) ? ret[i] /= this->centralValues()[i] : 0.;
      }
      return ret;
    }

  private:
    bshelpdata m_values;
  };

  /************************************************
    Display of Sim Beam Spot parameters
  *************************************************/
  template <class PayloadType>
  class DisplayParameters : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    DisplayParameters()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "Display of SimBeamSpot parameters") {}

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      gStyle->SetHistMinimumZero(kTRUE);

      m_payload = this->fetchPayload(std::get<1>(iov));

      TCanvas canvas("Sim Beam Spot Parameters Summary", "Sim BeamSpot Parameters summary", 1000, 1000);
      canvas.cd(1);
      canvas.cd(1)->SetTopMargin(0.05);
      canvas.cd(1)->SetBottomMargin(0.06);
      canvas.cd(1)->SetLeftMargin(0.25);
      canvas.cd(1)->SetRightMargin(0.01);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      auto h2_SimBSParameters = std::make_unique<TH2F>("Parameters", "", 1, 0.0, 1.0, END_OF_TYPES, 0, END_OF_TYPES);
      h2_SimBSParameters->SetStats(false);

      std::function<double(parameters)> cutFunctor = [this](parameters my_param) {
        double ret(-999.);
        switch (my_param) {
          case X:
            return m_payload->x();
          case Y:
            return m_payload->y();
          case Z:
            return m_payload->z();
          case sigmaZ:
            return m_payload->sigmaZ();
          case betaStar:
            return m_payload->betaStar();
          case emittance:
            return m_payload->emittance();
          case phi:
            return m_payload->phi();
          case alpha:
            return m_payload->alpha();
          case timeOffset:
            return m_payload->timeOffset();
          case expTransWidth:
            return (1 / std::sqrt(2)) * std::sqrt(m_payload->emittance() * m_payload->betaStar()) * cmToUm;
          case END_OF_TYPES:
            return ret;
          default:
            return ret;
        }
      };

      h2_SimBSParameters->GetXaxis()->SetBinLabel(1, "Value");

      unsigned int yBin = END_OF_TYPES;
      for (int foo = parameters::X; foo <= parameters::timeOffset; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = getStringFromParamEnum(param, true);
        h2_SimBSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_SimBSParameters->SetBinContent(1, yBin, cutFunctor(param));
        yBin--;
      }

      h2_SimBSParameters->GetXaxis()->LabelsOption("h");
      h2_SimBSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_SimBSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_SimBSParameters->SetMarkerSize(1.5);
      h2_SimBSParameters->Draw("TEXT");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.025);
      //ltx.SetTextAlign(11);

      auto runLS = beamSpotPI::unpack(std::get<0>(iov));

      ltx.SetTextAlign(32);  // Set text alignment to left (left-aligned)
      ltx.DrawLatexNDC(1 - gPad->GetRightMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       ("#color[2]{" + tagname + "} IOV: #color[4]{" + std::to_string(runLS.first) + "," +
                        std::to_string(runLS.second) + "}")
                           .c_str());

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  protected:
    std::shared_ptr<PayloadType> m_payload;

  private:
    static constexpr double cmToUm = 10000.f;
  };

  /************************************************
    Display of Sim Beam Spot parameters difference
  *************************************************/
  template <class PayloadType, cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class DisplayParametersDiff : public cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags> {
  public:
    DisplayParametersDiff()
        : cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags>(
              "Display of Sim BeamSpot parameters differences") {}

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      l_payload = this->fetchPayload(std::get<1>(lastiov));
      f_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas(
          "Sim Beam Spot Parameters Difference Summary", "Sim Beam Spot Parameters Difference summary", 1000, 1000);
      canvas.cd(1);
      canvas.cd(1)->SetTopMargin(0.10);
      canvas.cd(1)->SetBottomMargin(0.06);
      canvas.cd(1)->SetLeftMargin(0.23);
      canvas.cd(1)->SetRightMargin(0.16);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      // for the "text"-filled histogram
      auto h2_SimBSParameters = std::make_unique<TH2F>("Parameters", "", 1, 0.0, 1.0, END_OF_TYPES, 0, END_OF_TYPES);
      h2_SimBSParameters->SetStats(false);
      h2_SimBSParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_SimBSParameters->GetXaxis()->LabelsOption("h");
      h2_SimBSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_SimBSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_SimBSParameters->SetMarkerSize(1.5);

      // prepare the arrays to fill the histogram
      simBeamSpotPI::SimBSParamsHelper fBS(f_payload);
      simBeamSpotPI::SimBSParamsHelper lBS(l_payload);

#ifdef MM_DEBUG
      std::stringstream ss1, ss2;
      edm::LogPrint("") << "**** first payload";
      fBS.printDebug(ss1);
      edm::LogPrint("") << ss1.str();
      edm::LogPrint("") << "**** last payload";
      lBS.printDebug(ss2);
      edm::LogPrint("") << ss2.str();
#endif

      const auto diffPars = fBS.diffCentralValues(lBS);
      //const auto pullPars = fBS.diffCentralValues(lBS,true /*normalize*/);

      unsigned int yBin = END_OF_TYPES;
      for (int foo = parameters::X; foo <= parameters::timeOffset; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = simBeamSpotPI::getStringFromParamEnum(param, true /*use units*/);
        h2_SimBSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_SimBSParameters->SetBinContent(1, yBin, diffPars[foo]); /* profiting of the parameters enum indexing */
        yBin--;
      }

      // for the "colz"-filled histogram (clonde from the text-based one)
      auto h2_SimBSShadow = (TH2F*)(h2_SimBSParameters->Clone("shadow"));
      h2_SimBSShadow->GetZaxis()->SetTitle("#Delta Parameter(payload A - payload B)");
      h2_SimBSShadow->GetZaxis()->CenterTitle();
      h2_SimBSShadow->GetZaxis()->SetTitleOffset(1.5);

      // this is the fine gradient palette (blue to red)
      double max = h2_SimBSShadow->GetMaximum();
      double min = h2_SimBSShadow->GetMinimum();
      double val_white = 0.;
      double per_white = (max != min) ? ((val_white - min) / (max - min)) : 0.5;

      const int number = 3;
      double Red[number] = {0., 1., 1.};
      double Green[number] = {0., 1., 0.};
      double Blue[number] = {1., 1., 0.};
      double Stops[number] = {0., per_white, 1.};
      int nb = 256;
      h2_SimBSShadow->SetContour(nb);
      TColor::CreateGradientColorTable(number, Stops, Red, Green, Blue, nb);

      h2_SimBSShadow->Draw("colz");
      h2_SimBSParameters->Draw("TEXTsame");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.025);
      ltx.SetTextAlign(11);

      // compute the (run,LS) pairs
      auto l_runLS = beamSpotPI::unpack(std::get<0>(lastiov));
      std::string l_runLSs = "(" + std::to_string(l_runLS.first) + "," + std::to_string(l_runLS.second) + ")";
      auto f_runLS = beamSpotPI::unpack(std::get<0>(firstiov));
      std::string f_runLSs = "(" + std::to_string(f_runLS.first) + "," + std::to_string(f_runLS.second) + ")";

      if (this->m_plotAnnotations.ntags == 2) {
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin(),
            1 - gPad->GetTopMargin() + 0.025,
            (fmt::sprintf(
                 "#splitline{A = #color[4]{%s}: %s}{B = #color[4]{%s}: %s}", f_tagname, f_runLSs, l_tagname, l_runLSs))
                .c_str());
      } else {
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin(),
            1 - gPad->GetTopMargin() + 0.025,
            (fmt::sprintf("#splitline{#color[4]{%s}}{A = %s | B = %s}", f_tagname, l_runLSs, f_runLSs)).c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  protected:
    std::shared_ptr<PayloadType> f_payload;
    std::shared_ptr<PayloadType> l_payload;
  };

}  // namespace simBeamSpotPI

#endif
