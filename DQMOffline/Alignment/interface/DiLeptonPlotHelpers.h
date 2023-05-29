#ifndef Alignment_OfflineValidation_DiLeptonVertexHelpers_h
#define Alignment_OfflineValidation_DiLeptonVertexHelpers_h

#include <vector>
#include <string>
#include <fmt/printf.h>
#include "TH2F.h"
#include "TLorentzVector.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace DiLepPlotHelp {

  enum flavour { MM = 0, EE = 1, UNDEF = -1 };

  //
  // Ancillary class for plotting
  //
  class PlotsVsKinematics {
  public:
    PlotsVsKinematics(flavour FLAV) : m_name(""), m_title(""), m_ytitle(""), m_isBooked(false), m_flav(FLAV) {}

    //________________________________________________________________________________//
    // overloaded constructor
    PlotsVsKinematics(flavour FLAV, const std::string& name, const std::string& tt, const std::string& ytt)
        : m_name(name), m_title(tt), m_ytitle(ytt), m_isBooked(false), m_flav(FLAV) {
      if (m_flav < 0) {
        edm::LogError("PlotsVsKinematics") << "The initialization flavour is not correct!" << std::endl;
      }
    }

    ~PlotsVsKinematics() = default;

    //________________________________________________________________________________//
    inline void bookFromPSet(dqm::reco::DQMStore::IBooker& iBooker, const edm::ParameterSet& hpar) {
      std::string namePostfix;
      std::string titlePostfix;
      float xmin, xmax;

      std::string sed = (m_flav ? "e" : "#mu");

      for (const auto& xAx : axisChoices) {
        switch (xAx) {
          case xAxis::Z_PHI:
            xmin = -M_PI;
            xmax = M_PI;
            namePostfix = m_flav ? "EEPhi" : "MuMuPhi";
            titlePostfix = fmt::sprintf("%s%s pair #phi;%s^{+}%s^{-} #phi", sed, sed, sed, sed);
            break;
          case xAxis::Z_ETA:
            xmin = -3.5;
            xmax = 3.5;
            namePostfix = m_flav ? "EEEta" : "MuMuEta";
            titlePostfix = fmt::sprintf("%s%s pair #eta;%s^{+}%s^{-} #eta", sed, sed, sed, sed);
            break;
          case xAxis::LP_PHI:
            xmin = -M_PI;
            xmax = M_PI;
            namePostfix = m_flav ? "EPlusPhi" : "MuPlusPhi";
            titlePostfix = fmt::sprintf("%s^{+} #phi;%s^{+} #phi [rad]", sed, sed);
            break;
          case xAxis::LP_ETA:
            xmin = -2.4;
            xmax = 2.4;
            namePostfix = m_flav ? "EPlusEta" : "MuPlusEta";
            titlePostfix = fmt::sprintf("%s^{+} #eta;%s^{+} #eta", sed, sed);
            break;
          case xAxis::LM_PHI:
            xmin = -M_PI;
            xmax = M_PI;
            namePostfix = m_flav ? "EMinusPhi" : "MuMinusPhi";
            titlePostfix = fmt::sprintf("%s^{-} #phi;%s^{-} #phi [rad]", sed, sed);
            break;
          case xAxis::LM_ETA:
            xmin = -2.4;
            xmax = 2.4;
            namePostfix = m_flav ? "EMinusEta" : "MuMinusEta";
            titlePostfix = fmt::sprintf("%s^{-} #eta;%s^{+} #eta", sed, sed);
            break;
          case xAxis::DELTA_ETA:
            xmin = -hpar.getParameter<double>("maxDeltaEta");
            xmax = hpar.getParameter<double>("maxDeltaEta");
            namePostfix = m_flav ? "EEDeltEta" : "MuMuDeltaEta";
            titlePostfix = fmt::sprintf("%s^{-}%s^{+} #Delta#eta;%s^{+}%s^{-} #Delta#eta", sed, sed, sed, sed);
            break;
          case xAxis::COSTHETACS:
            xmin = -1.;
            xmax = 1.;
            namePostfix = "CosThetaCS";
            titlePostfix =
                fmt::sprintf("%s^{+}%s^{-} cos(#theta_{CS});%s^{+}%s^{-} cos(#theta_{CS})", sed, sed, sed, sed);
            break;
          default:
            throw cms::Exception("LogicalError") << " there is not such Axis choice as " << xAx;
        }

        const auto& h2name = fmt::sprintf("%sVs%s", hpar.getParameter<std::string>("name"), namePostfix);
        const auto& h2title = fmt::sprintf("%s vs %s;%s% s",
                                           hpar.getParameter<std::string>("title"),
                                           titlePostfix,
                                           hpar.getParameter<std::string>("title"),
                                           hpar.getParameter<std::string>("yUnits"));

        m_h2_map[xAx] = iBooker.book2D(h2name.c_str(),
                                       h2title.c_str(),
                                       hpar.getParameter<int32_t>("NxBins"),
                                       xmin,
                                       xmax,
                                       hpar.getParameter<int32_t>("NyBins"),
                                       hpar.getParameter<double>("ymin"),
                                       hpar.getParameter<double>("ymax"));
      }

      // flip the is booked bit
      m_isBooked = true;
    }

    //________________________________________________________________________________//
    inline void bookPlots(dqm::reco::DQMStore::IBooker& iBooker,
                          const float valmin,
                          const float valmax,
                          const int nxbins,
                          const int nybins) {
      if (m_name.empty() && m_title.empty() && m_ytitle.empty()) {
        edm::LogError("PlotsVsKinematics")
            << "In" << __FUNCTION__ << "," << __LINE__
            << "trying to book plots without the right constructor being called!" << std::endl;
        return;
      }

      std::string dilep = (m_flav ? "e^{+}e^{-}" : "#mu^{+}#mu^{-}");
      std::string lep = (m_flav ? "e^{+}" : "#mu^{+}");
      std::string lem = (m_flav ? "e^{-}" : "#mu^{-}");

      static constexpr float maxMuEta = 2.4;
      static constexpr float maxMuMuEta = 3.5;
      TH1F::SetDefaultSumw2(kTRUE);

      // clang-format off
      m_h2_map[xAxis::Z_ETA] = iBooker.book2D(fmt::sprintf("%sVsMuMuEta", m_name).c_str(),
					      fmt::sprintf("%s vs %s pair #eta;%s #eta;%s", m_title, dilep, dilep, m_ytitle).c_str(),
					      nxbins, -M_PI, M_PI,
					      nybins, valmin, valmax);
      
      m_h2_map[xAxis::Z_PHI] = iBooker.book2D(fmt::sprintf("%sVsMuMuPhi", m_name).c_str(),
					      fmt::sprintf("%s vs %s pair #phi;%s #phi [rad];%s", m_title, dilep, dilep, m_ytitle).c_str(),
					      nxbins, -maxMuMuEta, maxMuMuEta,
					      nybins, valmin, valmax);
      
      m_h2_map[xAxis::LP_ETA] = iBooker.book2D(fmt::sprintf("%sVsMuPlusEta", m_name).c_str(),
					       fmt::sprintf("%s vs %s #eta;%s #eta;%s", m_title, lep, lep, m_ytitle).c_str(),
					       nxbins, -maxMuEta, maxMuEta,
					       nybins, valmin, valmax);
      
      m_h2_map[xAxis::LP_PHI] = iBooker.book2D(fmt::sprintf("%sVsMuPlusPhi", m_name).c_str(),
					       fmt::sprintf("%s vs %s #phi;%s #phi [rad];%s", m_title, lep, lep, m_ytitle).c_str(),
					       nxbins, -M_PI, M_PI,
					       nybins, valmin, valmax);
      
      m_h2_map[xAxis::LM_ETA] = iBooker.book2D(fmt::sprintf("%sVsMuMinusEta", m_name).c_str(),
					       fmt::sprintf("%s vs %s #eta;%s #eta;%s", m_title, lem, lem, m_ytitle).c_str(),
					       nxbins, -maxMuEta, maxMuEta,
					       nybins, valmin, valmax);
      
      m_h2_map[xAxis::LM_PHI] = iBooker.book2D(fmt::sprintf("%sVsMuMinusPhi", m_name).c_str(),
					       fmt::sprintf("%s vs %s #phi;%s #phi [rad];%s", m_title, lem, lem,  m_ytitle).c_str(),
					       nxbins, -M_PI, M_PI,
					       nybins, valmin, valmax);

      m_h2_map[xAxis::DELTA_ETA] = iBooker.book2D(fmt::sprintf("%sVsMuMuDeltaEta", m_name).c_str(),
						  fmt::sprintf("%s vs %s #Delta#eta;%s #Delta#eta;%s", m_title, dilep, dilep,  m_ytitle).c_str(),
						  nxbins, -4., 4.,
						  nybins, valmin, valmax);

      m_h2_map[xAxis::COSTHETACS] = iBooker.book2D(fmt::sprintf("%sVsCosThetaCS", m_name).c_str(),
						   fmt::sprintf("%s vs %s cos(#theta_{CS});%s cos(#theta_{CS});%s", m_title, dilep, dilep,  m_ytitle).c_str(),
						   nxbins, -1., 1.,
						   nybins, valmin, valmax);
      // clang-format on

      // flip the is booked bit
      m_isBooked = true;
    }

    //________________________________________________________________________________//
    inline void fillPlots(const float val, const std::pair<TLorentzVector, TLorentzVector>& momenta) {
      if (!m_isBooked) {
        edm::LogError("PlotsVsKinematics")
            << "In" << __FUNCTION__ << "," << __LINE__ << "trying to fill a plot not booked!" << std::endl;
        return;
      }

      m_h2_map[xAxis::Z_ETA]->Fill((momenta.first + momenta.second).Eta(), val);
      m_h2_map[xAxis::Z_PHI]->Fill((momenta.first + momenta.second).Phi(), val);
      m_h2_map[xAxis::LP_ETA]->Fill((momenta.first).Eta(), val);
      m_h2_map[xAxis::LP_PHI]->Fill((momenta.first).Phi(), val);
      m_h2_map[xAxis::LM_ETA]->Fill((momenta.second).Eta(), val);
      m_h2_map[xAxis::LM_PHI]->Fill((momenta.second).Phi(), val);

      // follows here kinematics
      double deltaEta = (momenta.first).Eta() - (momenta.second).Eta();

      double muplus = 1. / sqrt(2.) * (momenta.first.E() + momenta.first.Z());
      double muminus = 1. / sqrt(2.) * (momenta.first.E() - momenta.first.Z());
      double mubarplus = 1. / sqrt(2.) * (momenta.second.E() + momenta.second.Z());
      double mubarminus = 1. / sqrt(2.) * (momenta.second.E() - momenta.second.Z());

      const auto& mother = momenta.first + momenta.second;
      double cosThetaCS = 2. / mother.Mag() / sqrt(pow(mother.Mag(), 2) + pow(mother.Pt(), 2)) *
                          (muplus * mubarminus - muminus * mubarplus);

      m_h2_map[xAxis::DELTA_ETA]->Fill(deltaEta, val);
      m_h2_map[xAxis::COSTHETACS]->Fill(cosThetaCS, val);
    }

  private:
    enum xAxis { Z_PHI, Z_ETA, LP_PHI, LP_ETA, LM_PHI, LM_ETA, DELTA_ETA, COSTHETACS };
    const std::vector<xAxis> axisChoices = {xAxis::Z_PHI,
                                            xAxis::Z_ETA,
                                            xAxis::LP_PHI,
                                            xAxis::LP_ETA,
                                            xAxis::LM_PHI,
                                            xAxis::LM_ETA,
                                            xAxis::DELTA_ETA,
                                            xAxis::COSTHETACS};

    const std::string m_name;
    const std::string m_title;
    const std::string m_ytitle;

    bool m_isBooked;
    flavour m_flav;

    std::map<xAxis, dqm::reco::MonitorElement*> m_h2_map;
  };
}  // namespace DiLepPlotHelp
#endif
