#ifndef Alignment_OfflineValidation_DiLeptonVertexHelpers_h
#define Alignment_OfflineValidation_DiLeptonVertexHelpers_h

#include <vector>
#include <string>
#include <fmt/printf.h>
#include "TH2F.h"
#include "TLorentzVector.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace DiLeptonHelp {

  //
  // Ancillary struct for counting
  //
  struct Counts {
    unsigned int eventsTotal;
    unsigned int eventsAfterMult;
    unsigned int eventsAfterPt;
    unsigned int eventsAfterEta;
    unsigned int eventsAfterVtx;
    unsigned int eventsAfterDist;
    
  public:
    void printCounts() {
      std::cout << " Total Events: " << eventsTotal << "\n"
		<< " After multiplicity: " << eventsAfterMult << "\n"
		<< " After pT cut: " << eventsAfterPt << "\n"
		<< " After eta cut: " << eventsAfterEta << "\n"
		<< " After Vtx: " << eventsAfterVtx << "\n"
		<< " After VtxDist: " << eventsAfterDist << std::endl;
    }
    
    void zeroAll() {
      eventsTotal = 0;
      eventsAfterMult = 0;
      eventsAfterPt = 0;
      eventsAfterEta = 0;
      eventsAfterVtx = 0;
      eventsAfterDist = 0;
    }
  };

  //
  // Ancillary class for plotting
  //
  class PlotsVsKinematics {
  public:
    PlotsVsKinematics() : m_name(""), m_title(""), m_ytitle(""), m_isBooked(false) {}
    
    //________________________________________________________________________________//
    // overloaded constructor
    PlotsVsKinematics(const std::string& name, const std::string& tt, const std::string& ytt)
      : m_name(name), m_title(tt), m_ytitle(ytt), m_isBooked(false) {}
    
    ~PlotsVsKinematics() = default;
    
    //________________________________________________________________________________//
    inline void bookFromPSet(const TFileDirectory& fs, const edm::ParameterSet& hpar) {
      std::string namePostfix;
      std::string titlePostfix;
      float xmin, xmax;
      
      for (const auto& xAx : axisChoices) {
	switch (xAx) {
        case xAxis::Z_PHI:
          xmin = -M_PI;
          xmax = M_PI;
          namePostfix = "MuMuPhi";
          titlePostfix = "#mu#mu pair #phi;#mu^{+}#mu^{-} #phi";
          break;
        case xAxis::Z_ETA:
          xmin = -3.5;
          xmax = 3.5;
          namePostfix = "MuMuEta";
          titlePostfix = "#mu#mu pair #eta;#mu^{+}#mu^{-} #eta";
          break;
        case xAxis::MP_PHI:
          xmin = -M_PI;
          xmax = M_PI;
          namePostfix = "MuPlusPhi";
          titlePostfix = "#mu^{+} #phi;#mu^{+} #phi [rad]";
          break;
        case xAxis::MP_ETA:
          xmin = -2.4;
          xmax = 2.4;
          namePostfix = "MuPlusEta";
          titlePostfix = "#mu^{+} #eta;#mu^{+} #eta";
          break;
        case xAxis::MM_PHI:
          xmin = -M_PI;
          xmax = M_PI;
          namePostfix = "MuMinusPhi";
          titlePostfix = "#mu^{-} #phi;#mu^{-} #phi [rad]";
          break;
        case xAxis::MM_ETA:
          xmin = -2.4;
          xmax = 2.4;
          namePostfix = "MuMinusEta";
          titlePostfix = "#mu^{-} #eta;#mu^{+} #eta";
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

	m_h2_map[xAx] = fs.make<TH2F>(h2name.c_str(),
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
    inline void bookPlots(TFileDirectory& fs, const float valmin, const float valmax, const int nxbins, const int nybins) {
      if (m_name.empty() && m_title.empty() && m_ytitle.empty()) {
	edm::LogError("PlotsVsKinematics")
          << "In" << __FUNCTION__ << "," << __LINE__
          << "trying to book plots without the right constructor being called!" << std::endl;
	return;
      }

      static constexpr float maxMuEta = 2.4;
      static constexpr float maxMuMuEta = 3.5;
      TH1F::SetDefaultSumw2(kTRUE);

      // clang-format off
      m_h2_map[xAxis::Z_ETA] = fs.make<TH2F>(fmt::sprintf("%sVsMuMuEta", m_name).c_str(),
					   fmt::sprintf("%s vs #mu#mu pair #eta;#mu^{+}#mu^{-} #eta;%s", m_title, m_ytitle).c_str(),
					     nxbins, -M_PI, M_PI,
					     nybins, valmin, valmax);
      
      m_h2_map[xAxis::Z_PHI] = fs.make<TH2F>(fmt::sprintf("%sVsMuMuPhi", m_name).c_str(),
					     fmt::sprintf("%s vs #mu#mu pair #phi;#mu^{+}#mu^{-} #phi [rad];%s", m_title, m_ytitle).c_str(),
					     nxbins, -maxMuMuEta, maxMuMuEta,
					     nybins, valmin, valmax);
      
      m_h2_map[xAxis::MP_ETA] = fs.make<TH2F>(fmt::sprintf("%sVsMuPlusEta", m_name).c_str(),
					      fmt::sprintf("%s vs #mu^{+} #eta;#mu^{+} #eta;%s", m_title, m_ytitle).c_str(),
					      nxbins, -maxMuEta, maxMuEta,
					      nybins, valmin, valmax);
      
      m_h2_map[xAxis::MP_PHI] = fs.make<TH2F>(fmt::sprintf("%sVsMuPlusPhi", m_name).c_str(),
					      fmt::sprintf("%s vs #mu^{+} #phi;#mu^{+} #phi [rad];%s", m_title, m_ytitle).c_str(),
					      nxbins, -M_PI, M_PI,
					      nybins, valmin, valmax);
      
      m_h2_map[xAxis::MM_ETA] = fs.make<TH2F>(fmt::sprintf("%sVsMuMinusEta", m_name).c_str(),
					      fmt::sprintf("%s vs #mu^{-} #eta;#mu^{-} #eta;%s", m_title, m_ytitle).c_str(),
					      nxbins, -maxMuEta, maxMuEta,
					      nybins, valmin, valmax);
      
      m_h2_map[xAxis::MM_PHI] = fs.make<TH2F>(fmt::sprintf("%sVsMuMinusPhi", m_name).c_str(),
					      fmt::sprintf("%s vs #mu^{-} #phi;#mu^{-} #phi [rad];%s", m_title, m_ytitle).c_str(),
					      nxbins, -M_PI, M_PI,
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
      m_h2_map[xAxis::MP_ETA]->Fill((momenta.first).Eta(), val);
      m_h2_map[xAxis::MP_PHI]->Fill((momenta.first).Phi(), val);
      m_h2_map[xAxis::MM_ETA]->Fill((momenta.second).Eta(), val);
      m_h2_map[xAxis::MM_PHI]->Fill((momenta.second).Phi(), val);
    }
    
  private:
    enum xAxis { Z_PHI, Z_ETA, MP_PHI, MP_ETA, MM_PHI, MM_ETA };
    const std::vector<xAxis> axisChoices = {
      xAxis::Z_PHI, xAxis::Z_ETA, xAxis::MP_PHI, xAxis::MP_ETA, xAxis::MM_PHI, xAxis::MM_ETA};

    const std::string m_name;
    const std::string m_title;
    const std::string m_ytitle;

    bool m_isBooked;

    std::map<xAxis, TH2F*> m_h2_map;
  };
} 
#endif
