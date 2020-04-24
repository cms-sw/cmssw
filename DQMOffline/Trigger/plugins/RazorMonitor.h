// -----------------------------
//
// Offline DQM for razor triggers. The razor inclusive analysis measures trigger efficiency
// in SingleElectron events (orthogonal to analysis), as a 2D function of the razor variables
// M_R and R^2. Also monitor dPhi_R, used offline for  QCD and/or detector-related MET tail
// rejection.
// Based on DQMOffline/Trigger/plugins/METMonitor.*
//
// ----------------------------- 
#ifndef DQMOFFLINE_TRIGGER_RAZORMONITOR_H
#define DQMOFFLINE_TRIGGER_RAZORMONITOR_H

#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//DataFormats
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//Hemispheres                                                                                                                                                                  
#include "HLTrigger/JetMET/interface/HLTRHemisphere.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TLorentzVector.h"

class GenericTriggerEventFlag;

struct RazorME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};
//
// class declaration
//

class RazorMonitor : public DQMEDAnalyzer 
{
public:
  RazorMonitor( const edm::ParameterSet& );
  ~RazorMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  static double CalcMR(const math::XYZTLorentzVector& ja, const math::XYZTLorentzVector& jb);
  static double CalcR(double MR, const math::XYZTLorentzVector& ja, const math::XYZTLorentzVector& jb, const edm::Handle<std::vector<reco::PFMET> >& met);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(RazorME& me, const std::string& titleX, const std::string& titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       jetToken_;
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector> > theHemispheres_;
  
  std::vector<double> rsq_binning_;
  std::vector<double> mr_binning_;
  std::vector<double> dphiR_binning_;

  RazorME MR_ME_;
  RazorME Rsq_ME_;
  RazorME dPhiR_ME_;
  RazorME MRVsRsq_ME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>      metSelection_;
  StringCutObjectSelector<reco::PFJet,true >   jetSelection_;
  unsigned int njets_;
  float rsqCut_;
  float mrCut_;

};

#endif // DQMOFFLINE_TRIGGER_RAZORMONITOR_H
