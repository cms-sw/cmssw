#ifndef GetLumi_H
#define GetLumi_H

// system include files
#include <memory> 

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class LumiDetails;
class LumiSummary;

class GetLumi
{
 public:

  enum SQRT_S{
    SQRT_S_7TeV,
    SQRT_S_8TeV,
    SQRT_S_13TeV
  };

  static const unsigned int NUM_BX = 3564;
  static constexpr double FREQ_ORBIT = 11246.; // Hz
  static constexpr double SECONDS_PER_LS = double(0x40000)/double(FREQ_ORBIT);

  static constexpr double INELASTIC_XSEC_7TeV  = 68.0; // mb
  static constexpr double INELASTIC_XSEC_8TeV  = 69.3; // mb
  static constexpr double INELASTIC_XSEC_13TeV = 71.3; // mb from http://inspirehep.net/record/1447965/files/FSQ-15-005-pas.pdf

  // from http://cmslxr.fnal.gov/source/DQM/PixelLumi/plugins/PixelLumiDQM.h
  // Using all pixel clusters:
  static constexpr double XSEC_PIXEL_CLUSTER = 10.08e-24; //in cm^2
  static constexpr double XSEC_PIXEL_CLUSTER_UNC = 0.17e-24;
  
  // Excluding the inner barrel layer.
  static constexpr double rXSEC_PIXEL_CLUSTER = 9.4e-24; //in cm^2
  static constexpr double rXSEC_PIXEL_CLUSTER_UNC = 0.119e-24;
  static constexpr double CM2_TO_NANOBARN = 1.0/1.e-33;
  static const unsigned int lastBunchCrossing = 3564;

  GetLumi(const edm::ParameterSet&);
  GetLumi(const edm::InputTag&, double);
  GetLumi(const edm::ParameterSet&,edm::ConsumesCollector& iC);
  GetLumi(const edm::InputTag&, double,edm::ConsumesCollector& iC);
  virtual ~GetLumi();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  double getRawValue(const edm::Event&);
  double getValue   (const edm::Event&);

  double getRawValue(edm::LuminosityBlock const&,edm::EventSetup const&);
  double getValue   (edm::LuminosityBlock const&,edm::EventSetup const&);

  double convert2PU(double,double);
  double convert2PU(double,int);

  // ----------member data ---------------------------
  edm::InputTag     lumiInputTag_;
  double            lumiScale_;

  edm::EDGetTokenT<LumiDetails> lumiDetailsToken_;
  edm::EDGetTokenT<LumiSummary> lumiSummaryToken_;

};
#endif
