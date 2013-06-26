#ifndef GetLumi_H
#define GetLumi_H

// system include files
#include <memory> 

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GetLumi
{
 public:

  enum SQRT_S{
    SQRT_S_7TeV,
    SQRT_S_8TeV
  };

  static const unsigned int NUM_BX = 3564;
  static constexpr double FREQ_ORBIT = 11246.; // Hz
  static constexpr double SECONDS_PER_LS = double(0x40000)/double(FREQ_ORBIT);

  static constexpr double INELASTIC_XSEC_7TeV = 68.0; // mb
  static constexpr double INELASTIC_XSEC_8TeV = 69.3; // mb

  GetLumi(const edm::ParameterSet&);
  GetLumi(const edm::InputTag&, double);
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


};
#endif
