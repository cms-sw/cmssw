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
  GetLumi(const edm::ParameterSet&);
  GetLumi(edm::InputTag, double);
  virtual ~GetLumi();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  double getRawValue(const edm::Event&);
  double getValue   (const edm::Event&);

  // ----------member data ---------------------------
  edm::InputTag     lumiInputTag_;
  double            lumiScale_;


};
#endif
