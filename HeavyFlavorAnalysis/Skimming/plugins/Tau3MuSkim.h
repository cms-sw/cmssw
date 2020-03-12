#ifndef TAU3MUSKIM
#define TAU3MUSKIM

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class Tau3MuReco;

class Tau3MuSkim : public edm::EDFilter {
public:
  explicit Tau3MuSkim(const edm::ParameterSet&);
  ~Tau3MuSkim() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  Tau3MuReco* m_Tau3MuReco;
};

#endif
