#ifndef Phase2L1Trigger_DTTrigger_MPQualityEnhancerFilterBayes_h
#define Phase2L1Trigger_DTTrigger_MPQualityEnhancerFilterBayes_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPQualityEnhancerFilterBayes : public MPFilter {
public:
  // Constructors and destructor
  MPQualityEnhancerFilterBayes(const edm::ParameterSet &pset);
  ~MPQualityEnhancerFilterBayes() override = default;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPath,
           std::vector<cmsdt::metaPrimitive> &outMPath) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMPath,
           MuonPathPtrs &outMPath) override{};

  void finish() override;

  // Other public methods

  // Public attributes
  int areCousins(cmsdt::metaPrimitive mp1, cmsdt::metaPrimitive mp2);
  int shareSL(cmsdt::metaPrimitive mp1, cmsdt::metaPrimitive mp2);
  bool areSame(cmsdt::metaPrimitive mp1, cmsdt::metaPrimitive mp2);
  int rango(cmsdt::metaPrimitive mp);
  int BX(cmsdt::metaPrimitive mp);
  void printmP(cmsdt::metaPrimitive mP);

private:
  // Private methods
  void filterCousins(std::vector<cmsdt::metaPrimitive> &inMPath, std::vector<cmsdt::metaPrimitive> &outMPath);

  // Private attributes
  const bool debug_;
};

#endif
