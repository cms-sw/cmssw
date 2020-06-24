#ifndef Phase2L1Trigger_DTTrigger_MPQualityEnhancerFilter_h
#define Phase2L1Trigger_DTTrigger_MPQualityEnhancerFilter_h

#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MPQualityEnhancerFilter : public MPFilter {
public:
  // Constructors and destructor
  MPQualityEnhancerFilter(const edm::ParameterSet &pset);
  ~MPQualityEnhancerFilter() override;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<metaPrimitive> &inMPath,
           std::vector<metaPrimitive> &outMPath) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMPath,
           MuonPathPtrs &outMPath) override{};

  void finish() override;

  // Other public methods

  // Public attributes
  int areCousins(metaPrimitive mp1, metaPrimitive mp2);
  int rango(metaPrimitive mp);
  void printmP(metaPrimitive mP);

private:
  // Private methods
  void filterCousins(std::vector<metaPrimitive> &inMPath, std::vector<metaPrimitive> &outMPath);
  void refilteringCousins(std::vector<metaPrimitive> &inMPath, std::vector<metaPrimitive> &outMPath);
  void filterTanPhi(std::vector<metaPrimitive> &inMPath, std::vector<metaPrimitive> &outMPath);
  void filterUnique(std::vector<metaPrimitive> &inMPath, std::vector<metaPrimitive> &outMPath);

  // Private attributes
  bool debug_;
  bool filter_cousins_;
};

#endif
