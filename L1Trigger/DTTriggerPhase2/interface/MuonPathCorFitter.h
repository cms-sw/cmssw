#ifndef L1Trigger_DTTriggerPhase2_MuonPathCorFitter_h
#define L1Trigger_DTTriggerPhase2_MuonPathCorFitter_h

#include "L1Trigger/DTTriggerPhase2/interface/MuonPathFitter.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

inline bool bxSort(const cmsdt::bx_sl_vector &vA, const cmsdt::bx_sl_vector &vB) {
  if (vA.bx > vB.bx)
    return true;
  else if (vA.bx == vB.bx)
    return (vA.sl > vB.sl);
  else
    return false;
}

using mp_group = std::vector<cmsdt::metaPrimitive>;

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathCorFitter : public MuonPathFitter {
public:
  // Constructors and destructor
  MuonPathCorFitter(const edm::ParameterSet &pset,
                    edm::ConsumesCollector &iC,
                    std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer);
  ~MuonPathCorFitter() override;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<cmsdt::metaPrimitive> &metaPrimitives) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<lat_vector> &lateralities,
           std::vector<cmsdt::metaPrimitive> &metaPrimitives) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           MuonPathPtrs &outMPath) override{};

  void finish() override;

  // Other public methods

  // luts
  edm::FileInPath both_sl_filename_;

private:
  // Private methods
  void analyze(mp_group mp, std::vector<cmsdt::metaPrimitive> &metaPrimitives);
  void fillLuts();
  int get_rom_addr(mp_group mps, std::vector<int> missing_hits);
  bool canCorrelate(cmsdt::metaPrimitive mp_sl1, cmsdt::metaPrimitive mp_sl3);

  // Private attributes
  int dT0_correlate_TP_;

  // double chi2Th_;
  std::vector<std::vector<int>> lut_2sl;
};

#endif
