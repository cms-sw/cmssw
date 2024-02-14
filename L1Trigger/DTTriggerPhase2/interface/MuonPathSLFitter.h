#ifndef L1Trigger_DTTriggerPhase2_MuonPathSLFitter_h
#define L1Trigger_DTTriggerPhase2_MuonPathSLFitter_h

#include "L1Trigger/DTTriggerPhase2/interface/MuonPathFitter.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathSLFitter : public MuonPathFitter {
public:
  // Constructors and destructor
  MuonPathSLFitter(const edm::ParameterSet &pset,
                   edm::ConsumesCollector &iC,
                   std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer);
  ~MuonPathSLFitter() override;

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
           std::vector<cmsdt::metaPrimitive> &metaPrimitives) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           MuonPathPtrs &outMPath) override{};

  void finish() override;

  // Other public methods

  //shift theta
  edm::FileInPath shift_theta_filename_;
  std::map<int, float> shiftthetainfo_;

  // luts
  edm::FileInPath sl1_filename_;
  edm::FileInPath sl2_filename_;
  edm::FileInPath sl3_filename_;

private:
  // Private methods
  void analyze(MuonPathPtr &inMPath, lat_vector lat_combs, std::vector<cmsdt::metaPrimitive> &metaPrimitives);
  void fillLuts();
  int get_rom_addr(MuonPathPtr &inMPath, latcomb lats);

  // Private attributes

  // double chi2Th_;
  std::vector<std::vector<int>> lut_sl1;
  std::vector<std::vector<int>> lut_sl2;
  std::vector<std::vector<int>> lut_sl3;
};

#endif
