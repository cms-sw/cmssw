#ifndef UCTSummaryCard_hh
#define UCTSummaryCard_hh

#include <vector>
#include <list>
#include <memory>

#include "UCTGeometryExtended.hh"

class UCTLayer1;
class UCTObject;
class UCTRegion;

class UCTSummaryCard {
public:
  UCTSummaryCard(const std::vector<std::vector<std::vector<uint32_t>>>* l,
                 uint32_t jetSeedIn = 10,
                 uint32_t tauSeedIn = 10,
                 double tauIsolationFactorIn = 0.3,
                 uint32_t eGammaSeedIn = 5,
                 double eGammaIsolationFactorIn = 0.3);

  // No copy constructor is needed

  UCTSummaryCard(const UCTSummaryCard&) = delete;

  // No equality operator is needed

  const UCTSummaryCard& operator=(const UCTSummaryCard&) = delete;

  virtual ~UCTSummaryCard() = default;

  // To set up event data before processing

  const UCTRegion* getRegion(int regionEtaIndex, uint32_t regionPhiIndex) const;

  // UCTSummaryCard process event

  bool clearEvent();
  bool clearRegions();
  bool setRegionData(
      std::vector<UCTRegion> inputRegions);  // Use when the region collection is available and no direct access to TPGs
  bool process();

  // Access to data

  const std::list<std::shared_ptr<UCTObject>>& getEMObjs() { return emObjs; }
  const std::list<std::shared_ptr<UCTObject>>& getIsoEMObjs() { return isoEMObjs; }
  const std::list<std::shared_ptr<UCTObject>>& getTauObjs() { return tauObjs; }
  const std::list<std::shared_ptr<UCTObject>>& getIsoTauObjs() { return isoTauObjs; }
  const std::list<std::shared_ptr<UCTObject>>& getCentralJetObjs() { return centralJetObjs; }
  const std::list<std::shared_ptr<UCTObject>>& getForwardJetObjs() { return forwardJetObjs; }
  const std::list<std::shared_ptr<UCTObject>>& getBoostedJetObjs() { return boostedJetObjs; }

  const std::shared_ptr<UCTObject> getET() { return ET; }
  const std::shared_ptr<UCTObject> getMET() { return MET; }

  const std::shared_ptr<UCTObject> getHT() { return HT; }
  const std::shared_ptr<UCTObject> getMHT() { return MHT; }

  void print();

private:
  // Helper functions

  bool processRegion(UCTRegionIndex regionIndex);

  // Parameters specified at constructor level

  //  const UCTLayer1 *uctLayer1;
  const std::vector<std::vector<std::vector<uint32_t>>>* pumLUT;
  uint32_t jetSeed;
  uint32_t tauSeed;
  double tauIsolationFactor;
  uint32_t eGammaSeed;
  double eGammaIsolationFactor;

  // Owned card level data

  std::vector<UCTRegion> regions;

  double sinPhi[73];  // Make one extra so caloPhi : 1-72 can be used as index directly
  double cosPhi[73];

  std::list<std::shared_ptr<UCTObject>> emObjs;
  std::list<std::shared_ptr<UCTObject>> isoEMObjs;
  std::list<std::shared_ptr<UCTObject>> tauObjs;
  std::list<std::shared_ptr<UCTObject>> isoTauObjs;
  std::list<std::shared_ptr<UCTObject>> centralJetObjs;
  std::list<std::shared_ptr<UCTObject>> forwardJetObjs;
  std::list<std::shared_ptr<UCTObject>> boostedJetObjs;

  std::shared_ptr<UCTObject> ET;
  std::shared_ptr<UCTObject> MET;

  std::shared_ptr<UCTObject> HT;
  std::shared_ptr<UCTObject> MHT;

  uint32_t cardSummary;

  // Pileup subtraction LUT based on multiplicity of regions > threshold

  uint32_t pumLevel;
  uint32_t pumBin;
};

#endif
