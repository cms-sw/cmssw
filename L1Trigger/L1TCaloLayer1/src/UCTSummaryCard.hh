#ifndef UCTSummaryCard_hh
#define UCTSummaryCard_hh

#include <vector>
#include <list>

#include "UCTGeometryExtended.hh"

class UCTLayer1;
class UCTObject;
class UCTRegion;

class UCTSummaryCard {
public:
  UCTSummaryCard(const std::vector<std::vector<std::vector<uint32_t> > >* l,
                 uint32_t jetSeedIn = 10,
                 uint32_t tauSeedIn = 10,
                 double tauIsolationFactorIn = 0.3,
                 uint32_t eGammaSeedIn = 5,
                 double eGammaIsolationFactorIn = 0.3);

  virtual ~UCTSummaryCard();

  // To set up event data before processing

  const UCTRegion* getRegion(int regionEtaIndex, uint32_t regionPhiIndex) const;

  // UCTSummaryCard process event

  bool clearEvent();
  bool clearRegions();
  bool setRegionData(
      std::vector<UCTRegion*> inputRegions);  // Use when the region collection is available and no direct access to TPGs
  bool process();

  // Access to data

  const std::list<UCTObject*>& getEMObjs() { return emObjs; }
  const std::list<UCTObject*>& getIsoEMObjs() { return isoEMObjs; }
  const std::list<UCTObject*>& getTauObjs() { return tauObjs; }
  const std::list<UCTObject*>& getIsoTauObjs() { return isoTauObjs; }
  const std::list<UCTObject*>& getCentralJetObjs() { return centralJetObjs; }
  const std::list<UCTObject*>& getForwardJetObjs() { return forwardJetObjs; }
  const std::list<UCTObject*>& getBoostedJetObjs() { return boostedJetObjs; }

  const UCTObject* getET() { return ET; }
  const UCTObject* getMET() { return MET; }

  const UCTObject* getHT() { return HT; }
  const UCTObject* getMHT() { return MHT; }

  void print();

private:
  // No copy constructor is needed

  UCTSummaryCard(const UCTSummaryCard&) = delete;

  // No equality operator is needed

  const UCTSummaryCard& operator=(const UCTSummaryCard&) = delete;

  // Helper functions

  bool processRegion(UCTRegionIndex regionIndex);

  // Parameters specified at constructor level

  //  const UCTLayer1 *uctLayer1;
  const std::vector<std::vector<std::vector<uint32_t> > >* pumLUT;
  uint32_t jetSeed;
  uint32_t tauSeed;
  double tauIsolationFactor;
  uint32_t eGammaSeed;
  double eGammaIsolationFactor;

  // Owned card level data

  std::vector<UCTRegion*> regions;

  double sinPhi[73];  // Make one extra so caloPhi : 1-72 can be used as index directly
  double cosPhi[73];

  std::list<UCTObject*> emObjs;
  std::list<UCTObject*> isoEMObjs;
  std::list<UCTObject*> tauObjs;
  std::list<UCTObject*> isoTauObjs;
  std::list<UCTObject*> centralJetObjs;
  std::list<UCTObject*> forwardJetObjs;
  std::list<UCTObject*> boostedJetObjs;

  UCTObject* ET;
  UCTObject* MET;

  UCTObject* HT;
  UCTObject* MHT;

  uint32_t cardSummary;

  // Pileup subtraction LUT based on multiplicity of regions > threshold

  uint32_t pumLevel;
  uint32_t pumBin;
};

#endif
