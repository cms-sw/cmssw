#ifndef UCTCard_hh
#define UCTCard_hh

#include <vector>

#include "UCTGeometry.hh"

class UCTRegion;

class UCTCard {
public:

  UCTCard(uint32_t crt, uint32_t crd);

  virtual ~UCTCard();

  // To set up event data before processing

  const std::vector<UCTRegion*>& getRegions() const {return regions;}
  const UCTRegion* getRegion(uint32_t rgn) const {return regions[rgn];}
  const UCTRegion* getRegion(UCTRegionIndex r) const;

  // To process event

  bool clearEvent();
  bool setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET);
  bool setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET);
  bool process();

  // More access functions

  const uint32_t getCrate() const {return crate;}
  const uint32_t getCard() const {return card;}

  const uint32_t et() const {return cardSummary;}

  friend std::ostream& operator<<(std::ostream&, const UCTCard&);

private:

  // No default constructor is needed

  UCTCard();

  // No copy constructor is needed

  UCTCard(const UCTCard&);

  // No equality operator is needed

  const UCTCard& operator=(const UCTCard&);

  // Helper functions

  const UCTRegion* getRegion(bool negativeEta, uint32_t caloEta, uint32_t caloPhi) const;

  // Owned card level data 

  uint32_t crate;
  uint32_t card;

  std::vector<UCTRegion*> regions;

  uint32_t cardSummary;

};

#endif
