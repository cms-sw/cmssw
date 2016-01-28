#ifndef UCTCrate_hh
#define UCTCrate_hh

#include <vector>

#include "UCTGeometry.hh"

class UCTCard;

class UCTCrate {
public:

  UCTCrate(uint32_t crt);

  virtual ~UCTCrate();

  // To set up event data before processing

  const std::vector<UCTCard*>& getCards() {return cards;}
  const UCTCard* getCard(uint32_t crd) const {return cards[crd];}
  const UCTCard* getCard(UCTTowerIndex t) const;
  const UCTCard* getCard(UCTRegionIndex r) const {
    UCTGeometry g;
    return getCard(g.getUCTTowerIndex(r));
  }

  // To process event

  bool clearEvent();
  bool setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET);
  bool setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET);
  bool process();

  // More access functions

  const uint32_t getCrate() const {return crate;}
  const uint32_t getCrateSummary() const {return crateSummary;}


  const uint32_t et() const {return crateSummary;}

  friend std::ostream& operator<<(std::ostream&, const UCTCrate&);

private:

  // No default constructor is needed

  UCTCrate();

  // No copy constructor is needed

  UCTCrate(const UCTCrate&);

  // No equality operator is needed

  const UCTCrate& operator=(const UCTCrate&);

  // Owned crate level data 

  uint32_t crate;
  std::vector<UCTCard*> cards;
  uint32_t crateSummary;

};

#endif
