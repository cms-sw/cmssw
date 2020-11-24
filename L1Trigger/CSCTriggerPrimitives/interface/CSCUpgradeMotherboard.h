#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboard_h

/** \class CSCUpgradeMotherboard
 *
 * Base class for upgrade TMBs (MEX/1) chambers, that either run the
 * upgrade CSC-only TMB algorithm or the CSC-GEM algorithm
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeAnodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeCathodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/LCTContainer.h"

// generic container type
namespace {

  // first: raw detid, second: digi
  template <class T>
  using match = std::pair<unsigned int, T>;

  // vector of template above
  template <class T>
  using matches = std::vector<std::pair<unsigned int, T> >;

  // first: BX number, second: vector of template above
  template <class T>
  using matchesBX = std::map<int, std::vector<std::pair<unsigned int, T> > >;

}  // namespace

class CSCUpgradeMotherboard : public CSCMotherboard {
public:
  // standard constructor
  CSCUpgradeMotherboard(unsigned endcap,
                        unsigned station,
                        unsigned sector,
                        unsigned subsector,
                        unsigned chamber,
                        const edm::ParameterSet& conf);

  //Default constructor for testing
  CSCUpgradeMotherboard();

  ~CSCUpgradeMotherboard() override;

  // Empty the LCT container
  void clear();

  // Compare two matches of type <ID,DIGI>
  // The template is match<GEMPadDigi> or match<GEMCoPadDigi>
  template <class S>
  bool compare(const S& p, const S& q) const;

  // Get the common matches of type <ID,DIGI>. Could be more than 1
  // The template is matches<GEMPadDigi> or matches<GEMCoPadDigi>
  template <class S>
  void intersection(const S& d1, const S& d2, S& result) const;

  /** Methods to sort the LCTs */
  static bool sortLCTsByQuality(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&);
  static bool sortLCTsByGEMDphi(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&);
  // generic sorting function
  // provide an LCT collection and a sorting function
  void sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts,
                bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const;

  /** get CSCPart from HS, station, ring number **/
  enum CSCPart getCSCPart(int keystrip) const;

  // run TMB with GEM pad clusters as input
  void run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc) override;

  /* readout the two best LCTs in this CSC */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs() const override;

protected:
  void correlateLCTs(const CSCALCTDigi& bestALCT,
                     const CSCALCTDigi& secondALCT,
                     const CSCCLCTDigi& bestCLCT,
                     const CSCCLCTDigi& secondCLCT,
                     CSCCorrelatedLCTDigi& lct1,
                     CSCCorrelatedLCTDigi& lct2) const;

  Parity theParity;

  void setPrefIndex();

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  LCTContainer allLCTs;

  /** "preferential" index array in matching window for cross-BX sorting */
  int pref[CSCConstants::MAX_LCT_TBINS];

  bool match_earliest_alct_only;
  bool match_earliest_clct_only;

  /* type of algorithm to sort the stubs */
  unsigned int tmb_cross_bx_algo;

  /** maximum lcts per BX in MEX1: 2, 3, 4 or 999 */
  unsigned int max_lcts;

  // debug gem matching
  bool debug_matching;

  // check look-up-tables
  bool debug_luts;
};

template <class S>
bool CSCUpgradeMotherboard::compare(const S& p, const S& q) const {
  return (p.first == q.first) and (p.second == q.second);
}

template <class S>
void CSCUpgradeMotherboard::intersection(const S& d1, const S& d2, S& result) const {
  for (const auto& p : d1) {
    for (const auto& q : d2) {
      if (compare(p, q)) {
        result.push_back(p);
      }
    }
  }
}

#endif
