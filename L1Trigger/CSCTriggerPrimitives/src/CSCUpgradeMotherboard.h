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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUT.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUTGenerator.h"

// generic container type
namespace{

// first: raw detid, second: digi
template <class T>
using match = std::pair<unsigned int, T>;

// vector of template above
template <class T>
using matches = std::vector<std::pair<unsigned int, T> >;

// first: BX number, second: vector of template above
template <class T>
using matchesBX = std::map<int, std::vector<std::pair<unsigned int, T> > >;

}

class CSCGeometry;
class CSCChamber;

class CSCUpgradeMotherboard : public CSCMotherboard
{
public:

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  class LCTContainer {
  public:
    // constructor
    LCTContainer(unsigned int trig_window_size);

    // access the LCT in a particular ALCT BX, a particular CLCT matched BX
    // and particular LCT number
    CSCCorrelatedLCTDigi& operator()(int bx, int match_bx, int lct);

    // get the matching LCTs for a certain ALCT BX
    void getTimeMatched(const int bx, std::vector<CSCCorrelatedLCTDigi>&) const;

    // get all LCTs in the 16 BX readout window
    void getMatched(std::vector<CSCCorrelatedLCTDigi>&) const;

    // array with stored LCTs
    CSCCorrelatedLCTDigi data[CSCConstants::MAX_LCT_TBINS][15][2];

    // matching trigger window
    const unsigned int match_trig_window_size;
  };

  // standard constructor
  CSCUpgradeMotherboard(unsigned endcap, unsigned station, unsigned sector,
                        unsigned subsector, unsigned chamber,
                        const edm::ParameterSet& conf);

   //Default constructor for testing
  CSCUpgradeMotherboard();

  ~CSCUpgradeMotherboard() override;

  // Compare two matches of type <ID,DIGI>
  // The template is match<GEMPadDigi> or match<GEMCoPadDigi>
  template <class S>
  bool compare(const S& p, const S& q) const;

  // Get the common matches of type <ID,DIGI>. Could be more than 1
  // The template is matches<GEMPadDigi> or matches<GEMCoPadDigi>
  template <class S>
  void intersection(const S& d1, const S& d2, S& result) const;

  /** Methods to sort the LCTs */
  static bool sortLCTsByQuality(const CSCCorrelatedLCTDigi&,
                                const CSCCorrelatedLCTDigi&);
  static bool sortLCTsByGEMDphi(const CSCCorrelatedLCTDigi&,
                                const CSCCorrelatedLCTDigi&);
  // generic sorting function
  // provide an LCT collection and a sorting function
  void sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts,
                bool (*sorter)(const CSCCorrelatedLCTDigi&,
                               const CSCCorrelatedLCTDigi&)) const;

  // functions to setup geometry and LUTs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setupGeometry();
  void debugLUTs();

 protected:

  /** Chamber id (trigger-type labels). */
  unsigned theRegion;
  unsigned theChamber;
  Parity par;

  edm::ParameterSet tmbParams_;
  edm::ParameterSet commonParams_;

  const CSCGeometry* csc_g;
  const CSCChamber* cscChamber;

  std::vector<CSCALCTDigi> alctV;

  std::unique_ptr<CSCUpgradeMotherboardLUTGenerator> generator_;

  /** "preferential" index array in matching window for cross-BX sorting */
  int pref[CSCConstants::MAX_LCT_TBINS];

  bool match_earliest_alct_only;
  bool match_earliest_clct_only;

  /** if true: use regular CLCT-to-ALCT matching in TMB
      if false: do ALCT-to-CLCT matching */
  bool clct_to_alct;

  /** whether to not reuse CLCTs that were used by previous matching ALCTs
      in ALCT-to-CLCT algorithm */
  bool drop_used_clcts;

  unsigned int tmb_cross_bx_algo;

  /** maximum lcts per BX in MEX1: 2, 3, 4 or 999 */
  unsigned int max_lcts;

  // debug gem matching
  bool debug_matching;

  // check look-up-tables
  bool debug_luts;
};

template <class S>
bool CSCUpgradeMotherboard::compare(const S& p, const S& q) const
{
  return (p.first == q.first) and (p.second == q.second);
}

template <class S>
void CSCUpgradeMotherboard::intersection(const S& d1, const S& d2, S& result) const
{
  for (const auto& p: d1){
    for (const auto& q: d2){
      if (compare(p,q)){
        result.push_back(p);
      }
    }
  }
 }

#endif
