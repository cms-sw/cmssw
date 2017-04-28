#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboard_h

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUT.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUTGenerator.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <boost/variant.hpp>

// useful forward declarations

// generic container type
template <class T>
using match = std::pair<unsigned int, T>;

template <class T>
using matches = std::vector<std::pair<unsigned int, T> >;

template <class T>
using matchesBX = std::map<int, std::vector<std::pair<unsigned int, T> > >;

typedef match<GEMPadDigi>   GEMPadDigiId;
typedef matches<GEMPadDigi> GEMPadDigiIds;
typedef matchesBX<GEMPadDigi> GEMPadDigiIdsBX;

typedef match<GEMCoPadDigi>   GEMCoPadDigiId;
typedef matches<GEMCoPadDigi> GEMCoPadDigiIds;
typedef matchesBX<GEMCoPadDigi> GEMCoPadDigiIdsBX;

typedef match<RPCDigi>   RPCDigiId;
typedef matches<RPCDigi> RPCDigiIds;
typedef matchesBX<RPCDigi> RPCDigiIdsBX;

class CSCGeometry;
class CSCChamber;

class CSCUpgradeMotherboard : public CSCMotherboard
{
public:

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  class LCTContainer {
  public:
    LCTContainer (unsigned int match_trig_window_size ) : match_trig_window_size(match_trig_window_size){}
    CSCCorrelatedLCTDigi& operator()(int bx, int match_bx, int lct) { return data[bx][match_bx][lct]; }
    std::vector<CSCCorrelatedLCTDigi> getTimeMatched(const int bx) const;
    std::vector<CSCCorrelatedLCTDigi> getMatched() const;
    CSCCorrelatedLCTDigi data[CSCMotherboard::MAX_LCT_BINS][15][2];
    const unsigned int match_trig_window_size;
  };

  CSCUpgradeMotherboard(unsigned endcap, unsigned station, unsigned sector,
                    unsigned subsector, unsigned chamber,
                    const edm::ParameterSet& conf);

   //Default constructor for testing
  CSCUpgradeMotherboard();

  virtual ~CSCUpgradeMotherboard();

  template <class S>
  bool compare(const S& p, const S& q);

  template <class S>
  S intersection(const S& d1, const S& d2);

  /** Methods to sort the LCTs */
  static bool sortLCTsByQuality(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&); 
  static bool sortLCTsByGEMDphi(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&); 
  void sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts, bool (*sorter)(const CSCCorrelatedLCTDigi&,const CSCCorrelatedLCTDigi&));

  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setupGeometry();
  void debugLUTs();

 protected:

  /** Chamber id (trigger-type labels). */
  unsigned theRegion;
  unsigned theChamber;
  bool isEven;

  edm::ParameterSet tmbParams_;
  edm::ParameterSet commonParams_;
  
  const CSCGeometry* csc_g;
  const CSCChamber* cscChamber;

  std::vector<CSCALCTDigi> alctV;

  CSCUpgradeMotherboardLUTGenerator* generator_;

  /** "preferential" index array in matching window for cross-BX sorting */
  int pref[MAX_LCT_BINS];

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
bool CSCUpgradeMotherboard::compare(const S& p, const S& q)
{ 
  return (p.first == q.first) and (p.second == q.second); 
}

template <class S>
S CSCUpgradeMotherboard::intersection(const S& d1, const S& d2)
{
  S result;
  for (auto p: d1){
    for (auto q: d2){
      if (compare(p,q)){
	result.push_back(p);
	break;
      }
    }
  }
  return result;
 }


#endif
