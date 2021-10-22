//-------------------------------------------------
//
/**  \class L1MuDTExtrapolationUnit
 *
 *   Extrapolation Unit:
 *
 *   The Extrapolation Unit attempts to join
 *   track segment pairs of different stations.
 *   it contains 12 Single Extrapolation Units
 *   to perform all extrapolations in its 
 *   own wheel and 6 Single Extrapolation Units
 *   to perform all extrapolations
 *   in the adjacent wheel (next wheel neighbour)
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_EXTRAPOLATION_UNIT_H
#define L1MUDT_EXTRAPOLATION_UNIT_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <map>
#include <bitset>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
class L1MuDTSectorProcessor;
class L1MuDTSEU;
class L1MuDTTFParameters;
class L1MuDTTFParametersRcd;
class L1MuDTExtLut;
class L1MuDTExtLutRcd;
//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTExtrapolationUnit {
public:
  typedef std::pair<Extrapolation, unsigned int> SEUId;
  typedef std::map<SEUId, L1MuDTSEU*, std::less<SEUId> > SEUmap;

  /// constructor
  L1MuDTExtrapolationUnit(const L1MuDTSectorProcessor&, edm::ConsumesCollector);

  /// destructor
  ~L1MuDTExtrapolationUnit();

  /// run Extrapolation Unit
  void run(const edm::EventSetup& c);

  /// reset Extrapolation Unit
  void reset();

  /// reset a single extrapolation
  void reset(Extrapolation ext, unsigned int startAdr, unsigned int relAdr);

  /// get extrapolation address from a given ERS
  unsigned short int getAddress(Extrapolation ext, unsigned int startAdr, int id) const;

  /// get extrapolation quality from a given ERS
  unsigned short int getQuality(Extrapolation ext, unsigned int startAdr, int id) const;

  /// get Extrapolator table for a given SEU
  const std::bitset<12>& getEXTable(Extrapolation ext, unsigned int startAdr) const;

  /// get Quality Sorter table for a given SEU
  const std::bitset<12>& getQSTable(Extrapolation ext, unsigned int startAdr) const;

  /// return number of successful extrapolations
  int numberOfExt() const;

  /// print all successful extrapolations
  void print(int level = 0) const;

  /// return station of start and target track segment for a given extrapolation
  static std::pair<int, int> which_ext(Extrapolation ext);

private:
  const L1MuDTSectorProcessor& m_sp;  // reference to Sector Processor

  SEUmap m_SEUs;  // Single Extrapolation Units

  const edm::ESGetToken<L1MuDTTFParameters, L1MuDTTFParametersRcd> m_parsToken;
  const edm::ESGetToken<L1MuDTExtLut, L1MuDTExtLutRcd> m_extLUTsToken;
};

#endif
