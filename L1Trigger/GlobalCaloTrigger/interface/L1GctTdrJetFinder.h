#ifndef L1GCTTDRJETFINDER_H_
#define L1GCTTDRJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <vector>

/*! \class L1GctTdrJetFinder
 * \brief 3*3 sliding window algorithm jet finder.
 *
 *  Locates the jets from 48 inputted L1CaloRegions.
 *  This uses the 3*3 sliding window algorithm.
 * 
 *  The the filling of the input L1CaloRegions happens in the L1GctJetFinderBase class
 *
 *  Inputted regions are expected in a certain order with respect
 *  to the index i:
 * 
 *  Regions should arrive running from the middle (eta=0) of the detector
 *  out towards the edge of the forward HCAL, and then moving across
 *  in columns like this but increasing in phi each time.
 * 
 *  E.g. for 48 inputted regions:
 *       region  0: phi=0, other side of eta=0 line (shared data).
 *       region  1: phi=0, but correct side of eta=0 (shared data).
 *       .
 *       . 
 *       region 11: phi=0, edge of Forward HCAL (shared data).
 *       region 12: phi=20, other side of eta=0 line (shared data)
 *       region 13: phi=20, start of jet search area
 *       .
 *       .
 *       region 23: phi=20, edge of HF (jet search area)
 *       etc.
 * 
 *  In the event of neighbouring regions having the same energy, this
 *  will locate the jet in the region furthest from eta=0 that has the
 *  lowest value of phi.
 * 
 *  The jet finder now stores jets with (eta, phi) information encoded
 *  in an L1CaloRegionDetId.
 *
 *  Modified to use L1GctJetFinderBase class by Greg Heath, June 2006.
 *  
 */
/*
 * \author Jim Brooke & Robert Frazier
 * \date March 2006
 */

class L1GctTdrJetFinder : public L1GctJetFinderBase {
public:
  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctTdrJetFinder(int id);

  ~L1GctTdrJetFinder() override;

  /// Overload << operator
  friend std::ostream& operator<<(std::ostream& os, const L1GctTdrJetFinder& algo);

  /// get input data from sources
  void fetchInput() override;

  /// process the data, fill output buffers
  void process() override;

protected:
  // Each jetFinder must define the constants as private and copy the
  // function definitions below.
  unsigned maxRegionsIn() const override { return MAX_REGIONS_IN; }
  unsigned centralCol0() const override { return CENTRAL_COL0; }
  unsigned int nCols() const override { return N_COLS; }

private:
  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN;  ///< Dependent on number of rows and columns.
  static const unsigned int N_COLS;
  static const unsigned int CENTRAL_COL0;

  /// Here is the TDR 3x3 sliding window jet finder algorithm
  void findJets();

  /// Returns true if region index is the centre of a jet. Set boundary = true if at edge of HCAL.
  bool detectJet(const UShort centreIndex, const bool boundary = false) const;

  /// Returns energy sum of the 9 regions centred (physically) about centreIndex. Set boundary = true if at edge of HCAL.
  ULong calcJetEnergy(const UShort centreIndex, const bool boundary = false) const;

  /// returns the encoded (eta, phi) position of the centre region
  L1CaloRegionDetId calcJetPosition(const UShort centreIndex) const;

  /// Returns combined tauVeto of the 9 regions centred (physically) about centreIndex. Set boundary = true if at edge of Endcap.
  bool calcJetTauVeto(const UShort centreIndex, const bool boundary = false) const;
};

std::ostream& operator<<(std::ostream& os, const L1GctTdrJetFinder& algo);

#endif /*L1GCTTDRJETFINDER_H_*/
