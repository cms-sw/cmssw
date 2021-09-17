#ifndef L1GCTSIMPLEJETFINDER_H_
#define L1GCTSIMPLEJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <vector>

/*! \class L1GctSimpleJetFinder
 * \brief simple jet finder for test purposes.
 *
 * Currently returns no jets
 * The intention is to return local maxima (without clustering)
 *  
 */
/*
 * \author Greg Heath
 * \date June 2006
 */

class L1GctSimpleJetFinder : public L1GctJetFinderBase {
public:
  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctSimpleJetFinder(int id);

  ~L1GctSimpleJetFinder();

  /// Overload << operator
  friend std::ostream& operator<<(std::ostream& os, const L1GctSimpleJetFinder& algo);

  /// get input data from sources
  virtual void fetchInput();

  /// process the data, fill output buffers
  virtual void process();

protected:
  // Each jetFinder must define the constants as private and copy the
  // function definitions below.
  virtual unsigned maxRegionsIn() const { return MAX_REGIONS_IN; }
  virtual unsigned centralCol0() const { return CENTRAL_COL0; }
  virtual unsigned nCols() const { return N_COLS; }

private:
  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN;  ///< Dependent on number of rows and columns.
  static const unsigned int N_COLS;
  static const unsigned int CENTRAL_COL0;

  void findJets();
};

std::ostream& operator<<(std::ostream& os, const L1GctSimpleJetFinder& algo);

#endif /*L1GCTSIMPLEJETFINDER_H_*/
