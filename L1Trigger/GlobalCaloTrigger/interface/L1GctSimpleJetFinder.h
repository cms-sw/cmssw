#ifndef L1GCTSIMPLEJETFINDER_H_
#define L1GCTSIMPLEJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <boost/cstdint.hpp> //for uint16_t
#include <vector>

/*! \class L1GctSimpleJetFinder
 * \brief 3*3 sliding window algorithm jet finder.
 *
 *  Locates the jets from 48 inputted L1GctRegions.
 *  This uses the 3*3 sliding window algorithm.
 * 
 *  SourceCard pointers should be set up according to:
 *  http://frazier.home.cern.ch/frazier/wiki_resources/GCT/jetfinder_sourcecard_wiring.jpg
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
 *  The jet finder outputs jets with a local eta/phi co-ordinate system 
 *  for the jets it finds in the 2*11 search area it is looking in.
 *  Eta runs from 0 to 10, with 0 being the region closest to the central
 *  eta=0 line, and 10 being the edge of which ever half of the detector
 *  the jet finder is operating in at eta= +/-5.  Phi data is set to either 
 *  0 or 1, to indicate increasing real-world phi co-ordinate.
 *  
 */
/*
 * \author Jim Brooke & Robert Frazier
 * \date March 2006
 */



class L1GctSimpleJetFinder : public L1GctJetFinderBase
{
 public:

  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctSimpleJetFinder(int id, std::vector<L1GctSourceCard*> sourceCards,
                 L1GctJetEtCalibrationLut* jetEtCalLut);
                 
  ~L1GctSimpleJetFinder();
   
  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctSimpleJetFinder& algo);

  /// get input data from sources
  virtual void fetchInput();

  /// process the data, fill output buffers
  virtual void process();

 protected:

  // Each jetFinder must define the constants as private and copy the
  // function definitions below.
  virtual unsigned maxRegionsIn() const { return MAX_REGIONS_IN; }
  virtual unsigned centralCol0() const { return CENTRAL_COL0; }
  virtual int nCols() const { return N_COLS; }

private:

  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN; ///< Dependent on number of rows and columns.
  static const int N_COLS;
  static const unsigned int CENTRAL_COL0;

  void findJets();  

};

std::ostream& operator << (std::ostream& os, const L1GctSimpleJetFinder& algo);

#endif /*L1GCTSIMPLEJETFINDER_H_*/
