#ifndef L1GCTHARDWAREJETFINDER_H_
#define L1GCTHARDWAREJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <vector>

/*! \class L1GctHardwareJetFinder
 * \brief Emulation of the hardware jet finder.
 *
 *  
 */
/*
 * \author Greg Heath
 * \date June 2006
 */

class L1GctHardwareJetFinder : public L1GctJetFinderBase {
public:
  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctHardwareJetFinder(int id);

  ~L1GctHardwareJetFinder() override;

  /// Overload << operator
  friend std::ostream& operator<<(std::ostream& os, const L1GctHardwareJetFinder& algo);

  /// include additional reset functionality
  virtual void reset();

  /// get input data from sources
  void fetchInput() override;

  /// process the data, fill output buffers
  void process() override;

protected:
  // Each jetFinder must define the constants as private and copy the
  // function definitions below.
  unsigned maxRegionsIn() const override { return MAX_REGIONS_IN; }
  unsigned centralCol0() const override { return CENTRAL_COL0; }
  unsigned nCols() const override { return N_COLS; }

private:
  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN;  ///< Dependent on number of rows and columns.
  static const unsigned int N_COLS;
  static const unsigned int CENTRAL_COL0;

  /// Local vectors used during both stages of clustering
  RegionsVector m_localMaxima;
  /// Each local maximum becomes a cluster
  RegionsVector m_clusters;

  /// The number of local Maxima/clusters found at each stage of clustering
  unsigned m_numberOfClusters;

  // Additional clusters to avoid double counting of jets across eta=0
  RegionsVector m_localMax00;
  RegionsVector m_cluster00;

  /// The first stage of clustering, called by fetchInput()
  void findProtoJets();
  L1GctRegion makeProtoJet(L1GctRegion localMax);
  /// The second stage of clustering, called by process()
  void findJets();

  /// Find local maxima in the search array
  void findLocalMaxima();
  /// Convert local maxima to clusters
  void findProtoClusters();
  /// Convert protojets to final jets
  void findFinalClusters();

  /// Organise the pre-clustered jets into the ones we keep and those we send to the neighbour
  void convertClustersToProtoJets();
  /// Organise the final clustered jets into L1GctJets
  void convertClustersToOutputJets();
};

std::ostream& operator<<(std::ostream& os, const L1GctHardwareJetFinder& algo);

#endif /*L1GCTHARDWAREJETFINDER_H_*/
