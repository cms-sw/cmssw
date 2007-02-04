#ifndef L1GCTJETFINDERBASE_H_
#define L1GCTJETFINDERBASE_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include <boost/cstdint.hpp> //for uint16_t
#include <vector>

/*! \class L1GctJetFinderBase
 * \brief Base class to allow implementation of jetFinder algorithms
 *
 *  The base class defines the reset() method, setXxx() and getXxx() methods.
 *  Individual jetFinders must define the fetchInput() and process() methods,
 *  using protected methods of the base class where necessary.
 *
 *  The jetFinder looks for jets over a 2x11 search area. It can pull in
 *  additional input regions from source cards to the left and right,
 *  and from across the eta=0 boundary between Wheels. The input regions
 *  are copied into a vector of dimension N_COLS*COL_OFFSET.
 *
 *  SourceCard pointers should be set up according to:
 *  http://frazier.home.cern.ch/frazier/wiki_resources/GCT/jetfinder_sourcecard_wiring.jpg
 * 
 *  The array of input regions is filled in a certain order with respect
 *  to the index i: 
 * 
 *  The jetFinder can also pull in "proto-jets" from adjacent jetFinders.
 *  If required by the algorithm, these must be calculated in the
 *  fetchInput() method;
 * 
 */
/*
 * \author Jim Brooke & Greg Heath
 * \date June 2006
 */



class L1GctJetFinderBase : public L1GctProcessor
{
public:
  //Typedefs
  typedef unsigned long int ULong;
  typedef unsigned short int UShort;
  typedef std::vector<L1CaloRegion> RegionsVector;
  typedef std::vector<L1GctJet> JetVector;

  //Statics
  static const unsigned int MAX_JETS_OUT;  ///< Max of 6 jets found per jetfinder in a 2*11 search area
  static const unsigned int MAX_SOURCE_CARDS; ///< Need data from 9 separate source cards to find jets in the 2*11 search region.
  static const unsigned int COL_OFFSET;  ///< The index offset between columns
  static const unsigned int N_JF_PER_WHEEL; ///< No of jetFinders per Wheel
    
  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctJetFinderBase(int id, std::vector<L1GctSourceCard*> sourceCards,
		     L1GctJetEtCalibrationLut* jetEtCalLut);
                 
  ~L1GctJetFinderBase();
   
  /// Set pointers to neighbours - needed to complete the setup
  void setNeighbourJetFinders(std::vector<L1GctJetFinderBase*> neighbours);

  /// Check setup is Ok
  bool gotNeighbourPointers() const { return m_gotNeighbourPointers; }

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetFinderBase& algo);

  /// clear internal buffers
  virtual void reset();

  /// get input data from sources; to be filled in by derived jetFinders
  virtual void fetchInput() = 0;

  /// process the data, fill output buffers; to be filled in by derived jetFinders
  virtual void process() = 0;

  /// Set input data
  void setInputRegion(unsigned i, L1CaloRegion region);
    
  /// Return input data   
  RegionsVector getInputRegions() const { return m_inputRegions; }

  /// get protoJets sent to neighbour
  RegionsVector getSentProtoJets() const { return m_sentProtoJets; }

  /// get protoJets received from neighbour
  RegionsVector getRcvdProtoJets() const { return m_rcvdProtoJets; }

  /// get protoJets kept
  RegionsVector getKeptProtoJets() const { return m_keptProtoJets; }

  /// Return pointer to calibration LUT
  L1GctJetEtCalibrationLut* getJetEtCalLut() const { return m_jetEtCalLut; }

  JetVector getJets() const { return m_outputJets; } ///< Get the located jets. 
  L1GctUnsignedInt<12> getEtStrip0() const { return m_outputEtStrip0; }  ///< Get transverse energy strip sum 0
  L1GctUnsignedInt<12> getEtStrip1() const { return m_outputEtStrip1; }  ///< Get transverse energy strip sum 1
  L1GctUnsignedInt<12> getHt() const { return m_outputHt; }              ///< Get the total calibrated energy in jets (Ht) found by this jet finder

 protected:

  /// different ways of getting the neighbour data
  enum fetchType { TOP, BOT, TOPBOT };

  /// algo ID
  int m_id;
	
  /// Store source card pointers
  std::vector<L1GctSourceCard*> m_sourceCards;
  
  /// Store neighbour pointers
  std::vector<L1GctJetFinderBase*> m_neighbourJetFinders;
  
  /// Remember whether the neighbour pointers have been stored
  bool m_gotNeighbourPointers;

  /// Jet Et Converstion LUT pointer
  L1GctJetEtCalibrationLut* m_jetEtCalLut;
    
  /// input data required for jet finding
  RegionsVector m_inputRegions;

  /// List of pre-clustered jets to be sent to neighbour after the first stage of clustering
  RegionsVector m_sentProtoJets;
  /// List of pre-clustered jets received from neighbour before the final stage of clustering
  RegionsVector m_rcvdProtoJets;
  /// List of pre-clustered jets retained locally as input to the final clustering
  RegionsVector m_keptProtoJets;

  /// output jets
  JetVector m_outputJets;

  /// output Et strip sums and Ht
  L1GctUnsignedInt<12> m_outputEtStrip0;
  L1GctUnsignedInt<12> m_outputEtStrip1;
  L1GctUnsignedInt<12> m_outputHt;
    
  //PROTECTED METHODS
  // Return the values of constants that might be changed by different jetFinders.
  // Each jetFinder must define the constants as private and copy the
  // function definitions below.
  virtual unsigned maxRegionsIn() const { return MAX_REGIONS_IN; }
  virtual unsigned centralCol0() const { return CENTRAL_COL0; }
  virtual int nCols() const { return N_COLS; }

  /// Helper functions for the fetchInput() and process() methods
  /// Get the input regions for the 2x11 search window plus eta=0 neighbours
  void fetchCentreStripsInput();
  /// Get the input regions for adjacent 2x11 search windows plus eta=0 neighbours
  void fetchEdgeStripsInput();
  /// fetch the protoJets from neighbour jetFinder
  void fetchProtoJetsFromNeighbour(const fetchType ft);
  /// Sort the found jets. All jetFinders should call this in process().
  void sortJets();
  /// Fill the Et strip sums and Ht sum. All jetFinders should call this in process().
  void doEnergySums();
    
  /// Copy the input regions from one source card into the m_inputRegions vector
  void fetchScInput(L1GctSourceCard* sourceCard, int col0);
  /// Copy the input regions from one eta=0 neighbour source card
  void fetchNeighbourScInput(L1GctSourceCard* sourceCard, int col0);

  /// Calculates total (raw) energy in a phi strip
  L1GctUnsignedInt<12> calcEtStrip(const UShort strip) const;

  /// Calculates total calibrated energy in jets (Ht) sum
  L1GctUnsignedInt<12> calcHt() const;
  
 private:

  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN; ///< Dependent on number of rows and columns.
  static const int N_COLS;
  static const unsigned int CENTRAL_COL0;

};

std::ostream& operator << (std::ostream& os, const L1GctJetFinderBase& algo);

#endif /*L1GCTJETFINDERBASE_H_*/
