#ifndef L1GCTJETFINDERBASE_H_
#define L1GCTJETFINDERBASE_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"

#include <boost/cstdint.hpp> //for uint16_t
#include <vector>

class L1GctJetFinderParams;
class L1GctJetEtCalibrationLut;
class L1CaloRegion;


/*! \class L1GctJetFinderBase
 * \brief Base class to allow implementation of jetFinder algorithms
 *
 *  The base class defines the reset() method, setXxx() and getXxx() methods.
 *  Individual jetFinders must define the fetchInput() and process() methods,
 *  using protected methods of the base class where necessary.
 *
 *  The jetFinder looks for jets over a 2x11 search area. Its 
 *  input region are pushed in from the appropriate (phi, eta) range,
 *  including across the eta=0 boundary between Wheels. The input regions
 *  are copied into a vector of dimension N_COLS*COL_OFFSET.
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
  typedef std::vector<L1GctRegion>  RegionsVector;
  typedef std::vector<L1GctJet>     RawJetVector;
  typedef std::vector<L1GctJetCand> JetVector;
  typedef Pipeline<L1GctJet>        RawJetPipeline;
  typedef L1GctUnsignedInt<L1GctEtTotal::kEtTotalNBits> etTotalType;
  typedef L1GctUnsignedInt<  L1GctEtHad::kEtHadNBits  > etHadType;

  // For HF-based triggers we sum the Etin the two "inner" (large eta) rings;
  // and count towers over threshold based on the "fineGrain" bit from the RCT.
  // Define a data type to transfer the result of all calculations.
  // The results are defined as L1GctJetCount types since they don't have
  // a separate overFlow bit. An overflow condition gives value=max.

  struct hfTowerSumsType {

    L1GctJetCount< L1GctJetCounts::kEtHfSumBits > etSum0;
    L1GctJetCount< L1GctJetCounts::kEtHfSumBits > etSum1;
    L1GctJetCount< 5 > nOverThreshold;

    // Define some constructors and an addition operator for our data type
    hfTowerSumsType() : etSum0(0), etSum1(0), nOverThreshold(0) {}
    hfTowerSumsType(unsigned e0, unsigned e1, unsigned n) : etSum0(e0), etSum1(e1), nOverThreshold(n) {}
    hfTowerSumsType(L1GctJetCount< L1GctJetCounts::kEtHfSumBits > e0,
                    L1GctJetCount< L1GctJetCounts::kEtHfSumBits > e1,
                    L1GctJetCount< 5 > n) : etSum0(e0), etSum1(e1), nOverThreshold(n) {}

    void reset() { etSum0.reset(); etSum1.reset(); nOverThreshold.reset(); }

    hfTowerSumsType operator+(const hfTowerSumsType& rhs) const {
      hfTowerSumsType temp( (this->etSum0+rhs.etSum0),
                            (this->etSum1+rhs.etSum1),
                            (this->nOverThreshold+rhs.nOverThreshold) );
      return temp;
    } 

  };

  //Statics
  static const unsigned int MAX_JETS_OUT;  ///< Max of 6 jets found per jetfinder in a 2*11 search area.
  static const unsigned int COL_OFFSET;  ///< The index offset between columns
  static const unsigned int N_JF_PER_WHEEL; ///< No of jetFinders per Wheel
    
  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctJetFinderBase(int id);
                 
  ~L1GctJetFinderBase();
   
  /// Set pointers to neighbours - needed to complete the setup
  void setNeighbourJetFinders(std::vector<L1GctJetFinderBase*> neighbours);

  /// Set pointer to parameters - needed to complete the setup
  void setJetFinderParams(const L1GctJetFinderParams* jfpars);

  /// Set pointer to calibration Lut - needed to complete the setup
  void setJetEtCalibrationLut(const L1GctJetEtCalibrationLut* lut);

  /// Check setup is Ok
  bool setupOk() const { return m_gotNeighbourPointers
                             && m_gotJetFinderParams
                             && (m_jetEtCalLut != 0); }

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetFinderBase& algo);

  /// get input data from sources; to be filled in by derived jetFinders
  virtual void fetchInput() = 0;

  /// process the data, fill output buffers; to be filled in by derived jetFinders
  virtual void process() = 0;

  /// Set input data
  void setInputRegion(const L1CaloRegion& region);
    
  /// Return input data   
  RegionsVector getInputRegions() const { return m_inputRegions; }

  /// get protoJets sent to neighbour
  RegionsVector getSentProtoJets() const { return m_sentProtoJets; }

  /// get protoJets received from neighbour
  RegionsVector getRcvdProtoJets() const { return m_rcvdProtoJets; }

  /// get protoJets kept
  RegionsVector getKeptProtoJets() const { return m_keptProtoJets; }

  /// get output jets in raw format
  RawJetVector getRawJets() const { return m_outputJetsPipe.contents; } 

  /// Return pointer to calibration LUT
  const L1GctJetEtCalibrationLut* getJetEtCalLut() const { return m_jetEtCalLut; }

  // The hardware output quantities
  JetVector getJets() const { return m_sortedJets; } ///< Get the located jets. 
  etTotalType getEtStrip0() const { return m_outputEtStrip0; }  ///< Get transverse energy strip sum 0
  etTotalType getEtStrip1() const { return m_outputEtStrip1; }  ///< Get transverse energy strip sum 1
  etHadType   getHt() const { return m_outputHt; }              ///< Get the total calibrated energy in jets (Ht) found by this jet finder

  hfTowerSumsType getHfSums() const { return m_outputHfSums; }  ///< Get the Hf tower Et sums and tower-over-threshold counts

  // comparison operator for sorting jets in the Wheel Fpga, JetFinder, and JetFinalStage
  struct rankGreaterThan : public std::binary_function<L1GctJetCand, L1GctJetCand, bool> 
  {
    bool operator()(const L1GctJetCand& x, const L1GctJetCand& y) {
      return ( x.rank() > y.rank() ) ;
    }
  };

 protected:

  /// Separate reset methods for the processor itself and any data stored in pipelines
  virtual void resetProcessor();
  virtual void resetPipelines();

  /// Initialise inputs with null objects for the correct bunch crossing if required
  virtual void setupObjects();

 protected:

  /// different ways of getting the neighbour data
  enum fetchType { TOP, BOT, TOPBOT };

  /// algo ID
  int m_id;
	
  /// Store neighbour pointers
  std::vector<L1GctJetFinderBase*> m_neighbourJetFinders;
  
  /// Remember whether the neighbour pointers have been stored
  bool m_gotNeighbourPointers;

  /// Remember whether jetfinder parameters have been stored
  bool m_gotJetFinderParams;

  /// jetFinder parameters (from EventSetup)
  unsigned m_CenJetSeed;
  unsigned m_FwdJetSeed;
  unsigned m_TauJetSeed;
  unsigned m_EtaBoundry;

  /// Jet Et Converstion LUT pointer
  const L1GctJetEtCalibrationLut* m_jetEtCalLut;
    
  /// input data required for jet finding
  RegionsVector m_inputRegions;

  /// List of pre-clustered jets to be sent to neighbour after the first stage of clustering
  RegionsVector m_sentProtoJets;
  /// List of pre-clustered jets received from neighbour before the final stage of clustering
  RegionsVector m_rcvdProtoJets;
  /// List of pre-clustered jets retained locally as input to the final clustering
  RegionsVector m_keptProtoJets;

  /// output jets
  RawJetVector m_outputJets;
  JetVector m_sortedJets;

  /// output Et strip sums and Ht
  etTotalType m_outputEtStrip0;
  etTotalType m_outputEtStrip1;
  etHadType   m_outputHt;

  hfTowerSumsType m_outputHfSums;
    
  //PROTECTED METHODS
  // Return the values of constants that might be changed by different jetFinders.
  // Each jetFinder must define the constants as private and copy the
  // function definitions below.
  virtual unsigned maxRegionsIn() const { return MAX_REGIONS_IN; }
  virtual unsigned centralCol0() const { return CENTRAL_COL0; }
  virtual unsigned nCols() const { return N_COLS; }

  /// Helper functions for the fetchInput() and process() methods
  /// fetch the protoJets from neighbour jetFinder
  void fetchProtoJetsFromNeighbour(const fetchType ft);
  /// Sort the found jets. All jetFinders should call this in process().
  void sortJets();
  /// Fill the Et strip sums and Ht sum. All jetFinders should call this in process().
  void doEnergySums();
    
  /// Calculates total (raw) energy in a phi strip
  etTotalType calcEtStrip(const UShort strip) const;

  /// Calculates total calibrated energy in jets (Ht) sum
  etHadType   calcHt() const;
  
  /// Calculates Et sum and number of towers over threshold in Hf
  hfTowerSumsType calcHfSums() const;

  /// parameter to determine which Regions belong in our acceptance
  unsigned m_minColThisJf;

 private:

  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN; ///< Dependent on number of rows and columns.
  static const unsigned int N_COLS;
  static const unsigned int CENTRAL_COL0;

  /// Output jets "pipeline memory" for checking
  RawJetPipeline m_outputJetsPipe;
};

std::ostream& operator << (std::ostream& os, const L1GctJetFinderBase& algo);

#endif /*L1GCTJETFINDERBASE_H_*/
