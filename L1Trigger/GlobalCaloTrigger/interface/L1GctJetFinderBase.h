#ifndef L1GCTJETFINDERBASE_H_
#define L1GCTJETFINDERBASE_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEtSum.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHtMiss.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"

#include <boost/cstdint.hpp> //for uint16_t
#include <vector>

class L1GctInternJetData;
class L1GctJetFinderParams;
class L1GctJetEtCalibrationLut;
class L1GctChannelMask;
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
  typedef L1GctUnsignedInt<L1GctInternEtSum::kTotEtOrHtNBits> etTotalType;
  typedef L1GctUnsignedInt<L1GctInternEtSum::kTotEtOrHtNBits> etHadType;
  typedef L1GctTwosComplement<  L1GctInternEtSum::kJetMissEtNBits > etCompInternJfType;
  typedef L1GctTwosComplement< L1GctInternHtMiss::kJetMissHtNBits > htCompInternJfType;

  enum maxValues {
    etTotalMaxValue = L1GctInternEtSum::kTotEtOrHtMaxValue,
    htTotalMaxValue = L1GctInternEtSum::kTotEtOrHtMaxValue
  };


  // For HF-based triggers we sum the Et in the two "inner" (large eta) rings;
  // and count towers over threshold based on the "fineGrain" bit from the RCT.
  // Define a data type to transfer the result of all calculations.
  // The results are defined as L1GctJetCount types since they don't have
  // a separate overFlow bit. An overflow condition gives value=max.

  struct hfTowerSumsType {

    enum numberOfBits {
      kHfEtSumBits = 8,
      kHfCountBits = 5
    };

    L1GctJetCount< kHfEtSumBits > etSum0;
    L1GctJetCount< kHfEtSumBits > etSum1;
    L1GctJetCount< kHfCountBits > nOverThreshold0;
    L1GctJetCount< kHfCountBits > nOverThreshold1;

    // Define some constructors and an addition operator for our data type
    hfTowerSumsType() : etSum0(0), etSum1(0), nOverThreshold0(0), nOverThreshold1(0) {}
    hfTowerSumsType(unsigned e0, unsigned e1, unsigned n0, unsigned n1) : 
      etSum0(e0), etSum1(e1), nOverThreshold0(n0), nOverThreshold1(n1) {}
    hfTowerSumsType(L1GctJetCount< kHfEtSumBits > e0,
                    L1GctJetCount< kHfEtSumBits > e1,
                    L1GctJetCount< kHfCountBits > n0,
                    L1GctJetCount< kHfCountBits > n1) : etSum0(e0), etSum1(e1), nOverThreshold0(n0), nOverThreshold1(n1) {}

    void reset() { etSum0.reset(); etSum1.reset(); nOverThreshold0.reset(); nOverThreshold1.reset(); }

    hfTowerSumsType operator+(const hfTowerSumsType& rhs) const {
      hfTowerSumsType temp( (this->etSum0+rhs.etSum0),
                            (this->etSum1+rhs.etSum1),
                            (this->nOverThreshold0+rhs.nOverThreshold0),
                            (this->nOverThreshold1+rhs.nOverThreshold1) );
      return temp;
    } 

  };

  typedef L1GctJet::lutPtr lutPtr;
  typedef std::vector<lutPtr> lutPtrVector;

  //Statics
  static const unsigned int MAX_JETS_OUT;  ///< Max of 6 jets found per jetfinder in a 2*11 search area.
  static const unsigned int COL_OFFSET;  ///< The index offset between columns
  static const unsigned int N_JF_PER_WHEEL; ///< No of jetFinders per Wheel
  static const unsigned int N_EXTRA_REGIONS_ETA00; ///< Number of additional regions to process on the "wrong" side of eta=0 (determines COL_OFFSET) 
    
  /// id is 0-8 for -ve Eta jetfinders, 9-17 for +ve Eta, for increasing Phi.
  L1GctJetFinderBase(int id);
                 
  ~L1GctJetFinderBase();
   
  /// Set pointers to neighbours - needed to complete the setup
  void setNeighbourJetFinders(const std::vector<L1GctJetFinderBase*>& neighbours);

  /// Set pointer to parameters - needed to complete the setup
  void setJetFinderParams(const L1GctJetFinderParams* jfpars);

  /// Set pointer to calibration Lut - needed to complete the setup
  void setJetEtCalibrationLuts(const lutPtrVector& jfluts);

  /// Set masks for energy summing
  void setEnergySumMasks(const L1GctChannelMask* chmask);

  /// Setup the tau algorithm parameters
  void setupTauAlgo(const bool useImprovedAlgo, const bool ignoreVetoBitsForIsolation) {
    m_useImprovedTauAlgo            = useImprovedAlgo;
    m_ignoreTauVetoBitsForIsolation = ignoreVetoBitsForIsolation;
  }

  /// Check setup is Ok
  bool setupOk() const { return m_idInRange
			   && m_gotNeighbourPointers
			   && m_gotJetFinderParams
			   && m_gotJetEtCalLuts
                           && m_gotChannelMask; }

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

  /// get output jets in raw format - to be stored in the event
  std::vector< L1GctInternJetData > getInternalJets() const;

  /// get et sums in raw format - to be stored in the event
  std::vector< L1GctInternEtSum  > getInternalEtSums() const;
  std::vector< L1GctInternHtMiss > getInternalHtMiss() const;

  /// Return pointers to calibration LUTs
  const lutPtrVector getJetEtCalLuts() const { return m_jetEtCalLuts; }

  // The hardware output quantities
  JetVector getJets() const { return m_sortedJets; } ///< Get the located jets.
  // The hardware output quantities - refactored
  etTotalType        getEtSum() const { return m_outputEtSum; }  ///< Get the scalar sum of Et summed over the input regions
  etCompInternJfType getExSum() const { return m_outputExSum; }  ///< Get the x component of vector Et summed over the input regions
  etCompInternJfType getEySum() const { return m_outputEySum; }  ///< Get the y component of vector Et summed over the input regions
  etHadType          getHtSum() const { return m_outputHtSum; }  ///< Get the scalar sum of Ht summed over jets above threshold
  htCompInternJfType getHxSum() const { return m_outputHxSum; }  ///< Get the x component of vector Ht summed over jets above threshold
  htCompInternJfType getHySum() const { return m_outputHySum; }  ///< Get the y component of vector Ht summed over jets above threshold

  hfTowerSumsType getHfSums() const { return m_outputHfSums; }  ///< Get the Hf tower Et sums and tower-over-threshold counts

  // Access to threshold and cut values
  unsigned getCenJetSeed() const { return m_CenJetSeed; }
  unsigned getFwdJetSeed() const { return m_FwdJetSeed; }
  unsigned getTauJetSeed() const { return m_TauJetSeed; }
  unsigned getEtaBoundry() const { return m_EtaBoundry; }
  unsigned getTauIsolationThreshold() const { return m_tauIsolationThreshold; }
  unsigned getHttSumJetThreshold() const { return m_HttSumJetThreshold; }
  unsigned getHtmSumJetThreshold() const { return m_HtmSumJetThreshold; }

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
  
  /// Remember whether range check on the input ID was ok
  bool m_idInRange;
  
  /// Remember whether the neighbour pointers have been stored
  bool m_gotNeighbourPointers;

  /// Remember whether jetfinder parameters have been stored
  bool m_gotJetFinderParams;

  /// Remember whether jet Et calibration Lut pointers have been stored
  bool m_gotJetEtCalLuts;

  /// Remember whether channel mask have been stored
  bool m_gotChannelMask;

  ///
  /// *** Geometry parameters ***
  ///
  /// Positive/negative eta flag
  bool m_positiveEtaWheel;

  /// parameter to determine which Regions belong in our acceptance
  unsigned m_minColThisJf;
  ///
  ///---------------------------------------------------------------------------------------
  ///
  /// *** Setup parameters for this jetfinder instance ***
  ///
  /// jetFinder parameters (from EventSetup)
  unsigned m_CenJetSeed;
  unsigned m_FwdJetSeed;
  unsigned m_TauJetSeed;
  unsigned m_EtaBoundry;

  /// Jet Et Conversion LUT pointer
  lutPtrVector m_jetEtCalLuts;

  /// Setup parameters for the tau jet algorithm
  // If the following parameter is set to false, the tau identification
  // is just based on the RCT tau veto bits from the nine regions
  bool m_useImprovedTauAlgo;

  // If useImprovedTauAlgo is true, these two parameters affect
  // the operation of the algorithm.

  // We can require the tau veto bits to be off in all nine regions,
  // or just in the central region.
  bool m_ignoreTauVetoBitsForIsolation;
    
  // In the improved tau algorithm, we require no more than one tower energy to be 
  // above the isolation threshold, in the eight regions surrounding the central one. 
  unsigned m_tauIsolationThreshold;

  // Thresholds on individual jet energies used in HTT and HTM summing
  unsigned m_HttSumJetThreshold;
  unsigned m_HtmSumJetThreshold;

  // Masks for restricting the eta range of energy sums
  bool m_EttMask[11];
  bool m_EtmMask[11];
  bool m_HttMask[11];
  bool m_HtmMask[11];

  ///
  /// *** End of setup parameters ***
  ///
  ///---------------------------------------------------------------------------------------
  ///
  /// *** Start of event data ***
  ///
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

  /// output Et strip sums and Ht - refactored
  etTotalType        m_outputEtSum;
  etCompInternJfType m_outputExSum;
  etCompInternJfType m_outputEySum;
  etHadType          m_outputHtSum;
  htCompInternJfType m_outputHxSum;
  htCompInternJfType m_outputHySum;

  hfTowerSumsType m_outputHfSums;
    
  ///
  /// *** End of event data ***
  ///
  ///---------------------------------------------------------------------------------------
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
  etTotalType calcHtStrip(const UShort strip) const;
  
  /// Calculates scalar and vector sum of Et over input regions
  void doEtSums() ;
  
  /// Calculates scalar and vector sum of Ht over calibrated jets
  void doHtSums() ;
  
  /// Calculates Et sum and number of towers over threshold in Hf
  hfTowerSumsType calcHfSums() const;

 private:

  /// The real jetFinders must define these constants
  static const unsigned int MAX_REGIONS_IN; ///< Dependent on number of rows and columns.
  static const unsigned int N_COLS;
  static const unsigned int CENTRAL_COL0;

  /// Output jets "pipeline memory" for checking
  RawJetPipeline m_outputJetsPipe;

  /// "Pipeline memories" for energy sums
  Pipeline< etTotalType        > m_outputEtSumPipe;
  Pipeline< etCompInternJfType > m_outputExSumPipe;
  Pipeline< etCompInternJfType > m_outputEySumPipe;
  Pipeline< etHadType          > m_outputHtSumPipe;
  Pipeline< htCompInternJfType > m_outputHxSumPipe;
  Pipeline< htCompInternJfType > m_outputHySumPipe;

  /// Private method for calculating MEt and MHt components
  template <int kBitsInput, int kBitsOutput>
    L1GctTwosComplement<kBitsOutput>
    etComponentForJetFinder(const L1GctUnsignedInt<kBitsInput>& etStrip0, const unsigned& fact0,
			    const L1GctUnsignedInt<kBitsInput>& etStrip1, const unsigned& fact1);


};

std::ostream& operator << (std::ostream& os, const L1GctJetFinderBase& algo);

#endif /*L1GCTJETFINDERBASE_H_*/
