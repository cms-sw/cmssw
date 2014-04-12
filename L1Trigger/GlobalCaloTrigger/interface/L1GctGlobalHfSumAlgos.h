#ifndef L1GCTGLOBALHFSUMALGOS_H_
#define L1GCTGLOBALHFSUMALGOS_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfEtSumsLut.h"

#include <vector>
#include <map>

class L1GctWheelJetFpga;
class L1GctHfLutSetup;
class L1GctHfBitCountsLut;

/*!
 * \class L1GctGlobalHfSumAlgos
 * \brief Emulates the GCT summing and packing of Hf Et sums and tower-over-threshold counts
 *
 * Gets the sums from the wheel cards and packs them
 * into the output after compression using look-up tables
 *
 * \author Greg Heath
 * \date 09/09/2008
 * 
 */

class L1GctGlobalHfSumAlgos : public L1GctProcessor
{
 public:

  typedef L1GctJetFinderBase::hfTowerSumsType hfTowerSumsType;
  
  /// Constructor needs the Wheel card Fpgas set up first
  L1GctGlobalHfSumAlgos(const std::vector<L1GctWheelJetFpga*>& WheelJetFpga);
  /// Destructor
  ~L1GctGlobalHfSumAlgos();

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctGlobalHfSumAlgos& fpga);

  /// get input data from sources; this is the standard way to provide input
  virtual void fetchInput();

  /// process the data, fill output buffers
  virtual void process();

  /// Access to output quantities
  std::vector<uint16_t> hfSumsOutput(const L1GctHfEtSumsLut::hfLutType type) const;
  std::vector<unsigned> hfSumsWord() const;

  /// Setup luts
  void setupLuts(const L1CaloEtScale* scale);

  /// Get lut pointers
  const L1GctHfBitCountsLut* getBCLut(const L1GctHfEtSumsLut::hfLutType type) const;
  const L1GctHfEtSumsLut* getESLut(const L1GctHfEtSumsLut::hfLutType type) const;

  /// Get thresholds
  std::vector<double> getThresholds(const L1GctHfEtSumsLut::hfLutType type) const;

  /// provide access to input pointer, Wheel Jet Fpga 1
  L1GctWheelJetFpga* getPlusWheelJetFpga() const { return m_plusWheelJetFpga; }
  /// provide access to input pointer, Wheel Jet Fpga 0
  L1GctWheelJetFpga* getMinusWheelJetFpga() const { return m_minusWheelJetFpga; }

  /// check setup
  bool setupOk() const { return m_setupOk; }
  
 protected:
  /// Separate reset methods for the processor itself and any data stored in pipelines
  virtual void resetProcessor();
  virtual void resetPipelines();

  /// Initialise inputs with null objects for the correct bunch crossing if required
  virtual void setupObjects() {}
	
 private:
  // Here are the algorithm types we get our inputs from
  L1GctWheelJetFpga* m_plusWheelJetFpga;
  L1GctWheelJetFpga* m_minusWheelJetFpga;

  // Here are the lookup tables
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfBitCountsLut*> m_bitCountLuts;
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfEtSumsLut*> m_etSumLuts;

  // Input data for one bunch crossing
  hfTowerSumsType m_hfInputSumsPlusWheel;
  hfTowerSumsType m_hfInputSumsMinusWheel;

  // Output data
  std::map<L1GctHfEtSumsLut::hfLutType, Pipeline<uint16_t> > m_hfOutputSumsPipe;

  bool m_setupOk;

  // private methods
  // Convert bit count value using LUT and store in the pipeline
  void storeBitCount(L1GctHfEtSumsLut::hfLutType type, uint16_t value);
  // Convert et sum value using LUT and store in the pipeline
  void storeEtSum(L1GctHfEtSumsLut::hfLutType type, uint16_t value);

};

std::ostream& operator << (std::ostream& os, const L1GctGlobalHfSumAlgos& fpga);

#endif /*L1GCTGLOBALHFSUMALGOS_H_*/
