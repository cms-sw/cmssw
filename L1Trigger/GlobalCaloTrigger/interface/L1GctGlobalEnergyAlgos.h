#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctMet.h"


#include <vector>

class L1GctWheelEnergyFpga;
class L1GctWheelJetFpga;
class L1GctGlobalHfSumAlgos;

/*!
 * \class L1GctGlobalEnergyAlgos
 * \brief Emulates the GCT global energy algorithms
 *
 * This class carries out the final stage of summing of
 * total Et, missing Et components Ex and Ey, calibrated
 * jet energy Ht, and jet counts. It converts the final
 * missing Ex and Ey sums to magnitude and direction.
 * The inputs come from the two Wheel cards
 * and the outputs are sent to the Global Trigger.
 *
 * \author Jim Brooke & Greg Heath
 * \date 20/2/2006
 * 
 */

class L1GctGlobalEnergyAlgos : public L1GctProcessor
{
public:
  
        typedef L1GctUnsignedInt< L1GctEtTotal::kEtTotalNBits   > etTotalType;
        typedef L1GctUnsignedInt<   L1GctEtHad::kEtHadNBits     > etHadType;
        typedef L1GctMet::etMissType     etMissType;
        typedef L1GctMet::etMissPhiType  etMissPhiType;
        typedef L1GctMet::etmiss_vec     etmiss_vec;
        typedef L1GctWheelEnergyFpga::etComponentType etComponentType;

	enum maxValues {
	  etTotalMaxValue = L1GctEtTotal::kEtTotalMaxValue,
	  etHadMaxValue   =   L1GctEtHad::kEtHadMaxValue,
	  etMissMaxValue  =  L1GctEtMiss::kEtMissMaxValue
	};

        /// Constructor needs the Wheel card Fpgas set up first
	 L1GctGlobalEnergyAlgos(const std::vector<L1GctWheelEnergyFpga*>& WheelFpga,
			       const std::vector<L1GctWheelJetFpga*>& WheelJetFpga);
	 /// Destructor
	~L1GctGlobalEnergyAlgos();

        /// Overload << operator
        friend std::ostream& operator << (std::ostream& os, const L1GctGlobalEnergyAlgos& fpga);

	/// clear internal buffers
	void reset();

	/// get input data from sources
	virtual void fetchInput();

	/// process the data, fill output buffers
	virtual void process();

	/// define the bunch crossing range to process
	void setBxRange(const int firstBx, const int numberOfBx);

	/// partially clear buffers
	void setNextBx(const int bx);

	/// set input Ex value per wheel (0 or 1); not used in normal operation
	void setInputWheelEx(unsigned wheel, int energy, bool overflow);
	/// set input Ey value per wheel (0 or 1); not used in normal operation
	void setInputWheelEy(unsigned wheel, int energy, bool overflow);
	/// set input Et value per wheel (0 or 1); not used in normal operation
	void setInputWheelEt(unsigned wheel, unsigned energy, bool overflow);
	/// set input Ht value per wheel (0 or 1); not used in normal operation
	void setInputWheelHt(unsigned wheel, unsigned energy, bool overflow);
	/// set input Ht component values per wheel (0 or 1); not used in normal operation
	void setInputWheelHx(unsigned wheel, unsigned energy, bool overflow);
	void setInputWheelHy(unsigned wheel, unsigned energy, bool overflow);

	/// set input jet count (number 0-11) per wheel (0 or 1); not used in normal operation
        void setInputWheelJc(unsigned wheel, unsigned jcnum, unsigned count);

	/// provide access to input pointer, Wheel Energy Fpga 1
	L1GctWheelEnergyFpga* getPlusWheelFpga() const { return m_plusWheelFpga; }
	/// provide access to input pointer, Wheel Energy Fpga 0
	L1GctWheelEnergyFpga* getMinusWheelFpga() const { return m_minusWheelFpga; }
	/// provide access to input pointer, Wheel Jet Fpga 1
	L1GctWheelJetFpga* getPlusWheelJetFpga() const { return m_plusWheelJetFpga; }
	/// provide access to input pointer, Wheel Jet Fpga 0
	L1GctWheelJetFpga* getMinusWheelJetFpga() const { return m_minusWheelJetFpga; }
	/// provide access to hf sum processor
	L1GctGlobalHfSumAlgos* getHfSumProcessor() const { return m_hfSumProcessor; }

	/// return input Ex value wheel 1
       inline std::vector< etComponentType > getInputExValPlusWheel() const { return m_exValPlusPipe.contents; }
	/// return input Ex value wheel 1
       inline std::vector< etComponentType > getInputEyValPlusWheel() const { return m_eyValPlusPipe.contents; }
	/// return input Ey value wheel 0
       inline std::vector< etComponentType > getInputExVlMinusWheel() const { return m_exVlMinusPipe.contents; }
	/// return input Ey value wheel 0
       inline std::vector< etComponentType > getInputEyVlMinusWheel() const { return m_eyVlMinusPipe.contents; }
	/// return input Et value wheel 1
	inline std::vector< etTotalType > getInputEtValPlusWheel() const { return m_etValPlusPipe.contents; }
	/// return input Ht value wheel 1
	inline std::vector< etHadType   > getInputHtValPlusWheel() const { return m_htValPlusPipe.contents; }
	/// return input Ht component values wheel 1
	inline std::vector< etComponentType > getInputHxValPlusWheel() const { return m_hxValPlusPipe.contents; }
	inline std::vector< etComponentType > getInputHyValPlusWheel() const { return m_hyValPlusPipe.contents; }
	/// return input Et value wheel 0
	inline std::vector< etTotalType > getInputEtVlMinusWheel() const { return m_etVlMinusPipe.contents; }
	/// return input Ht value wheel 0
	inline std::vector< etHadType   > getInputHtVlMinusWheel() const { return m_htVlMinusPipe.contents; }
	/// return input Ht value wheel 0
	inline std::vector< etComponentType > getInputHxVlMinusWheel() const { return m_hxVlMinusPipe.contents; }
	inline std::vector< etComponentType > getInputHyVlMinusWheel() const { return m_hyVlMinusPipe.contents; }

        /// Access to output quantities for all bunch crossings
	/// return output missing Et magnitude
	inline std::vector< etMissType >    getEtMissColl()    const { return m_outputEtMiss.contents; }
	/// return output missing Et value
	inline std::vector< etMissPhiType > getEtMissPhiColl() const { return m_outputEtMissPhi.contents; }
	/// return output total scalar Et
	inline std::vector< etTotalType >   getEtSumColl()     const { return m_outputEtSum.contents; }
	/// return std::vector< output calibrated jet Et
	inline std::vector< etHadType >     getEtHadColl()     const { return m_outputEtHad.contents; }
	/// return output missing Ht magnitude
	inline std::vector< etMissType >    getHtMissColl()    const { return m_outputHtMiss.contents; }
	/// return output missing Ht value
	inline std::vector< etMissPhiType > getHtMissPhiColl() const { return m_outputHtMissPhi.contents; }

	void setJetFinderParams(const L1GctJetFinderParams* const jfpars);
	void setHtMissScale(const L1CaloEtScale* const scale);

	// get the missing Ht LUT (used by L1GctPrintLuts)
	const L1GctHtMissLut* getHtMissLut() const { return m_mhtComponents.getHtMissLut(); }

	/// check setup
	bool setupOk() const;
  
 protected:
	/// Separate reset methods for the processor itself and any data stored in pipelines
	virtual void resetProcessor();
	virtual void resetPipelines();

	/// Initialise inputs with null objects for the correct bunch crossing if required
	virtual void setupObjects() {}
	
 private:
	// Here are the algorithm types we get our inputs from
	L1GctWheelEnergyFpga* m_plusWheelFpga;
	L1GctWheelEnergyFpga* m_minusWheelFpga;
	L1GctWheelJetFpga* m_plusWheelJetFpga;
	L1GctWheelJetFpga* m_minusWheelJetFpga;

	// Here's the class that does the Hf sums
	L1GctGlobalHfSumAlgos* m_hfSumProcessor;

	// Missing Et and missing Ht
	L1GctMet m_metComponents;
	L1GctMet m_mhtComponents;

	// input data
	etComponentType m_exValPlusWheel;
	etComponentType m_eyValPlusWheel;
	etTotalType m_etValPlusWheel;
	etHadType   m_htValPlusWheel;
	etComponentType m_hxValPlusWheel;
	etComponentType m_hyValPlusWheel;

	etComponentType m_exVlMinusWheel;
	etComponentType m_eyVlMinusWheel;
	etTotalType m_etVlMinusWheel;
	etHadType   m_htVlMinusWheel;
	etComponentType m_hxVlMinusWheel;
	etComponentType m_hyVlMinusWheel;

	// stored copies of input data
	Pipeline< etComponentType > m_exValPlusPipe;
	Pipeline< etComponentType > m_eyValPlusPipe;
	Pipeline< etTotalType > m_etValPlusPipe;
	Pipeline< etHadType >   m_htValPlusPipe;
	Pipeline< etComponentType > m_hxValPlusPipe;
	Pipeline< etComponentType > m_hyValPlusPipe;

	Pipeline< etComponentType > m_exVlMinusPipe;
	Pipeline< etComponentType > m_eyVlMinusPipe;
	Pipeline< etTotalType > m_etVlMinusPipe;
	Pipeline< etHadType >   m_htVlMinusPipe;
	Pipeline< etComponentType > m_hxVlMinusPipe;
	Pipeline< etComponentType > m_hyVlMinusPipe;

	// output data
	Pipeline<etMissType>    m_outputEtMiss;
	Pipeline<etMissPhiType> m_outputEtMissPhi;
	Pipeline<etTotalType>   m_outputEtSum;
	Pipeline<etHadType>     m_outputEtHad;
	Pipeline<etMissType>    m_outputHtMiss;
	Pipeline<etMissPhiType> m_outputHtMissPhi;

	bool m_setupOk;

};

std::ostream& operator << (std::ostream& os, const L1GctGlobalEnergyAlgos& fpga);

#endif /*L1GCTGLOBALENERGYALGOS_H_*/
