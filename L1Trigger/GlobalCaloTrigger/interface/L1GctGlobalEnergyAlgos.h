#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"


#include <vector>

class L1GctWheelEnergyFpga;
class L1GctWheelJetFpga;

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
        typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissNBits    > etMissType;
        typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissPhiNBits > etMissPhiType;
        typedef L1GctJetLeafCard::etComponentType etComponentType;

        /// Number of jet counter per wheel
        static const unsigned int N_JET_COUNTERS_USED;
        static const unsigned int N_JET_COUNTERS_MAX;

        /// Constructor needs the Wheel card Fpgas set up first
	 L1GctGlobalEnergyAlgos(std::vector<L1GctWheelEnergyFpga*> WheelFpga,
			       std::vector<L1GctWheelJetFpga*> WheelJetFpga);
	 /// Destructor
	~L1GctGlobalEnergyAlgos();

        /// Overload << operator
        friend std::ostream& operator << (std::ostream& os, const L1GctGlobalEnergyAlgos& fpga);

	/// clear internal buffers
	virtual void reset();

	/// get input data from sources; this is the standard way to provide input
	virtual void fetchInput();

	/// process the data, fill output buffers
	virtual void process();

	/// set input Ex value per wheel (0 or 1); not used in normal operation
	void setInputWheelEx(unsigned wheel, int energy, bool overflow);
	/// set input Ey value per wheel (0 or 1); not used in normal operation
	void setInputWheelEy(unsigned wheel, int energy, bool overflow);
	/// set input Et value per wheel (0 or 1); not used in normal operation
	void setInputWheelEt(unsigned wheel, unsigned energy, bool overflow);
	/// set input Ht value per wheel (0 or 1); not used in normal operation
	void setInputWheelHt(unsigned wheel, unsigned energy, bool overflow);

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

	/// return input Ex value wheel 1
       inline etComponentType getInputExValPlusWheel() const { return m_exValPlusWheel; }
	/// return input Ex value wheel 1
       inline etComponentType getInputEyValPlusWheel() const { return m_eyValPlusWheel; }
	/// return input Ey value wheel 0
       inline etComponentType getInputExVlMinusWheel() const { return m_exVlMinusWheel; }
	/// return input Ey value wheel 0
       inline etComponentType getInputEyVlMinusWheel() const { return m_eyVlMinusWheel; }
	/// return input Et value wheel 1
	inline etTotalType getInputEtValPlusWheel() const { return m_etValPlusWheel; }
	/// return input Ht value wheel 1
	inline etHadType   getInputHtValPlusWheel() const { return m_htValPlusWheel; }
	/// return input Et value wheel 0
	inline etTotalType getInputEtVlMinusWheel() const { return m_etVlMinusWheel; }
	/// return input Ht value wheel 0
	inline etHadType   getInputHtVlMinusWheel() const { return m_htVlMinusWheel; }
	/// return input jet count (number 0-11) wheel 1
       inline L1GctJetCount<3> getInputJcValPlusWheel(unsigned jcnum) const {return m_jcValPlusWheel.at(jcnum); }
	/// return input jet count (number 0-11) wheel 0
       inline L1GctJetCount<3> getInputJcVlMinusWheel(unsigned jcnum) const {return m_jcVlMinusWheel.at(jcnum); }

	/// return output missing Et magnitude
	inline etMissType    getEtMiss()    const { return m_outputEtMiss; }
	/// return output missing Et value
	inline etMissPhiType getEtMissPhi() const { return m_outputEtMissPhi; }
	/// return output total scalar Et
	inline etTotalType   getEtSum()     const { return m_outputEtSum; }
	/// return output calibrated jet Et
	inline etHadType     getEtHad()     const { return m_outputEtHad; }
	/// return output jet count (number 0-11)
	inline L1GctJetCount<5> getJetCount(unsigned jcnum) const
         { return ( jcnum<N_JET_COUNTERS_USED ? m_outputJetCounts.at(jcnum) : 0); }
	inline L1GctJetCount<5> getJetCountBits(unsigned jcnum) const
         { return ( jcnum<N_JET_COUNTERS_MAX  ? m_outputJetCounts.at(jcnum) : 0); }

       /// return vector of jet count values
       std::vector<unsigned> getJetCountValues() const;
	
private:
	
	// Here are the algorithm types we get our inputs from
	L1GctWheelEnergyFpga* m_plusWheelFpga;
	L1GctWheelEnergyFpga* m_minusWheelFpga;
	L1GctWheelJetFpga* m_plusWheelJetFpga;
	L1GctWheelJetFpga* m_minusWheelJetFpga;

	// input data
	etComponentType m_exValPlusWheel;
       etComponentType m_eyValPlusWheel;
	etTotalType m_etValPlusWheel;
	etHadType   m_htValPlusWheel;
	etComponentType m_exVlMinusWheel;
	etComponentType m_eyVlMinusWheel;
	etTotalType m_etVlMinusWheel;
	etHadType   m_htVlMinusWheel;

        std::vector< L1GctJetCount<3> > m_jcValPlusWheel;
        std::vector< L1GctJetCount<3> > m_jcVlMinusWheel;

	// output data
	etMissType    m_outputEtMiss;
	etMissPhiType m_outputEtMissPhi;
	etTotalType   m_outputEtSum;
	etHadType     m_outputEtHad;
        std::vector< L1GctJetCount<5> > m_outputJetCounts;

        // PRIVATE MEMBER FUNCTION
	// the Etmiss algorithm
        struct etmiss_vec {
	  etMissType    mag;
	  etMissPhiType phi;
	};
        etmiss_vec calculate_etmiss_vec (const etComponentType ex, const etComponentType ey) const ;
	
       // Function to use the jet count bits for Hf Et sums
       void packHfTowerSumsIntoJetCountBits(); 
};

std::ostream& operator << (std::ostream& os, const L1GctGlobalEnergyAlgos& fpga);

#endif /*L1GCTGLOBALENERGYALGOS_H_*/
