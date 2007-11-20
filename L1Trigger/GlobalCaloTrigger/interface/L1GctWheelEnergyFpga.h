#ifndef L1GCTWHEELENERGYFPGA_H_
#define L1GCTWHEELENERGYFPGA_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"

#include <vector>

class L1GctJetLeafCard;

/*!
 * \class L1GctWheelEnergyFpga
 * \brief Emulates the energy summing on a GCT Wheel card
 *
 * This class carries out the summing of total Et, and
 * missing Et components Ex and Ey, for a single Wheel.
 * The inputs come from the three Leaf cards and the
 * outputs are sent to the L1GctGlobalEnergyAlgos class.
 *
 * \author Jim Brooke & Greg Heath
 * \date 20/2/2006
 * 
 */

class L1GctWheelEnergyFpga : public L1GctProcessor
{
public:
        /// typedefs for energy values in fixed numbers of bits
        typedef L1GctUnsignedInt< L1GctEtTotal::kEtTotalNBits   > etTotalType;
        typedef L1GctJetLeafCard::etComponentType etComponentType;

        /// Max number of leaf card pointers
        static const unsigned int MAX_LEAF_CARDS;

        /// Constructor, needs the Leaf cards to be set up first. id should be 0 or 1.
	L1GctWheelEnergyFpga(int id, std::vector<L1GctJetLeafCard*> leafCards);
	/// Destructor
	~L1GctWheelEnergyFpga();

        /// Overload << operator
        friend std::ostream& operator << (std::ostream& os, const L1GctWheelEnergyFpga& fpga);

	/// clear internal buffers
	virtual void reset();

	/// get input data from sources; this is the standard way to provide input
	virtual void fetchInput();

	/// process the data, fill output buffers
	virtual void process();

	/// set input data; not used in normal operation
	void setInputEnergy(unsigned i, int ex, int ey, unsigned et);

	/// provide access to input Leaf card pointer (0-2)
	L1GctJetLeafCard* getinputLeafCard(unsigned leafnum) const { return m_inputLeafCards.at(leafnum); }

	/// get input Ex value from a Leaf card (0-2)
	inline etComponentType getInputEx(unsigned leafnum) const { return m_inputEx.at(leafnum); }
	/// get input Ey value from a Leaf card (0-2)
	inline etComponentType getInputEy(unsigned leafnum) const { return m_inputEy.at(leafnum); }
	/// get input Et value from a Leaf card (0-2)
	inline etTotalType getInputEt(unsigned leafnum) const { return m_inputEt.at(leafnum); }

	/// get output Ex value
	inline etComponentType getOutputEx() const { return m_outputEx; }
	/// get output Ey value
	inline etComponentType getOutputEy() const { return m_outputEy; }
	/// get output Et value
	inline etTotalType getOutputEt() const { return m_outputEt; }

private:

	///
	/// algo ID
	int m_id;
	///
	/// the jet leaf card
	std::vector<L1GctJetLeafCard*> m_inputLeafCards;
	///
	/// the input components from each input card
	std::vector< etComponentType > m_inputEx;
	std::vector< etComponentType > m_inputEy;
	std::vector< etTotalType > m_inputEt;
	///
	/// output data
	etComponentType m_outputEx;
	etComponentType m_outputEy;
	etTotalType m_outputEt;
	
	
};

std::ostream& operator << (std::ostream& os, const L1GctWheelEnergyFpga& fpga);

#endif /*L1GCTWHEELENERGYFPGA_H_*/
