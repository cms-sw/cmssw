#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"

#include <vector>
#include <functional>
#include <ostream>

using std::binary_function;

class L1GctEmLeafCard;

class L1GctElectronFinalSort : public L1GctProcessor
{
public:
       
	
         L1GctElectronFinalSort(bool iso, L1GctEmLeafCard* card1, L1GctEmLeafCard* card2);
	~L1GctElectronFinalSort();
	///
	/// clear internal buffers
	virtual void reset();
	///
	/// get input data from sources
	virtual void fetchInput();
	///
	/// process the data, fill output buffers
	virtual void process();
	///
	/// set input data
	void setInputEmCand(int i, L1GctEmCand cand);
	///
	/// return input data
	inline std::vector<L1GctEmCand> getInputCands() { return m_inputCands; }
	///
	/// return output data
	inline std::vector<L1GctEmCand> getOutputCands() { return m_outputCands; }
	friend std::ostream& operator<<(std::ostream& s,const L1GctElectronFinalSort& cand); 

 private:

	// comparison operator for sort
	struct rank_gt : public binary_function<L1GctEmCand, L1GctEmCand, bool> {
	  bool operator()(const L1GctEmCand& x, const L1GctEmCand& y) { return x.rank() > y.rank(); }
	};

 private:

	///
	/// type of em cand
	bool m_emCandsType;
	///
	/// the 1st stage electron sorters
	std::vector<L1GctEmLeafCard*> m_theLeafCards;
	///
	/// input data
	std::vector<L1GctEmCand> m_inputCands;
	///
	/// output data
	std::vector<L1GctEmCand> m_outputCands;

};

std::ostream& operator<<(std::ostream& s,const L1GctElectronFinalSort& cand); 

#endif /*L1GCTELECTRONFINALSORT_H_*/
