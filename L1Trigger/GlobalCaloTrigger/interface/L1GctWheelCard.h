#ifndef L1GCTWHEELCARD_H_
#define L1GCTWHEELCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"


#include <vector>

using namespace std;

/*
 * Class to represent a GCT Wheel Card
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 * This class does no processing, it is only here to
 * accurately model the hardware
 * 
 */

class L1GctWheelCard
{
public:
	L1GctWheelCard();
	~L1GctWheelCard();
		
	inline vector<L1GctJetLeafCard*> getJetLeafCards() { return jetLeafCards; }
		
	void process();
	
	vector<L1GctJet> getOutput();
	
private:

	vector<L1GctJetLeafCard*> jetLeafCards;
	
};

#endif /*L1GCTWHEELCARD_H_*/
