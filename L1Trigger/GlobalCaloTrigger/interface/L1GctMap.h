#ifndef L1GCTMAP_H_
#define L1GCTMAP_H_

class L1GctMap
{
public:
	~L1GctMap();
	
	static L1GctMap* theMap();
	
	// convert from bits to float
	float etaFromUnsigned(unsigned eta_u);
	float phiFromUnsigned(unsigned phi_u);
	
	// convert from float to bits
	unsigned etaFromFloat(float eta_f);
	unsigned phiFromFloat(float phi_f);
	
	// should this class also handle et scale???
	float etFromRank(unsigned rank_u);
	
private:
	L1GctMap();	
	
private:
	L1GctMap* instance;
	
};

#endif /*L1GCTMAP_H_*/
