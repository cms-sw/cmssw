#ifndef L1GCTSCALES_H_
#define L1GCTSCALES_H_

typdef unsigned long ULong;

class L1GctScales
{
public:
	~L1GctScales();

	static L1GctScales* getGctScales();
	
	float etFromRank(ULong rank);
	ULong rankFromEt(float et);
	
private:
	L1GctScales();
	
private:
	static L1GctScales* instance;

};

#endif /*L1GCTSCALES_H_*/
