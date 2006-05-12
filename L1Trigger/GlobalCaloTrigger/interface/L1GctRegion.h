#ifndef L1GCTREGION_H_
#define L1GCTREGION_H_

#include <bitset>

/*
 * A calorimeter trigger region
 * as represented in the GCT
 * author: Jim Brooke, Robert Frazier
 * date: 20/2/2006
 * 
 */

typedef unsigned long int ULong;

class L1GctRegion
{
public:
	L1GctRegion(ULong data=0);
	L1GctRegion(int eta, int phi, ULong et, bool mip, bool quiet, bool tauVeto, bool overFlow);
	~L1GctRegion();
    	
    // Getters
    int eta() const { return m_eta; }   ///< Get the eta number (0-21?) of the region
    int phi() const { return m_phi; }   ///< Get the phi number (0-17) of the region
    ULong getEt() const { return myEt.to_ulong(); }
    bool getMip() const { return myMip; }
    bool getQuiet() const { return myQuiet; }
    bool getTauVeto() const { return myTauVeto; }
    bool getOverFlow() const { return myOverFlow; }
    
    // Setters
    void setEta(int eta) { m_eta = eta; }
    void setPhi(int phi) { m_phi = phi; }
    void setEt(ULong et) { /*assert(et < (1 << ET_BITWIDTH));*/ myEt = et; } 
    void setMip(bool mip) { myMip = mip; }
    void setQuiet(bool quiet) { myQuiet = quiet; }
    void setTauVeto(bool tauVeto) { myTauVeto = tauVeto; }
    void setOverFlow(bool overFlow) { myOverFlow = overFlow; }

    //  ostream& operator << (ostream& os, const L1GctRegion& s);
    

		
private:

    static const int ET_BITWIDTH = 10;

    /// global eta position number of the region (0-21)
	int m_eta;
	/// global phi position number of the region (0-17)
    int m_phi;

	std::bitset<ET_BITWIDTH> myEt;
	bool myMip;
	bool myQuiet;
    bool myTauVeto;
    bool myOverFlow;
	
	
	
};

#endif /*L1GCTREGION_H_*/
