#ifndef L1TGlobalScales_h
#define L1TGlobalScales_h

/**
 * \class L1TGlobalScales
 *
 *
 * Description: Accessor Class for uGT Result
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 */

// system include files
#include <vector>
#include <string>
#include <iosfwd>

// class declaration

namespace l1t {

class L1TGlobalScales
{

public:

    // constructors
  L1TGlobalScales();

    // destructor
    virtual ~L1TGlobalScales();

public:

    /// typedef for a single object template
    struct ScaleParameters
    {
        double etMin;
	double etMax;
	double etStep;
	std::vector<std::pair<double,double>> etBins;

        double phiMin;
	double phiMax;
	double phiStep;
	std::vector<std::pair<double,double>> phiBins;
	
        double etaMin;
	double etaMax;
	double etaStep;
	std::vector<std::pair<double,double>> etaBins;	
	
    };
     
    inline void setEGScales(ScaleParameters& scales)   { m_egScales = scales;  }
    inline void setTauScales(ScaleParameters& scales)  { m_tauScales = scales; }
    inline void setMuonScales(ScaleParameters& scales) { m_muScales = scales;  }
    
    inline void setJetScales(ScaleParameters& scales) { m_jetScales = scales; }
    
    inline void setETTScales(ScaleParameters& scales) { m_ettScales = scales; }
    inline void setHTTScales(ScaleParameters& scales) { m_httScales = scales; }
    inline void setETMScales(ScaleParameters& scales) { m_etmScales = scales; }
    inline void setHTMScales(ScaleParameters& scales) { m_htmScales = scales; }
    
    inline void setScalesName(std::string name) { m_ScaleSetName = name; }
    virtual std::string getScalesName() const; 
    
    virtual void print(std::ostream& myCout) const;
    virtual void printScale(ScaleParameters scale, std::ostream& myCout) const;

private:

    
private:


   std::string m_ScaleSetName;
    
   ScaleParameters m_egScales;
   ScaleParameters m_tauScales;
   ScaleParameters m_muScales;
   
   ScaleParameters m_jetScales;
   
   ScaleParameters m_ettScales;
   ScaleParameters m_httScales;
   ScaleParameters m_etmScales;
   ScaleParameters m_htmScales; 
            
};

}
#endif
