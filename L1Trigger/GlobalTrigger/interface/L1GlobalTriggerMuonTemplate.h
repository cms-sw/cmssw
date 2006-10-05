#ifndef GlobalTrigger_L1GlobalTriggerMuonTemplate_h 
#define GlobalTrigger_L1GlobalTriggerMuonTemplate_h

/**
 * \class L1GlobalTriggerMuonTemplate
 * 
 * 
 * 
 * Description: Single particle chip, description for muon conditions
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder, H. Rohringer - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

// user include files
//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

 
// forward declarations
class L1GlobalTrigger;

// class declaration
class L1GlobalTriggerMuonTemplate : public L1GlobalTriggerConditions 
{

public:

    // constructor
    L1GlobalTriggerMuonTemplate(const L1GlobalTrigger&, const std::string&);
  
    // copy constructor
    L1GlobalTriggerMuonTemplate( const L1GlobalTriggerMuonTemplate& );

    // destructor
    virtual ~L1GlobalTriggerMuonTemplate();
  
    // assign operator
    L1GlobalTriggerMuonTemplate& operator= (const L1GlobalTriggerMuonTemplate&);

    // load muon candidates
    L1MuGMTCand* getCandidate( int indexCand ) const;

public:

    // type for a single particle template
    typedef struct                                   // TODO: keep it a` la C?
    {    
        unsigned int pt_h_threshold; // usually 5 bits
        unsigned int pt_l_threshold; // usually 5 bits
        bool en_mip; //
        bool en_iso; // isolation bit
        unsigned int quality; // usually 8 bits
        u_int64_t eta; // usually 64 bits // TODO really?
        unsigned int phi_h; // usually 8 bits
        unsigned int phi_l; // usually 8 bits
    } ParticleParameter;

    // correlation parameters
    typedef struct
    {
        unsigned int charge_correlation; // usually  3 bits
        u_int64_t delta_eta;             // usually 64 bits
        unsigned int delta_eta_maxbits;
                                         // 73 = 64+8 bits for delta phi
        u_int64_t delta_phil;            // 64 bits
        u_int64_t delta_phih;            //  9 bits
        unsigned int delta_phi_maxbits; 
    } ConditionParameter;
        
    // set functions
    void setConditionParameter(unsigned int numparticles, 
        const ParticleParameter *particlep, 
        const ConditionParameter *conditionp, bool wsc);
  
    // get functions
    inline const ParticleParameter* getParticleParameter() const { 
        return this->p_particleparameter; 
    }
    inline const ConditionParameter* getConditionParameter() const { 
        return &(this->p_conditionparameter); 
    }
    inline unsigned int getNumberParticles() const { return p_number; }
    inline bool getWsc() const { return p_wsc; }

    virtual const bool blockCondition() const;
  
    // print thresholds
    void printThresholds() const;

private:

    // number of particle parameters in this condition
    unsigned int p_number;

    // indicator that this condition contains a eta phi correlation condition
    bool p_wsc; 
   
    // variables containing the parameters
    ParticleParameter p_particleparameter[4]; // TODO: macro instead of 4
    ConditionParameter p_conditionparameter;
   
    // copy function for copy constructor and operator=
    void copy( const L1GlobalTriggerMuonTemplate& cp);

    const bool checkParticle(int nconditon, L1MuGMTCand &cand) const;

    // TODO why templated? 
    template<class Type1>
        const bool checkBitM(Type1 const &mask, unsigned int bitNumber) const; 
        
};

#endif 
