#ifndef GlobalTrigger_L1GlobalTriggerMuonTemplate_h 
#define GlobalTrigger_L1GlobalTriggerMuonTemplate_h

/**
 * \class L1GlobalTriggerMuonTemplate
 * 
 * 
 * Description: Single particle chip, description for muon conditions.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder, H. Rohringer - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <string>
#include <iosfwd>

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
        unsigned int pt_h_threshold;
        unsigned int pt_l_threshold;
        bool en_mip; 
        bool en_iso;                  // isolation bit
        unsigned int quality; 
        u_int64_t eta; 
        unsigned int phi_h; 
        unsigned int phi_l; 
    } ParticleParameter;

    // correlation parameters
    typedef struct
    {
        unsigned int charge_correlation; 
        u_int64_t delta_eta;             
        unsigned int delta_eta_maxbits;
                                         
        u_int64_t delta_phil;            
        u_int64_t delta_phih;            
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
    void printThresholds(std::ostream& myCout) const;

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
        
};

#endif 
