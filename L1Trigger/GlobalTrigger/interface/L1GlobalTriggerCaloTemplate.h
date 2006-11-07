#ifndef GlobalTrigger_L1GlobalTriggerCaloTemplate_h 
#define GlobalTrigger_L1GlobalTriggerCaloTemplate_h

/**
 * \class L1GlobalTriggerCaloTemplate
 * 
 * 
 * 
 * Description: Single particle chip - description for calo conditions
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
#include <iosfwd>
#include <string>

// user include files
//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

// forward declarations
class L1GlobalTrigger;

// class declaration
class L1GlobalTriggerCaloTemplate : public L1GlobalTriggerConditions 
{

public:

    // constructor
    L1GlobalTriggerCaloTemplate(const L1GlobalTrigger&, const std::string&);
    
    // copy constructor
    L1GlobalTriggerCaloTemplate( const L1GlobalTriggerCaloTemplate& );

    // destructor
    virtual ~L1GlobalTriggerCaloTemplate();
    
    // assign operator
    L1GlobalTriggerCaloTemplate& operator= (const L1GlobalTriggerCaloTemplate&);

public:

    // type for a single particle
    typedef struct
    {
        unsigned int et_threshold; 
        unsigned int eta;
        unsigned int phi;
    } ParticleParameter;

    // non particle parameters
    typedef struct
    {
        unsigned int delta_eta; // parameters for correlation
        unsigned int delta_phi;
        unsigned int delta_eta_maxbits;
        unsigned int delta_phi_maxbits;
    } ConditionParameter;
     
    enum ParticleType {
        EG=0,
        IEG,
        CJET,
        FJET,
        TJET
    };       
   
    // set functions
    void setConditionParameter(unsigned int numparticles, 
        const ParticleParameter *particlep, const ConditionParameter *conditionp, 
        ParticleType pType, bool wsc);
    
    // get functions
    inline const ParticleParameter* getParticleParameter() const { return this->p_particleparameter; }
    inline const ConditionParameter* getConditionParameter() const { return &(this->p_conditionparameter); }
    inline unsigned int getNumberParticles() const { return p_number; }
    inline bool getWsc() const { return p_wsc; }        
 
//    virtual const bool blockCondition() const;
    const bool blockCondition() const;

    // print thresholds
    void printThresholds(std::ostream& myCout) const;
    

private:   
   
    // number of particle parameters in this condition
    unsigned int p_number;

    // indicator that this condition contains a eta phi correlation condition
    bool p_wsc;

    /// what type of particles is this condition for
    ParticleType p_particletype;

    /// variables containing the parameters
    ParticleParameter p_particleparameter[4]; //TODO: macro instead of 4
    ConditionParameter p_conditionparameter;
     
    /// copy function for copy constructor and operator=
    void copy( const L1GlobalTriggerCaloTemplate& cp);

    // load calo candidates
    virtual L1GctCand* getCandidate( int indexCand ) const;

    // function to check a single particle if it matches a condition
    const bool checkParticle(int ncondition, L1GctCand &cand) const;

};
   
#endif 
