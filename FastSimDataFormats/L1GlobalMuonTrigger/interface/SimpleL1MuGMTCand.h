#ifndef SIMPLEL1MU_GMT_CAND_H
#define SIMPLEL1MU_GMT_CAND_H

#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h>

#include "DataFormats/Math/interface/LorentzVector.h"

class SimTrack;

namespace HepMC { 
  class GenParticle;
}

/** \class SimpleL1MuGMTCand
 *   Description: Simple L1 Global Muon Trigger Candidate 
 *   Inherits the basics from 'L1MuGMTCand' base class.
 *   Contains pointer to RawHepEventParticle from the
 *   event manager. Allows easy conversion from a 
 *   RawHepEventParticle.
 *
 *   Author:      Andrea Perrotta     05/09/2006
 */

class SimpleL1MuGMTCand : public L1MuGMTExtendedCand {

  public:

  typedef math::XYZTLorentzVector LorentzVector;

    /// constructor   
    SimpleL1MuGMTCand();
   
    /// copy constructor
    SimpleL1MuGMTCand(const SimpleL1MuGMTCand&);
    
    /// copy constructor from pointer
    SimpleL1MuGMTCand(const SimpleL1MuGMTCand*);
    
    /// convert a FSimTrack into a SimpleL1MuGMTCand (L1MuGMTExtendedCand)
    SimpleL1MuGMTCand(const SimTrack*);

    /// The same as above, but without relying on internal tables (safer)
    SimpleL1MuGMTCand(const SimTrack* p,
		      unsigned etaIndex, 
		      unsigned phiIndex,
		      unsigned pTIndex,
		      float etaValue,
		      float phiValue,
		      float pTValue);    

    /// destructor
    ~SimpleL1MuGMTCand() override;

    /// reset muon candidate
    void reset();

    /// get name of object
    inline std::string name() const { return m_name; }

    /// get phi-code
    inline unsigned int phi() const { return m_phi; }
    
    /// get eta-code
    inline unsigned int eta() const { return m_eta; }
    
    /// get pt-code
    inline unsigned int pt() const { return m_pt; }
    
    /// get charge
    inline int charge() const { return m_charge; }
    
    /// get rank
    inline unsigned int rank() const { return m_rank; }
     
    /// is it an empty  muon candidate?
    inline bool empty() const { return m_empty; }

    /// enable muon candidate
    inline void enable() { m_empty = false; }

    /// disable muon candidate
    inline void disable() { m_empty = true; }
    
    /// set phi-value and packed code of muon candidate
    void setPhi(float phi);
    
    /// set eta-value and packed code of muon candidate
    void setEta(float eta);
    
    /// set pt-value and packed code of muon candidate
    void setPt(float pt);
    
    /// set charge and packed code of muon candidate
    void setCharge(int charge);
    
    /// set rank
    inline void setRank(unsigned int rank) { m_rank = rank; }
    
    /// return the smeared L1 Pt value before discretization in 32-bit
    float smearedPt() const { return m_smearedPt; }
    
    /// nevermind this one 
    inline unsigned int linearizedPt(float lsbValue, unsigned maxScale) const { return 0; }

    /// get quality (not implemented for FAMOS)
    inline unsigned int quality() const { return m_quality; }

    unsigned int etaRegionIndex() const { return eta(); }

    unsigned int phiRegionIndex() const { return phi(); }

    // set and get the 4-momentum of the original (generator) particle
    void setMomentum(const LorentzVector& m) { myMomentum = m; }
    const LorentzVector getMomentum() const { return myMomentum; }

    /// assignment operator
    SimpleL1MuGMTCand& operator=(const SimpleL1MuGMTCand&);

    /// assignment operator for a FSimTrack
    SimpleL1MuGMTCand* operator=(const SimTrack*);

    /// equal operator
    bool operator==(const SimpleL1MuGMTCand&) const;
    
    /// unequal operator
    bool operator!=(const SimpleL1MuGMTCand&) const;

    /// print parameters of muon candidate
    void print() const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const SimpleL1MuGMTCand&);

    /// define a rank for muon candidates
    bool getRank( const SimpleL1MuGMTCand* first, const SimpleL1MuGMTCand* second ) const {
      unsigned int rank_f = (first) ? first->rank(): 0;
      unsigned int rank_s = (second) ? second->rank() : 0;
      return rank_f > rank_s;
    }

    static const float ptScale[32];
    static const float etaScale[63];
    static const float phiScale[144];

  private:

    std::string  m_name;
    bool	 m_empty;

    unsigned int m_phi;
    unsigned int m_eta;
    unsigned int m_pt;
    int          m_charge;
    unsigned int m_quality;
    unsigned int m_rank;
    float        m_smearedPt;

    LorentzVector myMomentum ;

};
  
#endif
