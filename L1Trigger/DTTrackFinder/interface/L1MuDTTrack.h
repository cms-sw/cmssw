//-------------------------------------------------
//
/**  \class L1MuDTTrack
 *
 *   L1 Muon Track Candidate
 *
 *
 *   $Date: 2007/04/10 13:04:51 $
 *   $Revision: 1.5 $
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDT_TRACK_H
#define L1MUDT_TRACK_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>
#include <string>
#include <vector>
#include <functional>

//----------------------
// Base Class Headers --
//----------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssParam.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTAddressArray.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegEta.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"


//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTTrack : public L1MuRegionalCand {

  public:

    /// default constructor
    L1MuDTTrack();

    /// constructor   
    L1MuDTTrack(const L1MuDTSecProcId&);
   
    /// copy constructor
    L1MuDTTrack(const L1MuDTTrack&);

    /// destructor
    virtual ~L1MuDTTrack();

    /// reset muon candidate
    void reset();

    /// get name of object
    inline std::string name() const { return m_name; }

    /// get pt-code (5 bits)
    inline unsigned int pt() const { return pt_packed(); }

    /// get phi-code (8 bits)
    inline unsigned int phi() const { return phi_packed(); }

    /// get eta-code (6 bits)
    inline unsigned int eta() const { return eta_packed(); }
    
    /// get fine eta bit
    inline bool fineEtaBit() const { return isFineHalo(); }
    
    /// get charge (1 bit)
    inline int charge() const { return chargeValue(); }
    
    /// get track-class
    inline TrackClass tc() const { return m_tc; }

    /// is it an empty  muon candidate?
    inline bool empty() const { return m_empty; }
    
    /// return Sector Processor in which the muon candidate was found
    inline const L1MuDTSecProcId& spid() const { return m_spid; }
    
    /// get address-array for this muon candidate
    inline L1MuDTAddressArray address() const { return m_addArray; }
    
    /// get relative address of a given station
    inline int address(int stat) const { return m_addArray.station(stat); }
    
    /// return number of phi track segments used to form the muon candidate
    inline int numberOfTSphi() const { return m_tsphiList.size(); }
    
    /// return number of eta track segments used to form the muon candidate
    inline int numberOfTSeta() const { return m_tsetaList.size(); }
    
    /// return all phi track segments of the muon candidate
    const std::vector<L1MuDTTrackSegPhi>& getTSphi() const { return m_tsphiList; }

    /// return start phi track segment of muon candidate
    const L1MuDTTrackSegPhi& getStartTSphi() const;
    
    /// return end phi track segment of muon candidate
    const L1MuDTTrackSegPhi& getEndTSphi() const;

    /// return all eta track segments of the muon candidate
    const std::vector<L1MuDTTrackSegEta>& getTSeta() const { return m_tsetaList; }
    
    /// return start eta track segment of muon candidate
    const L1MuDTTrackSegEta& getStartTSeta() const;
    
    /// return end eta track segment of muon candidate
    const L1MuDTTrackSegEta& getEndTSeta() const;

    /// enable muon candidate
    inline void enable() { m_empty = false; setType(0); }
    
    /// disable muon candidate
    inline void disable() { m_empty = true; }
       
    /// set name of object
    inline void setName(std::string name) { m_name = name; }
        
    /// set track-class of muon candidate
    inline void setTC(TrackClass tc) { m_tc = tc; }
    
    /// set phi-code of muon candidate
    inline void setPhi(int phi) { setPhiPacked(phi); }
    
    /// set eta-code of muon candidate
    void setEta(int eta);
    
    /// set fine eta bit
    inline void setFineEtaBit() { setFineHalo(true); }
    
    /// set pt-code of muon candidate
    inline void setPt(int pt) { setPtPacked(pt); }
    
    /// set charge of muon candidate
    inline void setCharge(int charge) { setChargeValue(charge); setChargeValid(true); }
    
    /// set quality of muon candidate
    inline void setQuality(unsigned int quality) { setQualityPacked(quality); }
    
    /// set relative addresses of muon candidate
    inline void setAddresses(const L1MuDTAddressArray& addr) { m_addArray = addr; }
    
    /// set phi track segments used to form the muon candidate
    void setTSphi(const std::vector<const L1MuDTTrackSegPhi*>& tsList);

    /// set eta track segments used to form the muon candidate 
    void setTSeta(const std::vector<const L1MuDTTrackSegEta*>& tsList);
 
    /// convert  pt value in GeV to pt code
    unsigned int triggerScale(float value, const edm::EventSetup& c) const;

    /// assignment operator
    L1MuDTTrack& operator=(const L1MuDTTrack&);

    /// equal operator
    bool operator==(const L1MuDTTrack&) const;
    
    /// unequal operator
    bool operator!=(const L1MuDTTrack&) const;

    /// print parameters of muon candidate
    void print() const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuDTTrack&);

    /// define a rank for muon candidates
    class Rank : std::binary_function< const L1MuDTTrack*, const L1MuDTTrack*, bool> {
      public :
        bool operator()( const L1MuDTTrack* first, const L1MuDTTrack* second ) const {
         unsigned short int rank_f = 0;  // rank of first
         unsigned short int rank_s = 0;  // rank of second
         if ( first )  rank_f = 10 * first->pt()  + first->quality(); 
         if ( second ) rank_s = 10 * second->pt() + second->quality(); 
         return rank_f > rank_s;
       }
    };


  private:

    L1MuDTSecProcId  m_spid;      // which SP found the track 
    std::string      m_name;
    bool             m_empty;      
    TrackClass       m_tc;

    L1MuDTAddressArray         m_addArray;
    std::vector<L1MuDTTrackSegPhi>  m_tsphiList;
    std::vector<L1MuDTTrackSegEta>  m_tsetaList;

};
  
#endif
