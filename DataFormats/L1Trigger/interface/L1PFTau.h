#ifndef L1PFTAU_H
#define L1PFTAU_H

#include <ostream>
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
//#include <TLorentzVector.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"


namespace l1t {

  class L1PFTau;
  typedef std::vector<L1PFTau> L1PFTauCollection;

  class L1PFTau : public reco::LeafCandidate {

  public:

    /// default constructor
    L1PFTau();
    
    //L1PFTau(const L1PFTau &input){  
    //setEt( input.et() );
    //setPtEtaPhiE( input.p4().Pt(), input.p4().Eta(), input.p4().Phi(), input.p4().Pt());
    //};
    
    /// destructor
    ~L1PFTau();
    
    
    // get/set methods for the data
    
    /// reset the data content (not position id!)
    void reset() { m_data = 0; }
    
    /// get raw data
    uint16_t raw() const { return m_data; }
    
    /// get Et
    //unsigned et() const { return (m_et); }
    int towerEtaSide() const{ return (m_towerEtaSide); }
    unsigned towerEta() const{ return (m_towerEta); }
    unsigned towerPhi() const{ return (m_towerPhi); }
    TTTrack< Ref_Phase2TrackerDigi_ > trackRef() const{return (m_trackRef);}
    
    void setEt(unsigned inputEt) { m_et = inputEt;}
    void setTowerEta(unsigned inputEta ) { (m_towerEta = inputEta); }
    void setTowerEtaSide(int inputEtaSide ) { (m_towerEtaSide = inputEtaSide); }
    void setTowerPhi(unsigned inputPhi ) { (m_towerPhi = inputPhi); }
    void setEoH(unsigned inputEoH) {(m_EoH = inputEoH);}
    void setHoE(float inputHoE){m_HoE = inputHoE;};  

    /// set data
    void setRawData(uint32_t data) { m_data = data; }

    // reco level quantities to be set manually, temporary aid for algo development
    //LorentzVector p4() const {return m_p4;};
    float ecalEnergy() const{return m_ecalEnergy;}
    float hcalEnergy() const{return m_hcalEnergy;}
    float caloEnergy() const{return m_caloEnergy;}
    float hwEta() const{return m_hwEta;}
    float hwPhi() const{return m_hwPhi;}
    float HoE() const{return m_HoE;}
    float EoH() const{return m_EoH;}
    float relIso() const{return m_relativeIsolation;}
    float rawIso() const{return m_rawIsolation;}
    float chargedIso() const{return m_chargedIsolation;}
    float neutralIso() const{return m_neutralIsolation;}
    short int tauType()const{return m_tauType;}

    //void setp4(LorentzVector input) {m_p4 = input;};
    //void setPtEtaPhiE(float pt, float eta, float phi, float et){m_p4.SetPtEtaPhiE(pt,eta,phi,et);};
    void setEcalEnergy(float input){ m_ecalEnergy = input;};
    void setHcalEnergy(float input){ m_hcalEnergy = input;};
    void setCaloEnergy(float input){ m_caloEnergy = input;};
    void setTrackRef(TTTrack< Ref_Phase2TrackerDigi_ > trackRef){m_trackRef = trackRef;};
    void setRelIso(float inputIso){m_relativeIsolation = inputIso;};

    void setTauType(short int input){ m_tauType = input;};
    void setRawIso(float inputIso){m_rawIsolation = inputIso;};
    void setChargedIso(float inputIso){m_chargedIsolation = inputIso;};
    void setNeutralIso(float inputIso){m_neutralIsolation = inputIso;};

    void setHWPhi(float inputPhi){
      m_hwPhi = round(inputPhi/0.0174)*0.0174;};

    void setHWEta(float inputEta){ 
      m_hwEta = round(inputEta/0.0174)*0.0174;};

    /// is there any information in the candidate
    bool empty() const { return (m_data == 0); }
    /*
     * Strip tracks i.e. e/g cands
     */

    /// print to stream

    friend std::ostream& operator << (std::ostream& os, const l1t::L1PFTau& reg);

  private:

    uint16_t m_data;
    LorentzVector m_p4;
    //for temporary use
    float m_ecalEnergy;
    float m_hcalEnergy;
    float m_caloEnergy;
    float m_relativeIsolation;
    
    float m_hwPhi;
    float m_hwEta;

    unsigned m_towerEta;
    int m_towerEtaSide;
    unsigned m_towerPhi;
    unsigned m_maxCrystalEta;
    unsigned m_maxCrystalPhi;
    unsigned m_et;
    unsigned m_EoH;
    unsigned m_HoE;

    short int m_tauType;
    float m_rawIsolation;
    float m_chargedIsolation;
    float m_neutralIsolation;


    TTTrack< Ref_Phase2TrackerDigi_ > m_trackRef;
  };

};
#endif /*L1PFTAU_H*/
