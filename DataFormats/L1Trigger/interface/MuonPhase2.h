#ifndef DataFormats_L1Trigger_MuonPhase2_h
#define DataFormats_L1Trigger_MuonPhase2_h

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {

  class MuonPhase2;
  typedef BXVector<MuonPhase2> MuonPhase2BxCollection;
  typedef edm::Ref< MuonPhase2BxCollection > MuonPhase2Ref ;
  typedef edm::RefVector< MuonPhase2BxCollection > MuonPhase2RefVector ;
  typedef std::vector< MuonPhase2Ref > MuonPhase2VectorRef ;

  class MuonPhase2 : public Muon {

  public:
    MuonPhase2();

    MuonPhase2( const MuonPhase2& );

    MuonPhase2( const Muon& );

    ~MuonPhase2() override;

  private:

    // set values
    inline void setPtNoVtx(float ptNoVtx) { ptNoVtx_ = ptNoVtx; }

    // methods to retrieve values
    inline float getPtNoVtx() const { return ptNoVtx_; }

    // additional phase-2 members
    float ptNoVtx_;
  };
}

#endif
