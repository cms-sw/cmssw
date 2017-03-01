// Class for muon tracks in EMTF - AWB 04.01.16
// Mostly copied from L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h

#ifndef __l1t_EMTFTrackExtra_h__
#define __l1t_EMTFTrackExtra_h__

#include <vector>
#include <boost/cstdint.hpp> 

#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/EMTFHitExtra.h"


namespace l1t {
  class EMTFTrackExtra: public EMTFTrack {
  public:
    
  EMTFTrackExtra() :
    first_bx(-999), second_bx(-999), pt_XML(-999), theta_int(-999), theta_deg(-999), theta_rad(-999), 
      type(-999), rank(-999), layer(-999), straightness(-999), strip(-999), isGMT(-999), numHitsExtra(0)
	{};
    
    virtual ~EMTFTrackExtra() {};

    EMTFTrack CreateEMTFTrack();

    void set_HitsExtra(EMTFHitExtraCollection bits)  { _HitsExtra = bits;                numHitsExtra = _HitsExtra.size(); }
    void push_HitExtra(EMTFHitExtra bits)            { _HitsExtra.push_back(bits);       numHitsExtra = _HitsExtra.size(); }
    void set_HitExtraIndices(std::vector<uint> bits) { _HitExtraIndices = bits;          numHitsExtra = _HitExtraIndices.size(); }
    void push_HitExtraIndex(uint bits)               { _HitExtraIndices.push_back(bits); numHitsExtra = _HitExtraIndices.size(); }

    int NumHitsExtra()                             const { return numHitsExtra; }
    EMTFHitExtraCollection HitsExtra()                   { return _HitsExtra; }
    std::vector<uint> HitExtraIndices()                  { return _HitExtraIndices; }
    const EMTFHitExtraCollection * PtrHitsExtra()  const { return &_HitsExtra; }
    const std::vector<uint> * PtrHitExtraIndices() const { return &_HitExtraIndices; }
    
    /* // Can't have a vector of vectors of vectors in ROOT files */
    /* void set_deltas (vector< vector<int> > _deltas) { deltas = _deltas; } */
    void set_phis   (std::vector<int> _phis)   { phis   = _phis; }
    void set_thetas (std::vector<int> _thetas) { thetas = _thetas; }
    
    void set_first_bx      (int  bits) { first_bx     = bits; }
    void set_second_bx     (int  bits) { second_bx    = bits; }
    void set_pt_XML        (float val) { pt_XML       = val;  }
    void set_theta_int     (int  bits) { theta_int    = bits; }
    void set_theta_deg     (float val) { theta_deg    = val;  }
    void set_theta_rad     (float val) { theta_rad    = val;  }
    void set_type          (int  bits) { type         = bits; }
    void set_rank          (int  bits) { rank         = bits; }
    void set_layer         (int  bits) { layer        = bits; }
    void set_straightness  (int  bits) { straightness = bits; }
    void set_strip         (int  bits) { strip        = bits; }
    void set_isGMT         (int  bits) { isGMT        = bits; }
    
    int   First_BX()      const { return  first_bx;     }
    int   Second_BX()     const { return  second_bx;    }
    float Pt_XML()        const { return  pt_XML;       }
    int   Theta_int()     const { return  theta_int;    }
    float Theta_deg()     const { return  theta_deg;    }
    float Theta_rad()     const { return  theta_rad;    }
    int   Type()          const { return  type;         }
    int   Rank()          const { return  rank;         }
    int   Layer()         const { return  layer;        }
    int   Straightness()  const { return  straightness; }
    int   Strip()         const { return  strip;        }
    int   IsGMT()         const { return  isGMT;        }
    
  private:
    
    EMTFHitExtraCollection _HitsExtra;
    std::vector<uint>  _HitExtraIndices;

    /* // Can't have a vector of vectors of vectors in ROOT files */
    /* std::vector< std::vector<int> > deltas; */
    std::vector<int> phis;
    std::vector<int> thetas;
    
    int   first_bx;     //  ? -  ?.  Filled in emulator.
    int   second_bx;    //  ? -  ?.  Filled in emulator.
    float pt_XML;       //  ? -  ?.  Filled in emulator.
    int   theta_int;    //  ? -  ?.  Filled in emulator.
    float theta_deg;    //  ? -  ?.  Filled in emulator.
    float theta_rad;    //  ? -  ?.  Filled in emulator.
    int   type;         //  Don't remember what this is - AWB 06.04.16
    int   rank;         //  ? -  ?.  Filled in emulator.
    int   layer;        //  ? -  ?.  Computed in BXAnalyzer.h.  How can we access?
    int   straightness; //  ? -  ?.  Filled in emulator.
    int   strip;        //  ? -  ?.  Computed in SortSector.h.  How can we access?
    int   isGMT;        //  0 or 1.  Filled in emulator.
    int   numHitsExtra;
    
  }; // End of class EMTFTrackExtra
  
  // Define a vector of EMTFTrackExtra
  typedef std::vector<EMTFTrackExtra> EMTFTrackExtraCollection;
  
} // End of namespace l1t

#endif /* define __l1t_EMTFTrackExtra_h__ */
