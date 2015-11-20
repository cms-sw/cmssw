#ifndef __L1TMUON_CORRIDORPTREFINEMENT_H__
#define __L1TMUON_CORRIDORPTREFINEMENT_H__
// 
// Class: L1TMuon::CorridorPtRefinement
//
// Info: Implements the 'corridor' (confidence interval)
//       based tail-clipping from B. Scurlock.
//
// Author: L. Gray (FNAL), B. Scurlock (UF)
//
#include <memory>
#include <vector>
#include <map>
#include "L1Trigger/L1TMuonEndCap/interface/PtRefinementUnit.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class TGraph;

namespace L1TMuon {
  
  class CorridorPtRefinement: public PtRefinementUnit {    
  public:
    typedef std::unique_ptr<TGraph> pTGraph;
    typedef std::map<unsigned, std::map<unsigned, pTGraph > > 
      corridor_2stn_map;

    CorridorPtRefinement(const edm::ParameterSet&);
    ~CorridorPtRefinement() {}

    virtual void refinePt(const edm::EventSetup&, 
			  InternalTrack&) const;
  private:
    void get_corridors_from_file();
    double solveCorridor(double rawPtHypothesis, 
			 double observable,
			 const pTGraph &corridor_belt) const;
    edm::FileInPath _fcorridors;
    unsigned _N_PT_BINS;
    double _COR_PT_MAX;
    corridor_2stn_map _dphi_corridors;
    std::map<unsigned, std::unique_ptr<TGraph> > _phib_corridors;    
  };
}

#endif
