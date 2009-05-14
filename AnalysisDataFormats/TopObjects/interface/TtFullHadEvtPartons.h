#ifndef TtFullHadEvtPartons_h
#define TtFullHadEvtPartons_h

#include <vector>

/**
   \class   TtFullHadEvtPartons TtFullHadEvtPartons.h "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

   \brief   Class to fill partons in a well defined order for fully-hadronic ttbar events

   This class is mainly used for the jet-parton matching in TopTools.
*/

namespace reco { class Candidate; }
class TtGenEvent;

class TtFullHadEvtPartons {

 public:

  /// fully-hadronic parton enum used to define the order 
  /// in the vector for lepton and jet combinatorics
  enum { LightQTop, LightQBarTop, B, LightQTopBar, LightQBarTopBar, BBar};

 public:

  /// empty constructor
  TtFullHadEvtPartons(){};
  /// default destructor
  ~TtFullHadEvtPartons(){};

  /// return vector of partons in the order defined in the corresponding enum
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt);
};

#endif
