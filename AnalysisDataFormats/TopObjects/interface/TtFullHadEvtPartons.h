#ifndef TtFullLepEvtPartons_h
#define TtFullLepEvtPartons_h

#include "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"

#include <vector>

/**
   \class   TtFullLepEvtPartons TtFullLepEvtPartons.h "AnalysisDataFormats/TopObjects/interface/TtFullLepEvtPartons.h"

   \brief   Class to fill partons in a well defined order for fully-leptonic ttbar events

   This class is mainly used for the jet-parton matching in TopTools.
*/

namespace reco { class Candidate; }
class TtGenEvent;

class TtFullHadEvtPartons : public TtEventPartons {

 public:

  /// fully-hadronic parton enum used to define the order 
  /// in the vector for lepton and jet combinatorics
  enum { LightQ, LightQBar, B, LightP, LightPBar, BBar};

 public:

  /// default constructor
  TtFullHadEvtPartons(const std::vector<std::string>& partonsToIgnore = std::vector<std::string>());
  /// default destructor
  ~TtFullHadEvtPartons(){};

  /// return vector of partons in the order defined in the corresponding enum
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt);

};

#endif
