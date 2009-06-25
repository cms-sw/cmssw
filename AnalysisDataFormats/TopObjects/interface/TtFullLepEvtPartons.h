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

class TtFullLepEvtPartons : public TtEventPartons {

 public:

  /// fully-leptonic parton enum used to define the order 
  /// in the vector for lepton and jet combinatorics
  enum { B, BBar, Lepton, LeptonBar };

 public:

  /// default constructor
  TtFullLepEvtPartons(const std::vector<std::string>& partonsToIgnore = std::vector<std::string>());
  /// default destructor
  ~TtFullLepEvtPartons(){};

  /// return vector of partons in the order defined in the corresponding enum
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt);

};

#endif
