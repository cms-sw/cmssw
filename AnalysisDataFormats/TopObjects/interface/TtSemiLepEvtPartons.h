#ifndef TtSemiLepEvtPartons_h
#define TtSemiLepEvtPartons_h

#include "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"
#include "DataFormats/Candidate/interface/CandidateOnlyFwd.h"

#include <vector>

/**
   \class   TtSemiLepEvtPartons TtSemiLepEvtPartons.h "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

   \brief   Class to fill partons in a well defined order for semi-leptonic ttbar events

   This class is mainly used for the jet-parton matching in TopTools.
*/

class TtGenEvent;

class TtSemiLepEvtPartons : public TtEventPartons {
public:
  /// semi-leptonic parton enum used to define the order
  /// in the vector for lepton and jet combinatorics
  enum { LightQ, LightQBar, HadB, LepB, Lepton };

public:
  /// default constructor
  TtSemiLepEvtPartons(const std::vector<std::string>& partonsToIgnore = std::vector<std::string>());

  /// return vector of partons in the order defined in the corresponding enum
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt) const override;
};

#endif
