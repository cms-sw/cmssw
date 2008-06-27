#ifndef GlobalFitJetCorrector_h
#define GlobalFitJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitCorrector.h"


/// classes declarations
namespace edm {
  class ParameterSet;
}

namespace reco {
  class Jet;
}

class GlobalFitCorrector;


class GlobalFitJetCorrector : public JetCorrector{
 public:
  GlobalFitJetCorrector(const edm::ParameterSet&);
  virtual ~GlobalFitJetCorrector();

  /// apply correction using Jet information only
  virtual double correction (const reco::Jet&) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  GlobalFitCorrector* corrector_;
};

#endif
