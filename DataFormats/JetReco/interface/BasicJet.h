#ifndef JetReco_BasicJet_h
#define JetReco_BasicJet_h

/** \class reco::BasicJet
 *
 * \short Jets made from CaloTowers
 *
 * BasicJet represents generic Jets witjout
 * any specific information
 * in addition to generic Jet parameters
 *
 * \author Fedor Ratnikov, UMd
 *
 ************************************************************/

#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {
  class BasicJet : public Jet {
  public:
    /** Default constructor*/
    BasicJet() {}

    /** Constructor from values*/
    BasicJet(const LorentzVector& fP4, const Point& fVertex);
    BasicJet(const LorentzVector& fP4, const Point& fVertex, const Jet::Constituents& fConstituents);

    ~BasicJet() override{};

    /// Polymorphic clone
    BasicJet* clone() const override;

    /// Print object
    std::string print() const override;

  private:
    /// Polymorphic overlap
    bool overlap(const Candidate&) const override;
  };
}  // namespace reco
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/BasicJetCollection.h"  //INCLUDECHECKER:SKIP
#endif
