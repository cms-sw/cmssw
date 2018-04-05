#ifndef Candidate_NamedCompositeCandidate_H
#define Candidate_NamedCompositeCandidate_H
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include <memory>
/** \class reco::NamedCompositeCandidate
 *
 * A Candidate composed of daughters. 
 * The daughters are owned by the composite candidate.
 *
 * \author Luca Lista, INFN
 *
 *
 */

#include "DataFormats/Candidate/interface/NamedCompositeCandidateFwd.h"
#include <string>
#include <map>

namespace reco {

  class NamedCompositeCandidate : public CompositeCandidate {
  public:
    typedef std::vector<std::string>                   role_collection;

    /// default constructor
    NamedCompositeCandidate(std::string name="") : CompositeCandidate(), name_(name) { }
    NamedCompositeCandidate(std::string name,
			    const role_collection & roles  ) : 
      CompositeCandidate(), name_(name), roles_(roles) { }
    /// constructor from values
    NamedCompositeCandidate( std::string name, 
			     const role_collection & roles,
			     Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			     int pdgId = 0, int status = 0, bool integerCharge = true ) :
      CompositeCandidate( q, p4, vtx, pdgId, status, integerCharge ), 
      name_ (name), roles_(roles) { }
    /// constructor from values
    NamedCompositeCandidate( std::string name, 
			     const role_collection & roles, 
			     const Candidate & p );
 
    /// destructor
    ~NamedCompositeCandidate() override;
    /// returns a clone of the candidate
    NamedCompositeCandidate * clone() const override;
    // get name
    std::string             name() const { return name_; }
    // set name
    void                    setName( std::string n ) { name_ = n; }
    // get roles
    const NamedCompositeCandidate::role_collection & roles() const { return roles_;}
    // set roles
    void                    setRoles( const NamedCompositeCandidate::role_collection & roles ) { roles_.clear(); roles_ = roles; }
    // Get candidate based on role
    Candidate *       daughter(const std::string& s ) override;
    const Candidate * daughter(const std::string& s ) const override;
    // Get candidate based on index
    Candidate *       daughter( size_type i ) override { return CompositeCandidate::daughter(i); }
    const Candidate * daughter( size_type i ) const override  { return CompositeCandidate::daughter(i); }
    // Add daughters
    void                    addDaughter( const Candidate &, const std::string&s );
    void                    addDaughter( std::unique_ptr<Candidate>, const std::string& s );
    // Clear daughters and roles
    void                    clearDaughters() { CompositeCandidate::clearDaughters(); }
    void                    clearRoles() { roles_.clear(); }
    // Apply the roles to the objects
    void                    applyRoles();
  private:
    std::string      name_;
    role_collection  roles_;
  };

}

#endif
