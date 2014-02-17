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
 * \version $Id: NamedCompositeCandidate.h,v 1.5 2008/12/05 12:15:18 hegner Exp $
 *
 */

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"
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
    virtual ~NamedCompositeCandidate();
    /// returns a clone of the candidate
    virtual NamedCompositeCandidate * clone() const;
    // get name
    std::string             name() const { return name_; }
    // set name
    void                    setName( std::string n ) { name_ = n; }
    // get roles
    const NamedCompositeCandidate::role_collection & roles() const { return roles_;}
    // set roles
    void                    setRoles( const NamedCompositeCandidate::role_collection & roles ) { roles_.clear(); roles_ = roles; }
    // Get candidate based on role
    virtual Candidate *       daughter(const std::string& s );
    virtual const Candidate * daughter(const std::string& s ) const;
    // Get candidate based on index
    virtual Candidate *       daughter( size_type i ) { return CompositeCandidate::daughter(i); }
    virtual const Candidate * daughter( size_type i ) const  { return CompositeCandidate::daughter(i); }
    // Add daughters
    void                    addDaughter( const Candidate &, const std::string&s );
    void                    addDaughter( std::auto_ptr<Candidate>, const std::string& s );
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
