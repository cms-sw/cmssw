#ifndef Candidate_CompositeCandidate_h
#define Candidate_CompositeCandidate_h
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include <memory>
/** \class reco::CompositeCandidate
 *
 * A Candidate composed of daughters. 
 * The daughters are owned by the composite candidate.
 *
 * \author Luca Lista, INFN
 *
 *
 */

#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include <string>
#include <vector> 

namespace reco {

  class CompositeCandidate : public LeafCandidate {
  public:
    /// collection of daughters
    typedef CandidateCollection daughters;
    typedef std::vector<std::string> role_collection; 
    /// default constructor
    CompositeCandidate(std::string name="") : LeafCandidate(), name_(name) { }
    /// constructor from values
    template<typename P4>
    CompositeCandidate( Charge q, const P4 & p4, const Point & vtx = Point( 0, 0, 0 ),
			int pdgId = 0, int status = 0, bool integerCharge = true,
			std::string name="") :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ), name_(name) { }
   /// constructor from values
    explicit CompositeCandidate( const Candidate & p, const std::string& name="" );
    /// constructor from values
    explicit CompositeCandidate( const Candidate & p, const std::string& name, role_collection const & roles );
    /// destructor
    ~CompositeCandidate() override;
    /// get the name of the candidate
    std::string name() const { return name_;}
    /// set the name of the candidate
    void        setName(std::string name) { name_ = name;}
    /// get the roles
    role_collection const & roles() const { return roles_; }
    /// set the roles    
    void                    setRoles( const role_collection & roles ) { roles_.clear(); roles_ = roles; }
    /// returns a clone of the candidate
    CompositeCandidate * clone() const override;
    /// number of daughters
    size_type numberOfDaughters() const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    const Candidate * daughter( size_type ) const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    Candidate * daughter( size_type ) override;
    // Get candidate based on role
    Candidate *       daughter(const std::string& s ) override;
    const Candidate * daughter(const std::string& s ) const override;
    /// add a clone of the passed candidate as daughter 
    void addDaughter( const Candidate &, const std::string& s="" );
    /// add a clone of the passed candidate as daughter 
    void addDaughter( std::unique_ptr<Candidate>, const std::string& s="" );
    /// clear daughters
    void clearDaughters() { dau.clear(); }
    // clear roles
    void clearRoles() { roles_.clear(); }
    // Apply the roles to the objects
    void applyRoles();
    /// number of mothers (zero or one in most of but not all the cases)
    size_type numberOfMothers() const override;
    /// return pointer to mother
    const Candidate * mother( size_type i = 0 ) const override;

  private:
    /// collection of daughters
    daughters dau;
    /// check overlap with another daughter
    bool overlap( const Candidate & ) const override;
    /// candidate name
    std::string name_;
    /// candidate roles
    role_collection roles_;
  };

}

#endif
