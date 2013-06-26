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
 * \version $Id: CompositeCandidate.h,v 1.30 2012/10/14 07:31:36 innocent Exp $
 *
 */

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"
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
    virtual ~CompositeCandidate();
    /// get the name of the candidate
    std::string name() const { return name_;}
    /// set the name of the candidate
    void        setName(std::string name) { name_ = name;}
    /// get the roles
    role_collection const & roles() const { return roles_; }
    /// set the roles    
    void                    setRoles( const role_collection & roles ) { roles_.clear(); roles_ = roles; }
    /// returns a clone of the candidate
    virtual CompositeCandidate * clone() const;
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter const_iterator
    virtual iterator end();
    /// number of daughters
    virtual size_type numberOfDaughters() const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type );
    // Get candidate based on role
    virtual Candidate *       daughter(const std::string& s );
    virtual const Candidate * daughter(const std::string& s ) const;
    /// add a clone of the passed candidate as daughter 
    void addDaughter( const Candidate &, const std::string& s="" );
    /// add a clone of the passed candidate as daughter 
    void addDaughter( std::auto_ptr<Candidate>, const std::string& s="" );
    /// clear daughters
    void clearDaughters() { dau.clear(); }
    // clear roles
    void clearRoles() { roles_.clear(); }
    // Apply the roles to the objects
    void applyRoles();
    /// number of mothers (zero or one in most of but not all the cases)
    virtual size_type numberOfMothers() const;
    /// return pointer to mother
    virtual const Candidate * mother( size_type i = 0 ) const;

  private:
    // const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    // iterator implementation
    typedef candidate::iterator_imp_specific<daughters> iterator_imp_specific;
    /// collection of daughters
    daughters dau;
    /// check overlap with another daughter
    virtual bool overlap( const Candidate & ) const;
    /// candidate name
    std::string name_;
    /// candidate roles
    role_collection roles_;
  };

}

#endif
