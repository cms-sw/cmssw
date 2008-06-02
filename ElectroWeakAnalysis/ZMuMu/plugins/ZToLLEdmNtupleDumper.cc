/* \class ZToLLEdmNtupleDumper
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <vector> 

class ZToLLEdmNtupleDumper : public edm::EDProducer {
public:
  ZToLLEdmNtupleDumper( const edm::ParameterSet & );
private:
  void produce( edm::Event &, const edm::EventSetup & );
  std::vector<std::string> zName_;
  std::vector<edm::InputTag> z_, zGenParticlesMatch_, isolations1_, isolations2_;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"

using namespace edm;
using namespace std;
using namespace reco;

ZToLLEdmNtupleDumper::ZToLLEdmNtupleDumper( const ParameterSet & cfg ) {
  string alias;
  vector<ParameterSet> psets = cfg.getParameter<vector<ParameterSet> > ( "zBlocks" );
  for( std::vector<edm::ParameterSet>::const_iterator i = psets.begin(); i != psets.end(); ++ i ) {
    string zName = i->getParameter<string>( "zName" );
    InputTag z =  i->getParameter<InputTag>( "z" );
    InputTag zGenParticlesMatch = i->getParameter<InputTag>( "zGenParticlesMatch" );
    InputTag isolations1 = i->getParameter<InputTag>( "isolations1" );
    InputTag isolations2 = i->getParameter<InputTag>( "isolations2" );
    zName_.push_back( zName );
    z_.push_back( z );
    zGenParticlesMatch_.push_back( zGenParticlesMatch );
    isolations1_.push_back( isolations1 );
    isolations2_.push_back( isolations2 );
    produces<vector<float> >( alias = zName + "Mass" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Y" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Iso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Iso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TrueMass" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TruePt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TrueEta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TruePhi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TrueY" ).setBranchAlias( alias );
  }
}

void ZToLLEdmNtupleDumper::produce( Event & evt, const EventSetup & ) {
  size_t size = z_.size();
  for( size_t c = 0; c < size; ++ c ) {
    Handle<CandidateCollection> zColl;
    evt.getByLabel( z_[c], zColl );
    Handle<CandMatchMap> zGenParticlesMatch;
    evt.getByLabel( zGenParticlesMatch_[c], zGenParticlesMatch );
    typedef AssociationVector<CandidateRefProd,vector<double> > IsolationCollection;
    Handle<IsolationCollection> isolations1, isolations2;
    evt.getByLabel( isolations1_[c], isolations1 );
    evt.getByLabel( isolations2_[c], isolations2 );
    size_t zSize = zColl->size();
    auto_ptr<vector<float> > zMass( new vector<float> );
    auto_ptr<vector<float> > zPt( new vector<float> );
    auto_ptr<vector<float> > zEta( new vector<float> );
    auto_ptr<vector<float> > zPhi( new vector<float> );
    auto_ptr<vector<float> > zY( new vector<float> );
    auto_ptr<vector<float> > zDau1Pt( new vector<float> );
    auto_ptr<vector<float> > zDau2Pt( new vector<float> );
    auto_ptr<vector<float> > zDau1Eta( new vector<float> );
    auto_ptr<vector<float> > zDau2Eta( new vector<float> );
    auto_ptr<vector<float> > zDau1Phi( new vector<float> );
    auto_ptr<vector<float> > zDau2Phi( new vector<float> );
    auto_ptr<vector<float> > zDau1Iso( new vector<float> );
    auto_ptr<vector<float> > zDau2Iso( new vector<float> );
    auto_ptr<vector<float> > trueZMass( new vector<float> );
    auto_ptr<vector<float> > trueZPt( new vector<float> );
    auto_ptr<vector<float> > trueZEta( new vector<float> );
    auto_ptr<vector<float> > trueZPhi( new vector<float> );
    auto_ptr<vector<float> > trueZY( new vector<float> );
    for( size_t i = 0; i < zSize; ++ i ) {
      const Candidate & z = (*zColl)[ i ];
      CandidateRef zRef( zColl, i );
      zMass->push_back( z.mass() );
      zPt->push_back( z.pt() );
      zEta->push_back( z.eta() );
      zPhi->push_back( z.phi() );
      zY->push_back( z.rapidity() );
      const Candidate * dau1 = z.daughter(0); 
      const Candidate * dau2 = z.daughter(1); 
      zDau1Pt->push_back( dau1->pt() );
      zDau2Pt->push_back( dau2->pt() );
      zDau1Eta->push_back( dau1->eta() );
      zDau2Eta->push_back( dau2->eta() );
      zDau1Phi->push_back( dau1->phi() );
      zDau2Phi->push_back( dau2->phi() );
      double iso1 = (*isolations1)[ dau1->masterClone().castTo<CandidateRef>() ];
      double iso2 = (*isolations2)[ dau2->masterClone().castTo<CandidateRef>() ];
      zDau1Iso->push_back( iso1 );
      zDau2Iso->push_back( iso2 );
      CandMatchMap::const_iterator trueZIter = zGenParticlesMatch->find( zRef );
      if ( trueZIter != zGenParticlesMatch->end() ) {
	CandidateRef trueZRef = trueZIter->val;
	if( trueZRef.isNonnull() ) {
	  const Candidate & z = * trueZRef;
	  trueZMass->push_back( z.mass() );
	  trueZPt->push_back( z.pt() );
	  trueZEta->push_back( z.eta() );      
	  trueZPhi->push_back( z.phi() );      
	  trueZY->push_back( z.rapidity() );      
	} else {
	  trueZMass->push_back( -100 );
	  trueZPt->push_back( -100 );
	  trueZEta->push_back( -100 );      
	  trueZPhi->push_back( -100 );      
	  trueZY->push_back( -100 );      
	}
      }
    }
    const string & zName = zName_[c];
    evt.put( zMass, zName +  "Mass" );
    evt.put( zPt, zName + "Pt" );
    evt.put( zEta, zName + "Eta" );
    evt.put( zPhi, zName + "Phi" );
    evt.put( zY, zName + "Y" );
    evt.put( zDau1Pt, zName + "Dau1Pt" );
    evt.put( zDau2Pt, zName + "Dau2Pt" );
    evt.put( zDau1Eta, zName + "Dau1Eta" );
    evt.put( zDau2Eta, zName + "Dau2Eta" );
    evt.put( zDau1Phi, zName + "Dau1Phi" );
    evt.put( zDau2Phi, zName + "Dau2Phi" );
    evt.put( zDau1Iso, zName + "Dau1Iso" );
    evt.put( zDau2Iso, zName + "Dau2Iso" );
    evt.put( trueZMass, zName +  "TrueMass" );
    evt.put( trueZPt, zName + "TruePt" );
    evt.put( trueZEta, zName + "TrueEta" );
    evt.put( trueZPhi, zName + "TruePhi" );
    evt.put( trueZY, zName + "TrueY" );
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZToLLEdmNtupleDumper );

