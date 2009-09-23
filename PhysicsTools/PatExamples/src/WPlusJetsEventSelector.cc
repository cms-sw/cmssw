
#include "PhysicsTools/PatExamples/interface/WPlusJetsEventSelector.h"

#include <iostream>

using namespace std;

WPlusJetsEventSelector::WPlusJetsEventSelector( 
    edm::InputTag const & muSrc, 
    edm::InputTag const & eleSrc,
    edm::InputTag const & jetSrc,
    edm::InputTag const & metSrc,
    boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor> & muonIdTight,
    boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor> & electronIdTight,
    boost::shared_ptr<JetIDSelectionFunctor> & jetIdTight,
    boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor> & muonIdLoose,
    boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor> & electronIdLoose,
    boost::shared_ptr<JetIDSelectionFunctor> & jetIdLoose,
    double muPtMin  ,
    double elePtMin ,
    double jetPtMin
						) :
  muSrc_( muSrc ), eleSrc_(eleSrc), jetSrc_(jetSrc), metSrc_(metSrc),
  muonIdTight_(muonIdTight),
  electronIdTight_(electronIdTight),
  jetIdTight_(jetIdTight),
  muonIdLoose_(muonIdLoose),
  electronIdLoose_(electronIdLoose),
  jetIdLoose_(jetIdLoose),
  muPtMin_(muPtMin),
  elePtMin_(elePtMin),
  jetPtMin_(jetPtMin)
{
  // make the bitset
  push_back( "Inclusive"      );
  push_back( "Trigger"        );
  push_back( ">= 1 Lepton"    );
  push_back( "== 1 Lepton"    );
  push_back( ">= 1 Tight Jet" );
  push_back( "MET > 20"       );
  push_back( "Z Veto"         );
  push_back( "Conversion Veto");
  push_back( "Cosmic Veto"    );
}

bool WPlusJetsEventSelector::operator() (edm::EventBase const & t, std::strbitset & ret)
{
  selectedJets_.clear();
  selectedMuons_.clear();
  selectedElectrons_.clear();
  selectedMETs_.clear();

  passCut( ret, "Inclusive");
  edm::Handle< std::vector<pat::Jet> >       allJets;
  edm::Handle< std::vector<pat::Muon> >      allMuons;
  edm::Handle< std::vector<pat::Electron> >  allElectrons;
  edm::Handle< std::vector<pat::MET> >       allMETs;

  bool foundJets = t.getByLabel<std::vector<pat::Jet> >( jetSrc_, allJets);
  bool foundMuons = t.getByLabel<std::vector<pat::Muon> >( muSrc_, allMuons);
  bool foundElectrons = t.getByLabel<std::vector<pat::Electron> >( eleSrc_, allElectrons);
  bool foundMET = t.getByLabel<std::vector<pat::MET> >( metSrc_, allMETs);

  if ( !foundJets || !foundMuons || !foundElectrons || !foundMET ||
       !allJets.isValid() || !allMuons.isValid() || !allElectrons.isValid() || !allMETs.isValid() ) {
    std::cout << "unable to find collections" << std::endl;
    return false;
  }

  for ( std::vector<pat::Muon>::const_iterator muonBegin = allMuons->begin(),
	  muonEnd = allMuons->end(), imuon = muonBegin;
	imuon != muonEnd; ++imuon ) {
    std::strbitset iret = muonIdTight_->getBitTemplate();
    if ( imuon->pt() > muPtMin_ && (*muonIdTight_)(*imuon, iret) ) {
      selectedMuons_.push_back( *imuon );
    }
  }

  for ( std::vector<pat::Electron>::const_iterator electronBegin = allElectrons->begin(),
	  electronEnd = allElectrons->end(), ielectron = electronBegin;
	ielectron != electronEnd; ++ielectron ) {
    std::strbitset iret = electronIdTight_->getBitTemplate();
    if ( ielectron->pt() > elePtMin_ && (*electronIdTight_)(*ielectron, iret) ) {
      selectedElectrons_.push_back( *ielectron );
    }
  }

  for ( std::vector<pat::Jet>::const_iterator jetBegin = allJets->begin(),
	  jetEnd = allJets->end(), ijet = jetBegin;
	ijet != jetEnd; ++ijet ) {
    std::strbitset iret = jetIdTight_->getBitTemplate();
    if ( ijet->pt() > jetPtMin_ && (*jetIdTight_)(*ijet, iret) ) {
      selectedJets_.push_back( *ijet );
    }
  }

  if ( true ) passCut(ret, "Trigger");

  if ( (*this)[">= 1 Lepton"] || 
       ( selectedMuons_.size() > 0 || selectedElectrons_.size() > 0 ) ){
    passCut( ret, ">= 1 Lepton");
  }

  if ( (*this)["== 1 Lepton"] || 
       ( selectedMuons_.size() + selectedElectrons_.size() == 1 ) ){
    passCut(ret, "== 1 Lepton");
  }

  if ( (*this)[">= 1 Tight Jet"] ||
       selectedJets_.size() > 0 ){
    passCut(ret,">= 1 Tight Jet");
  }

  bool metCut = true;
  if ( (*this)["MET > 20"] ||
       metCut ) {
    passCut( ret, "MET > 20" );
  }

  bool zVeto = true;
  if ( selectedMuons_.size() == 2 ) {
  }
  if ( selectedElectrons_.size() == 2 ) {
  }
  if ( (*this)["Z Veto"] ||
       zVeto ){
    passCut(ret, "Z Veto");
  }
  
  bool conversionVeto = true;
  if ( (*this)["Conversion Veto"] ||
       conversionVeto ) {
    passCut(ret,"Conversion Veto");
  }


  bool cosmicVeto = true;
  if ( (*this)["Cosmic Veto"] ||
       cosmicVeto ) {
    passCut(ret,"Cosmic Veto");
  }

  return (bool)ret;
}
