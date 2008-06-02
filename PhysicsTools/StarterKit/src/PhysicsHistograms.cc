//------------------------------------------------------------------------
// -*- C++ -*-
//
//! \class PhysicsHistograms PhysicsHistograms.cc Demo/TempAnaToolkit/src/PhysicsHistograms.cc
//!
//!  Description: Demonstration of a simple analysis toolkit for starter analyses
//!
//
// Original Author:  Petar Maksimovic
//         Created:  Christmas 2007
// $Id: PhysicsHistograms.cc,v 1.3 2008/02/22 22:21:37 srappocc Exp $
//
// Revision History:
//------------------------------------------------------------------------


#include "PhysicsTools/StarterKit/interface/PhysicsHistograms.h"

#include <string>
#include <sstream>

//------------------------------------------------------------------------
//!  Create the objects that manage the histogram groups.
//------------------------------------------------------------------------
PhysicsHistograms::PhysicsHistograms()
{
  //--- Initialize histogram objects
  std::cout << "PhysicsHistograms: Creating muon histograms" << std::endl;
  muonHistograms_     = new pat::HistoMuon    ();
  std::cout << "PhysicsHistograms: Creating electron histograms" << std::endl;
  electronHistograms_ = new pat::HistoElectron();
  std::cout << "PhysicsHistograms: Creating tau histograms" << std::endl;
  tauHistograms_      = new pat::HistoTau     ();
  std::cout << "PhysicsHistograms: Creating jet histograms" << std::endl;
  jetHistograms_      = new pat::HistoJet     ();
  std::cout << "PhysicsHistograms: Creating met histograms" << std::endl;
  metHistograms_      = new pat::HistoMET     ();
  std::cout << "PhysicsHistograms: Creating photon histograms" << std::endl;
  photonHistograms_   = new pat::HistoPhoton  ();
  std::cout << "PhysicsHistograms: Creating track histograms" << std::endl;
  trackHistograms_    = new pat::HistoTrack   ();

  //--- Output file (still unused?)
  outputTextName_ = "blahblah.txt";  // &&& need to decide what to do with this.
  outputFile_.open( outputTextName_.c_str() );
}


//------------------------------------------------------------------------
//!  Destroy the objects that manage the histogram groups.
//!
//!  Note that the TH1's used by PhysVarHistos managed by these histo
//!  groups will *not* be deleted in the PhysVarHisto's destructor. So
//!  it's safe to delete both HistoGroups and PhysVarHistos.
//------------------------------------------------------------------------
PhysicsHistograms::~PhysicsHistograms()
{
  delete muonHistograms_     ;
  delete electronHistograms_ ;
  delete tauHistograms_      ;
  delete jetHistograms_      ;
  delete metHistograms_      ;
  delete photonHistograms_   ;
  delete trackHistograms_    ;

  outputFile_.close();
}



//------------------------------------------------------------------------
//!  Methods to configure (enable or disable) various PhysVarHistos one at the time,
//!  or in whole groups.
//------------------------------------------------------------------------
void
PhysicsHistograms::configure( std::string & histos_to_disable,   // comma separated list of names
			      std::string & histos_to_enable )   // comma separated list of names
{
  std::cout << "PhysicsHistograms:: configuring..."
	    << "\n   First disabling: " << histos_to_disable
	    << "\n   Then  enabling : " << histos_to_enable
	    << std::endl;


  //--- Pass this information to histogramGroups
  muonHistograms_    ->configure( histos_to_disable, histos_to_enable ) ;
  electronHistograms_->configure( histos_to_disable, histos_to_enable ) ;
  tauHistograms_     ->configure( histos_to_disable, histos_to_enable ) ;
  metHistograms_     ->configure( histos_to_disable, histos_to_enable ) ;
  jetHistograms_     ->configure( histos_to_disable, histos_to_enable ) ;
  photonHistograms_  ->configure( histos_to_disable, histos_to_enable ) ;
  trackHistograms_   ->configure( histos_to_disable, histos_to_enable ) ;

}


//------------------------------------------------------------------------
//!  Selection of a subset of PhysVarHistos.
//------------------------------------------------------------------------
void
PhysicsHistograms::select( std::string  vars_to_select,   // comma separated list of names
			   std::vector< pat::PhysVarHisto * > & selectedVars )
{
  std::cout << "PhysicsHistograms:: selecting the following variables:\n\t"
	    << vars_to_select
	    << std::endl;


  //--- Pass this information to histogramGroups
  muonHistograms_    ->select( vars_to_select, selectedVars ) ;
  electronHistograms_->select( vars_to_select, selectedVars ) ;
  tauHistograms_     ->select( vars_to_select, selectedVars ) ;
  metHistograms_     ->select( vars_to_select, selectedVars ) ;
  jetHistograms_     ->select( vars_to_select, selectedVars ) ;
  photonHistograms_  ->select( vars_to_select, selectedVars ) ;
  trackHistograms_   ->select( vars_to_select, selectedVars ) ;
  
  std::vector<pat::PhysVarHisto*>::iterator i = allVarHistos_.begin();
  std::vector<pat::PhysVarHisto*>::iterator end = allVarHistos_.end();
  std::string temp = "," + vars_to_select + ",";
  for ( ; i != end; ++i  ) {
    std::string test = "," + (*i)->name() + ",";
    std::cout << "testing " << test << std::endl;
    if ( temp.find( test ) != std::string::npos || temp == ",all," ) {
      std::cout << "FOUND!" << std::endl;
      selectedVars.push_back ( *i );
    }
  }

  std::cout << "PhysicsHistograms:: selected " << selectedVars.size()
	    << " variables." << std::endl;
}

void PhysicsHistograms::clearVec()
{
  muonHistograms_    ->clearVec() ;
  electronHistograms_->clearVec() ;
  tauHistograms_     ->clearVec() ;
  metHistograms_     ->clearVec() ;
  jetHistograms_     ->clearVec() ;
  photonHistograms_  ->clearVec() ;
  trackHistograms_   ->clearVec() ;
  for ( uint i = 0; i < allVarHistos_.size(); i++ ) {
    allVarHistos_[i]->clearVec();
  }
}

//------------------------------------------------------------------------
//!  Method called before seeing all events (in CMSSW) or before the
//!  event loop in FWLite.
//------------------------------------------------------------------------
void
PhysicsHistograms::beginJob()
{
  // Dummy for now
}


//------------------------------------------------------------------------
//!  Method called after seeing all events (in CMSSW) or after the
//!  event loop in FWLite.
//------------------------------------------------------------------------
void
PhysicsHistograms::endJob()
{
  // Dummy for now
}



//------------------------------------------------------------------------
//!  Method to print out reco::Candidates.
// &&& Design suggestion: this should go into HistoGroup<> instead.
//------------------------------------------------------------------------
std::ostream & operator<<( std::ostream & out, const reco::Candidate & cand )
{
  char buff[1000];
  sprintf( buff, "Pt, Eta, Phi, M = (%6.2f, %6.2f, %6.2f, %6.2f)",
           cand.pt(), cand.eta(), cand.phi(), cand.mass() );
  out << buff;
  return out;
}



//--- Code graveyard:
//
//   // Muons
//   typedef std::vector<Muon>::const_iterator muonIter ;
//   outputFile_ << "Muons: " << muons.size() << endl;
//   int imuon = 0;
//   for (muonIter muon = muons.begin(); muon != muons.end();++muon, ++imuon) {
//     outputFile_ << setw(6) << imuon << " : " << *muon << endl;
//     histoMuon_->fill( *muon, imuon+1 );
//   }


