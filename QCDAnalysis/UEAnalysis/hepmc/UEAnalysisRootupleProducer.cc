#include <iostream>
#include <list>

using std::cout;
using std::endl;

/// HepMC includes
#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "HepMC/IO_AsciiParticles.h"
#include "HepMC/SimpleVector.h"
#include "HepMC/ParticleDataTable.h"

/// HepPDT includes
/// see SimGeneral/HepPDTESSource/src/HepPDTESSource.cc
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"
#include "HepPDT/ParticleID.hh"

/// ROOT includes
#include "TFile.h"
#include "TMath.h"
#include "TH1.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TClonesArray.h"

/// FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/SISConePlugin.hh"

void fillMonteCarlo( std::list<HepMC::GenParticle*>& finalstateparticles,   // input collection
                     double                          _inputEtMin,           // threshold on input
                     TClonesArray&                   MonteCarlo );          // output TClonesArray

void fillChargedJet( std::list<HepMC::GenParticle*>& finalstateparticles,   // input collection
		     double                          _inputEtMin,           // threshold on input
		     fastjet::JetDefinition&         mJetDefinition,        // jet algorithm definition
		     fastjet::GhostedAreaSpec&       mActiveArea,           // area specifics
		     TClonesArray&                   ChargedJet );          // output TClonesArray

bool sortByPt(HepMC::GenParticle* particle1, HepMC::GenParticle* particle2)
{
  return particle1->momentum().perp() > particle2->momentum().perp();
}

bool sortPseudoJetsByPt(const fastjet::PseudoJet& particle1, const fastjet::PseudoJet& particle2)
{
  //  return particle1->perp() > particle2->perp();
  return particle1.perp() > particle2.perp();
}

// set cuts
const double CUTPTJET( 5. );

///_____________________________________________________________________
///
int main() { 

  // write histogramm out to file
  TFile* file = new TFile( "UEAnalysisRootuple.root", "RECREATE");

  file->mkdir("UEAnalysisRootpleOnlyMC500");
  file->mkdir("UEAnalysisRootpleOnlyMC900");
  file->mkdir("UEAnalysisRootpleOnlyMC1500");

  /// process type
  int    EventKind;
  double genEventScale;

  // save TClonesArrays of TLorentzVectors
  // i.e. store 4-vectors of particles and jets

  TClonesArray* MonteCarlo500  = new TClonesArray("TLorentzVector", 10000);
  TClonesArray* MonteCarlo900  = new TClonesArray("TLorentzVector", 10000);
  TClonesArray* MonteCarlo1500 = new TClonesArray("TLorentzVector", 10000);

  TClonesArray* ChargedJet500  = new TClonesArray("TLorentzVector", 10000);
  TClonesArray* ChargedJet900  = new TClonesArray("TLorentzVector", 10000);
  TClonesArray* ChargedJet1500 = new TClonesArray("TLorentzVector", 10000);

  file->cd("UEAnalysisRootpleOnlyMC500");
  TTree* AnalysisTree500 = new TTree("AnalysisTree","MBUE Analysis Tree");
  AnalysisTree500->Branch("EventKind",&EventKind,"EventKind/I");
  AnalysisTree500->Branch("genEventScale",&genEventScale,"genEventScale/D");  
  AnalysisTree500->Branch("MonteCarlo", "TClonesArray", &MonteCarlo500, 128000, 0);  
  AnalysisTree500->Branch("ChargedJet", "TClonesArray", &ChargedJet500, 128000, 0);

  file->cd("UEAnalysisRootpleOnlyMC900");
  TTree* AnalysisTree900 = new TTree("AnalysisTree","MBUE Analysis Tree");
  AnalysisTree900->Branch("EventKind",&EventKind,"EventKind/I");
  AnalysisTree900->Branch("genEventScale",&genEventScale,"genEventScale/D");  
  AnalysisTree900->Branch("MonteCarlo", "TClonesArray", &MonteCarlo900, 128000, 0);  
  AnalysisTree900->Branch("ChargedJet", "TClonesArray", &ChargedJet900, 128000, 0);

  file->cd("UEAnalysisRootpleOnlyMC1500");
  TTree* AnalysisTree1500 = new TTree("AnalysisTree","MBUE Analysis Tree");
  AnalysisTree1500->Branch("EventKind",&EventKind,"EventKind/I");
  AnalysisTree1500->Branch("genEventScale",&genEventScale,"genEventScale/D");
  AnalysisTree1500->Branch("MonteCarlo", "TClonesArray", &MonteCarlo1500, 128000, 0);
  AnalysisTree1500->Branch("ChargedJet", "TClonesArray", &ChargedJet1500, 128000, 0);

  // input file in HepMC format
  HepMC::IO_GenEvent ascii_in("infile",std::ios::in);

  // open output file for eye readable ascii output
  // begin scope of ascii_io 
  //  HepMC::IO_AsciiParticles ascii_io("readable_eventlist.dat",std::ios::out);
  
  /// define jet algorithm
  double coneRadius          ( 0.5   );
  double coneOverlapThreshold( 0.75  );
  int    maxPasses           ( 0     );
  double protojetPtMin       ( 0.    );
  bool   caching             ( false );
  fastjet::SISConePlugin::SplitMergeScale scale( fastjet::SISConePlugin::SM_pttilde );
  fastjet::SISConePlugin* mPlugin( new fastjet::SISConePlugin( coneRadius,
							       coneOverlapThreshold,
							       maxPasses,
							       protojetPtMin,
							       caching,
							       scale ) );
  fastjet::JetDefinition* mJetDefinition( new fastjet::JetDefinition (mPlugin) ); 

  /// do not calculate jet areas (default)
  double ghostEtaMax   ( 0. );
  int activeAreaRepeats( 0  );
  double ghostArea     ( 1. );
  fastjet::GhostedAreaSpec* mActiveArea = new fastjet::ActiveAreaSpec (ghostEtaMax, activeAreaRepeats, ghostArea);

  {
    int icount=0;

    // read in event in HepMC format
    HepMC::GenEvent* evt = ascii_in.read_next_event();

    while ( evt ) {
      icount++;
      //std::cout << "*************new event************* " << std::endl;

      //      if ( icount > 10000 ) break;

      if (icount==1 || icount%1000==0 ) {
	std::cout << "Processing Event Number " << icount
		  << " its # " << evt->event_number() << ", ";
	std::cout << "x1 = " << evt->pdf_info()->x1() << ", ";
	std::cout << "x2 = " << evt->pdf_info()->x2();
	std::cout << endl;

	// write the event out 
	//
	// to the human readable ascii file:
	// ascii_io << evt;
	//
	// or to stdout:
	// cout << evt;
      }

      //x1 = evt->pdf_info()->x1();
      //x2 = evt->pdf_info()->x2();

      //       HepMC::GenParticle* beamA( evt->beam_particles().first );
      //       HepMC::GenParticle* beamB( evt->beam_particles().first );
      //       cout << "beam particles are " << beamA->pdg_id() << " and " << beamB->pdg_id() << endl;

      EventKind = evt->signal_process_id();
      //cout << "event kind " << EventKind << ", event_scale () " << evt->event_scale() << endl;

      /// fill a list of all final state charged particles in the event (exclude neutrinos)
      std::list<HepMC::GenParticle*> finalstateparticles;
      for ( HepMC::GenEvent::particle_const_iterator p( evt->particles_begin() ), pEnd( evt->particles_end() ); 
	    p != pEnd; ++p )
	{
	  if ( (*p)->status()!=1  ) continue;
	  if ( (*p)->end_vertex() ) continue;
	  if ( (*p)->pdg_id()==12 ) continue;
	  if ( (*p)->pdg_id()==14 ) continue;
	  if ( (*p)->pdg_id()==16 ) continue;

 	  HepPDT::ParticleID* pid = new HepPDT::ParticleID( (*p)->pdg_id() );
  	  if ( pid->threeCharge()==0. ) continue;

	  /// particle is stable, does not come from end_vertex,
	  /// is charged,
	  /// and is no neutrino -> save it
	  finalstateparticles.push_back(*p);
	}
      finalstateparticles.sort( sortByPt );

      /// call jet algorithm
      double _inputEtMin( 0.5 );
      fillMonteCarlo( finalstateparticles,   // input collection
                      _inputEtMin,           // threshold on input
                      *MonteCarlo500 );      // output TClonesArray
      
      fillChargedJet( finalstateparticles,   // input collection
		      _inputEtMin,           // threshold on input
		      *mJetDefinition,       // jet algorithm definition
		      *mActiveArea,          // area specifics 
		      *ChargedJet500 );      // output TClonesArray 
      
      _inputEtMin = 0.9;
      fillMonteCarlo( finalstateparticles,   // input collection
		      _inputEtMin,           // threshold on input
		      *MonteCarlo900 );      // output TClonesArray

      fillChargedJet( finalstateparticles,   // input collection
		      _inputEtMin,           // threshold on input
		      *mJetDefinition,       // jet algorithm definition
		      *mActiveArea,          // area specifics 
		      *ChargedJet900 );      // output TClonesArray 

      _inputEtMin = 1.5;
      fillMonteCarlo( finalstateparticles,   // input collection
		      _inputEtMin,           // threshold on input
		      *MonteCarlo1500 );     // output TClonesArray

      fillChargedJet( finalstateparticles,   // input collection
		      _inputEtMin,           // threshold on input
		      *mJetDefinition,       // jet algorithm definition
		      *mActiveArea,          // area specifics 
		      *ChargedJet1500 );     // output TClonesArray 

      /// fill analysis tree
      AnalysisTree500 ->Fill();
      AnalysisTree900 ->Fill();
      AnalysisTree1500->Fill();

      /// clear memory from this event
      delete evt;
      /// read the next event
      ascii_in >> evt;
    }
    std::cout << icount << " events found. Finished." << std::endl;
  }

  file->Write();
  file->Close();
  return 0;
}

///_____________________________________________________________________
///
void fillChargedJet( std::list<HepMC::GenParticle*>& finalstateparticles,  // input collection
		     double                          _inputEtMin,          // threshold on input
		     fastjet::JetDefinition&         mJetDefinition,       // jet algorithm definition
		     fastjet::GhostedAreaSpec&       mActiveArea,          // area specifics
		     TClonesArray&                   ChargedJet )          // output TClonesArray
{ 
  ChargedJet.Clear();

  /// run fastjet finder on final state charged particles

  std::vector<fastjet::PseudoJet> fjInputs;
  fjInputs.reserve ( finalstateparticles.size() );
  
  int iJet( 0 );
  for ( std::list<HepMC::GenParticle*>::iterator p( finalstateparticles.begin() ),
	  pEnd( finalstateparticles.end() );
	p != pEnd; ++p )
    {
      if ( (*p)->momentum().perp() < _inputEtMin ) break;
      
      fjInputs.push_back (fastjet::PseudoJet ((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e()));
      fjInputs.back().set_user_index(iJet);
      ++iJet;
    }
  
  // here we need to keep both pointers, as "area" interfaces are missing in base class
  fastjet::ClusterSequenceActiveArea* clusterSequenceWithArea = 0;
  fastjet::ClusterSequenceWithArea* clusterSequence = 0;
  clusterSequenceWithArea = new fastjet::ClusterSequenceActiveArea (fjInputs, mJetDefinition, mActiveArea);
  clusterSequence = clusterSequenceWithArea;

  // retrieve jets for selected mode
  double mJetPtMin( 1. );
  std::vector<fastjet::PseudoJet> jets = clusterSequence->inclusive_jets (mJetPtMin);

  // get PU pt
  //      double median_Pt_Per_Area = clusterSequenceWithArea ? clusterSequenceWithArea->pt_per_unit_area() : 0.;

  std::sort (jets.begin(), jets.end(), sortPseudoJetsByPt );

  iJet = 0;
  int iSavedJet( 0 );

  // clear jet array from former entries
  ChargedJet.Clear();
  for (std::vector<fastjet::PseudoJet>::const_iterator jet=jets.begin(); jet!=jets.end();++jet , ++iJet)
    {
      if ( jet->perp() < CUTPTJET ) break; // do not save soft jets

      new((ChargedJet)[iSavedJet]) TLorentzVector(jet->px(), jet->py(), jet->pz(), jet->e());
      ++iSavedJet;
    }

  // cleanup
  if (clusterSequenceWithArea) delete clusterSequenceWithArea;
  else delete clusterSequence;
}


///_____________________________________________________________________
///
void fillMonteCarlo( std::list<HepMC::GenParticle*>& finalstateparticles,   // input collection
		     double                          _inputEtMin,           // threshold on input
		     TClonesArray&                   MonteCarlo )           // output TClonesArray
{
  MonteCarlo.Clear();
  std::list<HepMC::GenParticle*>::iterator it( finalstateparticles.begin()), itEnd( finalstateparticles.end() );
  for( int iMonteCarlo(0); it != itEnd; ++it, ++iMonteCarlo )
    {
      if ( (*it)->momentum().perp() < _inputEtMin ) break;
      
      new((MonteCarlo)[iMonteCarlo]) TLorentzVector( (*it)->momentum().px(),
						     (*it)->momentum().py(),
						     (*it)->momentum().pz(),
						     (*it)->momentum().e() );
    }
}
