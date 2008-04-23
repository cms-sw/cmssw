#ifndef StarterKit_HistoGroup_h
#define StarterKit_HistoGroup_h

//------------------------------------------------------------
// Title: HistoGroup.h
// Purpose: To histogram data objects deriving from PATObject
//          that is common (such as 4-vector information)
//
// Authors:
// Liz Sexton-Kennedy <sexton@fnal.gov>
// Eric Vaandering <ewv@fnal.gov >
// Petar Maksimovic <petar@jhu.edu>
// Sal Rappoccio <rappocc@fnal.gov>
//------------------------------------------------------------
//
// Interface:
//
//   HistoGroup ( string prepend, TFile * file );
//   Description: Constructor. Creates histograms and prepends
//                desired prefix. Stores file where histograms
//                should reside.
//
//   void fill( PATObject * );
//   Description: Fill object. Will fill relevant reco::Candidate
//                variables.
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoGroup
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------

// This package's include files
#include "PhysicsTools/StarterKit/interface/PhysVarHisto.h"

// CMSSW include files
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Candidate/interface/CompositeRefCandidateT.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TMath.h>

namespace pat {

  //! PHYS_OBJECT template argument must inherit from a reco::Particle
  template <class PHYS_OBJECT>
  class HistoGroup {

  public:

    HistoGroup( std::string dir = "cand", std::string groupName = "Candidate", std::string groupLabel = "cand",
		double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoGroup();

    //!  Fill all histograms for one Physics Object
    virtual void fill( const PHYS_OBJECT     * obj, uint imulti = 1, double weight = 1.0);
    virtual void fill( const PHYS_OBJECT     & obj, uint imulti = 1, double weight = 1.0)
    { fill( &obj, imulti, weight ); } // call the one above

    //!  Fill all histograms for *all* Phys Objects sitting in a collection!
    virtual void fillCollection( const std::vector<PHYS_OBJECT> & coll );

    //!  Register newly created HistGram objects
    virtual void addHisto( PhysVarHisto * hg );

    //!  Allow the user to add further histograms which do not exist
    //!  in the default histo maker.
    virtual void addHistoGroup( HistoGroup<PHYS_OBJECT> * hist_obj );

    //!  Direct access to the histograms owned by this HistoGroup.
    //!  Note that in addHistoGroup(), the histograms of the added HistoGroup
    //!  are added to histograms_ vector, which is thus the superset of all
    //!  histograms under the control of this HistoGroup.
    //
    inline std::vector< PhysVarHisto * > & histograms() { return histograms_ ; }


    //!  Configure this HistoGroup: enable or disable using the input comma-separated
    //!  lists of histograms.
    //
    virtual void configure( std::string  vars_to_disable,   // comma separated list of names
			    std::string  vars_to_enable );


    //!  A more generic method which allows access to a subset of PhysVarHistos.
    //
    virtual void select( std::string  vars_to_select,   // comma separated list of names
			 std::vector< PhysVarHisto * > & selectedHistos );


    //! THIS IS UGLY! Need to change this to the more elegant solution ASAP, this
    //! is a temporary solution
    virtual void setNBins    ( int nBins )              { nBins_ = nBins; }
    virtual void setPtRange  ( double pt1, double pt2 ) { pt1_ = pt1; pt2_ = pt2; }
    virtual void setMassRange( double m1,  double m2 )  { m1_ = m1; m2_ = m2; }

    // Reset vectors for ntuple caching
    virtual void clearVec();

  protected:
    edm::Service<TFileService> fs;

    std::string      prepend_ ;   // &&& should be groupLabel_ or something
    std::string      dir_ ;
    std::string      groupName_;

    int              verboseLevel_;

    TFileDirectory * currDir_;

    //! User-defined list of HistoGroups.  May be empty.
    //! These histo groups are new'ed by the user, but then given
    //! in ownership to this histo group.
    std::vector< HistoGroup<PHYS_OBJECT> * > histoGroups_ ;


    //! All PhysVarHisto objects owned by this HistoGroup
    std::vector< PhysVarHisto * > histograms_ ;

    //! Histogram axes: TEMPORARY SOLUTION
    int    nBins_;
    double pt1_, pt2_, m1_, m2_;

    //! Kinematic information
    PhysVarHisto * h_size_ ;   //!< the size of the collection
    PhysVarHisto * h_pt_   ;   //!< pt of each object
    PhysVarHisto * h_eta_  ;   //!< eta of each object
    PhysVarHisto * h_phi_  ;   //!< phi of each object
    PhysVarHisto * h_mass_ ;   //!< invariant mass of each object (useless for non-composite???)

    // &&& Design worry: if we add another group to this group,
    // &&&               will we end up with a duplicate of the above?
  };


  //------------------------------------------------------------------------
  //  The inline implementation.
  //------------------------------------------------------------------------


  //------------------------------------------------------------------------
  //!  Constructor, destructor
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  HistoGroup<PHYS_OBJECT>::
    HistoGroup( std::string dir, std::string groupName, std::string groupLabel,
		double pt1, double pt2, double m1, double m2)
    :
    prepend_(groupLabel),   //!<   What's used in histo names
    dir_ (dir),
    groupName_(groupName),  //!<   What's used in histo titles
    verboseLevel_(0),       //! verbosity: turned off by hand
    currDir_(0),
    nBins_(20), pt1_(pt1), pt2_(pt2), m1_(m1), m2_(m2),
    h_size_(0),
    h_pt_(0), h_eta_(0), h_phi_(0), h_mass_(0)
  {
    if ( verboseLevel_ > 10) {
      std::cout << "HistoGroup(" << dir_ << "/" << prepend_ << ")::in constructor"
		<< std::endl;
    }
    currDir_ = new TFileDirectory( fs->mkdir(dir_) );

    std::string name, title;

    name  = prepend_+"CollSize"; title = "Number of "+groupName_+"s";
    addHisto( h_size_ = new PhysVarHisto( name.c_str(), title.c_str(),  nBins_, -0.5, nBins_ + 0.5, currDir_, "", "I" ) );

    name  = prepend_+"Pt";  title = groupName_+"  p_{T};p_{T} (GeV/c)";
    addHisto( h_pt_   = new PhysVarHisto( name.c_str(), title.c_str(),  nBins_, pt1_, pt2_, currDir_, "", "vD") );

    name  = prepend_+"Eta"; title = groupName_+" #eta;#eta";
    addHisto( h_eta_  = new PhysVarHisto( name.c_str(), title.c_str(), nBins_, -3.0, 3.0, currDir_, "", "vD") );

    name  = prepend_+"Phi"; title = groupName_+" #phi;#phi";
    addHisto( h_phi_  = new PhysVarHisto( name.c_str(), title.c_str(), nBins_, -TMath::Pi(), TMath::Pi(), currDir_, "", "vD") );

    name  = prepend_+"Mass";   title = groupName_+" invariant mass;Mass (GeV/c^{2})";
    addHisto( h_mass_ = new PhysVarHisto( name.c_str(), title.c_str(), nBins_, m1_, m2_, currDir_, "", "vD") );

  }


  //------------------------------------------------------------------------
  //!  Destructor.  Note: the histograms are owned by TFileDirectory, so
  //!  PhysVarHistos won't delete them.  Thus it's safe to delete the
  //!  PhysVarHisto objects we manage.
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  HistoGroup<PHYS_OBJECT>::
  ~HistoGroup()
  {
    typename std::vector< HistoGroup<PHYS_OBJECT> * >::iterator
      hm    = histoGroups_.begin(),
      hmend = histoGroups_.end();
    for ( ; hm != hmend; ++hm ) {
      delete (*hm);             // delete the actual object
    }
    histoGroups_.erase( hm, hmend );    // free memory used by this small array
  }



  //------------------------------------------------------------------------
  //!  Fill the reco::Cand basic histograms: pt, eta, phi.
  //!  Plus also invoke the user-defined classes.
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  void
  HistoGroup<PHYS_OBJECT>::
  fill( const PHYS_OBJECT     * obj, uint imulti, double weight)
  {
    if ( verboseLevel_ > 10) {
      std::cout << "HistoGroup(" << dir_ << "/" << prepend_ << ")::fill: imulti = " << imulti
		<< std::endl;
    }
    h_pt_  ->fill( obj->p4().pt()  , imulti, weight );
    h_eta_ ->fill( obj->p4().eta() , imulti, weight );
    h_phi_ ->fill( obj->p4().phi() , imulti, weight );
    h_mass_->fill( obj->p4().mass(), imulti, weight );

    //--- Now fill the user-defined histograms
    typename std::vector< HistoGroup<PHYS_OBJECT> * >::const_iterator
      hg    = histoGroups_.begin(),
      hgend = histoGroups_.end();
    for ( ; hg != hgend; ++hg ) {
      (*hg)->fill( obj, imulti, weight );
    }
  }


  //------------------------------------------------------------------------
  //!  Fill all histograms for *all* Phys Objects sitting in a collection!
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  void
  HistoGroup<PHYS_OBJECT>::
  fillCollection( const std::vector<PHYS_OBJECT> & coll )
  {

    if ( verboseLevel_ > 10) {
      std::cout << "HistoGroup(" << dir_ << "/" << prepend_ << ")::fillCollection"
		<< std::endl;
    }
    h_size_->fill( coll.size() );     //! Save the size of the collection.

    typename std::vector<PHYS_OBJECT>::const_iterator
      iobj = coll.begin(),
      iend = coll.end();

    uint i = 1;              //! Fortran-style indexing
    for ( ; iobj != iend; ++iobj, ++i ) {
      fill( &*iobj, i);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
    }
  }




  //------------------------------------------------------------------------
  //!  Allow the user to add further histograms which do not exist
  //!  in the default histo maker.
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  void
  HistoGroup<PHYS_OBJECT>::
  addHistoGroup( HistoGroup<PHYS_OBJECT> * another_hist_group )
  {
    if (! another_hist_group ) return;  // &&& complain?
    histoGroups_.push_back( another_hist_group );

    //!  Now append the histograms owned by another_hist_group to our list
    std::vector< PhysVarHisto * >::iterator
      h    = another_hist_group->histograms_.begin(),
      hend = another_hist_group->histograms_.end();

    for ( ; h != hend; ++h ) {
      histograms_.push_back( *h );       // (*h) dereferences to a PhysVarHisto*
    }
  }



  //------------------------------------------------------------------------
  //!  Allow the user to add further histograms which do not exist
  //!  in the default histo maker.
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  void
  HistoGroup<PHYS_OBJECT>::
  addHisto( PhysVarHisto * hg )
  {

    if ( verboseLevel_ > 10)
      std::cout << "HistoGroup(" << dir_ << "/" << prepend_ << ")::addHisto("
		<< hg->name() << ")." << std::endl;

    if (! hg) return;
    histograms_.push_back( hg );
    hg->setTFileDirectory( currDir_ );
  }




  //------------------------------------------------------------------------
  //!  Configure: enable or disable.
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  void
  HistoGroup<PHYS_OBJECT>::
  configure( std::string  vars_to_disable,   // comma separated list of names
	     std::string  vars_to_enable
	     // , std::string  vars_to_ntuplize
	     )
  {
    //  We can manipulate these strings since we passed them whole and
    //  thus are dealing with local copies of the original arguments.
    vars_to_disable   = "," + vars_to_disable  + ",";
    vars_to_enable    = "," + vars_to_enable   + ",";
    // vars_to_ntuplize  = "," + vars_to_ntuplize + ",";

    //  Iterate over PhysVarHistos
    //
    std::vector< PhysVarHisto * >::iterator
      h    = histograms_.begin(),
      hend = histograms_.end();

    for ( ; h != hend; ++h ) {
      // (*h) dereferences to a PhysVarHisto*
      std::string test_name = "," + (*h)->name() + ",";

      if ( verboseLevel_ > 10)
	std::cout << "HistoGroup::configure (debug): test_name = " << test_name << std::endl;

      // is test_name a part of vars_to_disable?
      std::string::size_type loc = vars_to_disable.find( test_name, 0 );
      if ( loc != std::string::npos ) {
	// Found it!  It's on the list to disable
	(*h)->setSaveHist(false);
      }

      //--- Is test_name a part of vars_to_enable?
      loc = vars_to_enable.find( test_name, 0 );
      if ( loc != std::string::npos ) {
	// Found it!  It's on the list to enable
	(*h)->setSaveHist(true);
      }

    }
  }



  //------------------------------------------------------------------------
  //!  Configure: enable or disable.
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  inline
  void
  HistoGroup<PHYS_OBJECT>::
  select( std::string  vars_to_select,   // comma separated list of names
	  std::vector< PhysVarHisto * > & selectedHistos )
  {
    //  We can manipulate this string since we passed it whole and
    //  thus are dealing with a local copy of the original argument.
    vars_to_select   = "," + vars_to_select  + ",";


    //  Iterate over PhysVarHistos
    //
    std::vector< PhysVarHisto * >::iterator
      h    = histograms_.begin(),
      hend = histograms_.end();

    for ( ; h != hend; ++h ) {
      // (*h) dereferences to a PhysVarHisto*
      std::string test_name = "," + (*h)->name() + ",";

      if ( verboseLevel_ > 10)
	std::cout << "HistoGroup::select (debug): test_name = " << test_name << std::endl;

      // is test_name a part of vars_to_select?
      std::string::size_type loc = vars_to_select.find( test_name, 0 );
      if ( loc != std::string::npos || vars_to_select == ",all," ) {
	// Found it!  Add *h (since iterator h dereferences to PhysVarHisto*)
	// to the vector of selected PhysVarHistos.
	//
	selectedHistos.push_back( *h );
      }

    }
  }



  //------------------------------------------------------------------------
  //!  clearVec: resets PhysVarHisto ntuple cache
  //------------------------------------------------------------------------
  template <class PHYS_OBJECT>
  void
  HistoGroup<PHYS_OBJECT>::
  clearVec()
  {

    for ( std::vector<PhysVarHisto *>::iterator i = histograms_.begin();
	  i != histograms_.end(); i++ ) {
      (*i)->clearVec();
    }
  }

}
#endif
