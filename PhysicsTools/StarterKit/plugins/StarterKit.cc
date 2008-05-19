#include "PhysicsTools/StarterKit/interface/StarterKit.h"

using namespace std;
using namespace pat;


//
// constructors and destructor
//
StarterKit::StarterKit(const edm::ParameterSet& iConfig)
  :
  physHistos_(),
  verboseLevel_(0),
  ntVars_()
{
  // Initialize histogram objects
  outputTextName_ = iConfig.getParameter<string>    ("outputTextName");
  outputFile_.open( outputTextName_.c_str() );

  // Initialize TFileService
  edm::Service<TFileService> fs;
  TFileDirectory summary = TFileDirectory( fs->mkdir("summary") );


  // Get list of histograms to enable and disable
  string histos_to_disable =
    iConfig.getParameter<string>    ("disable");
  string histos_to_enable  =
    iConfig.getParameter<string>    ("enable");
  physHistos_.configure( histos_to_disable, histos_to_enable );



  // &&& Ntuple booking begin

  // Now that we know which variables are in the game, we could also
  // decide which ones to ntuplize
  string list_of_ntuple_vars =
    iConfig.getParameter<std::string>    ("ntuplize");

  if (list_of_ntuple_vars != "") {
    //
    //--- Collect all PhysVarHistos which need to store ntuple
    //--- variables and put them in here.
    physHistos_.select( list_of_ntuple_vars, ntVars_ );

    //--- Iterate over the list and "book" them via EDM
    std::vector< PhysVarHisto* >::iterator
      p    = ntVars_.begin(),
      pEnd = ntVars_.end();

    for ( ; p != pEnd; ++p ) {
      cout << "Adding ntuple variable " << (*p)->name() << endl;
      addNtupleVar( (*p)->name(), (*p)->type() );
    }
    //
  } // end if


}


StarterKit::~StarterKit()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
// void StarterKit::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
void StarterKit::produce( edm::Event & evt, const edm::EventSetup & es )
{
  using namespace edm;
  using namespace std;

  if ( verboseLevel_ > 10 )
    std::cout << "StarterKit:: in analyze()." << std::endl;

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------

  evt.getByLabel("selectedLayer1Muons",     muonHandle_);
  evt.getByLabel("selectedLayer1Electrons", electronHandle_);
  evt.getByLabel("selectedLayer1Taus",      tauHandle_);
  evt.getByLabel("selectedLayer1Jets",      jetHandle_);
  evt.getByLabel("selectedLayer1METs",      METHandle_);
  evt.getByLabel("selectedLayer1Photons",   photonHandle_);

  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------
  if ( verboseLevel_ > 10 )
    std::cout << "StarterKit::analyze: calling fillCollection()." << std::endl;
  physHistos_.clearVec();  // clears ntuple cache
  physHistos_.fillCollection( *muonHandle_ );
  physHistos_.fillCollection( *electronHandle_ );
  physHistos_.fillCollection( *tauHandle_ );
  physHistos_.fillCollection( *jetHandle_ );
  physHistos_.fillCollection( *METHandle_ );
  physHistos_.fillCollection( *photonHandle_ );

  // save the list of ntuple varibles to the event record
  saveNtuple( ntVars_, evt );

}

void
StarterKit::saveNtuple( const std::vector<pat::PhysVarHisto*> & ntvars,
			edm::Event & evt )
{
  //  Ntuplization
  if ( ntvars.size() ) {

    //--- Iterate over the list and "fill" them via EDM
    std::vector< PhysVarHisto* >::const_iterator
      p    = ntvars.begin(),
      pEnd = ntvars.end();

    for ( ; p != pEnd; ++p ) {

      if      ((*p)->type() == "D") {
	saveNtupleVar<double>( (*p)->name(), (*p)->value(), evt );
      }
      else if ((*p)->type() == "F") {
	saveNtupleVar<float>(  (*p)->name(), (*p)->value(), evt );
      }
      else if ((*p)->type() == "I") {
	saveNtupleVar<int>(  (*p)->name(), static_cast<int>( (*p)->value() ), evt );
      }
      else if ((*p)->type() == "i") {
	saveNtupleVar<unsigned int>(  (*p)->name(), static_cast<unsigned int>((*p)->value()), evt );
      }
      else if ((*p)->type() == "S") {
	saveNtupleVar<short>(  (*p)->name(), static_cast<short>((*p)->value()), evt );
      }
      else if ((*p)->type() == "s") {
	saveNtupleVar<unsigned short>(  (*p)->name(), static_cast<unsigned short>((*p)->value()), evt );
      }
      else if ((*p)->type() == "L") {
	saveNtupleVar<long>(  (*p)->name(), static_cast<long>((*p)->value()), evt );
      }
      else if ((*p)->type() == "l") {
	saveNtupleVar<unsigned long>(  (*p)->name(), static_cast<unsigned long>((*p)->value()), evt );
      }
      else if ((*p)->type() == "vD") {
	vector<double> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<double>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vF") {
	vector<float> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<float>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vI") {
	vector<int> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<int>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vi") {
	vector<unsigned int> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<unsigned int>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vS") {
	vector<short> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<short>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vs") {
	vector<unsigned short> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<unsigned short>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vL") {
	vector<long> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<long>( (*p)->name(), retvec, evt );
      }
      else if ((*p)->type() == "vl") {
	vector<unsigned long> retvec;
	(*p)->vec( retvec );
	saveNtupleVec<unsigned long>( (*p)->name(), retvec, evt );
      }
      // &&& else if (type == "p4") {
      // &&&   produces<math::XYZTLorentzVector> ( name ).setBranchAlias( name );
      // &&& }
      else {
	std::cout << "StarterKit::addNtupleVar (ERROR): "
		  << "unknown type " << std::endl;

	// &&& Throw an exception in order to abort the job!
      }


      (*p)->clearVec(); // Clear ntuple cache here as well

    }
    //
  } // end if
}



// &&& Design task: add all data types supported by PhysVarHisto
// &&& Design comments:
//     Here's a list of types accepted by ROOT:
//             - C : a character string terminated by the 0 character
//             - B : an 8 bit signed integer (Char_t)
//             - b : an 8 bit unsigned integer (UChar_t)
//             - S : a 16 bit signed integer (Short_t)
//             - s : a 16 bit unsigned integer (UShort_t)
//             - I : a 32 bit signed integer (Int_t)
//             - i : a 32 bit unsigned integer (UInt_t)
//             - F : a 32 bit floating point (Float_t)
//             - D : a 64 bit floating point (Double_t)
//             - L : a 64 bit signed integer (Long64_t)
//             - l : a 64 bit unsigned integer (ULong64_t)
void
StarterKit::addNtupleVar( std::string name, std::string type )
{
  if      (type == "D") {
    produces<double>( name ).setBranchAlias( name );
  }
  else if (type == "F") {
    produces<float>( name ).setBranchAlias( name );
  }
  else if (type == "I") {
    produces<int>( name ).setBranchAlias( name );
  }
  else if (type == "i") {
    produces<unsigned int>( name ).setBranchAlias( name );
  }
  else if (type == "S") {
    produces<short>( name ).setBranchAlias( name );
  }
  else if (type == "s") {
    produces<unsigned short>( name ).setBranchAlias( name );
  }
  else if (type == "L") {
    produces<long>( name ).setBranchAlias( name );
  }
  else if (type == "l") {
    produces<unsigned long>( name ).setBranchAlias( name );
  }
  else if (type == "vD") {
    produces<vector<double> >( name ).setBranchAlias( name );
  }
  else if (type == "vF") {
    produces<vector<float> >( name ).setBranchAlias( name );
  }
  else if (type == "vI") {
    produces<vector<int> >( name ).setBranchAlias( name );
  }
  else if (type == "vi") {
    produces<vector<unsigned int> >( name ).setBranchAlias( name );
  }
  else if (type == "vS") {
    produces<vector<short> >( name ).setBranchAlias( name );
  }
  else if (type == "vs") {
    produces<vector<unsigned short> >( name ).setBranchAlias( name );
  }
  else if (type == "vL") {
    produces<vector<long> >( name ).setBranchAlias( name );
  }
  else if (type == "vl") {
    produces<vector<unsigned long> >( name ).setBranchAlias( name );
  }
  // &&& else if (type == "p4") {
  // &&&   produces<math::XYZTLorentzVector> ( name ).setBranchAlias( name );
  // &&& }
  else {
    std::cout << "StarterKit::addNtupleVar (ERROR): "
	      << "unknown type " << type << std::endl;

    // &&& Throw an exception in order to abort the job!
  }
}

template <class T>
void
StarterKit::saveNtupleVar( std::string name, T value,
			      edm::Event & evt )
{
  std::auto_ptr<T> aptr( new T (value ) );
  evt.put( aptr, name );
}


template <class T>
void
StarterKit::saveNtupleVec( std::string name, const vector<T> & value,
			      edm::Event & evt )
{
  std::auto_ptr<vector<T> > aptr( new vector<T> ( value ) );
  evt.put( aptr, name );
}






// ------------ method called once each job just before starting event loop  ------------
void
StarterKit::beginJob(const edm::EventSetup&)
{
}



// ------------ method called once each job just after ending the event loop  ------------
void
StarterKit::endJob() {
}


// ------------ method to print out reco::Candidates -------------------------------------
std::ostream & operator<<( std::ostream & out, const reco::Candidate & cand )
{
  char buff[1000];
  sprintf( buff, "Pt, Eta, Phi, M = (%6.2f, %6.2f, %6.2f, %6.2f)",
           cand.pt(), cand.eta(), cand.phi(), cand.mass() );
  out << buff;
  return out;
}

//define this as a plug-in
DEFINE_FWK_MODULE(StarterKit);
