
#include "PhysicsTools/StarterKit/interface/PhysVarHisto.h"

#include <string>
#include <sstream>
#include <iostream>

using pat::PhysVarHisto;

// Constructor:
PhysVarHisto::
PhysVarHisto( std::string name,
	      std::string title,
	      int         nbins,
	      double      xlow,
	      double      xhigh,
	      TFileDirectory * currDir,
	      std::string units,
	      std::string type,
	      bool        saveHist,
	      bool        saveNtup )
  :
  currDir_ (currDir),
  name_  (name),
  type_  (type),
  title_ (title),
  nbins_ (nbins),
  xlow_  (xlow),
  xhigh_ (xhigh),
  units_ (units),
  //
  histos_ (),             // vector of 0 elements
  value_ (-999999),
  value_ext_ (0),
  //
  saveHist_ (saveHist),
  saveNtup_ (saveNtup),
  //
  verboseLevel_ (0)      // default = very verbose (&&& hardcoded)
{
  if (verboseLevel_ > 5)
    std::cout << "PhysVarHisto(" << name_ << "):: in constructor." << std::endl;

  // &&& Not sure what (if anything) will be needed to deal with the ntuple.
}


void
PhysVarHisto::
makeTH1()
{
  if (verboseLevel_ > 5)
    std::cout << "PhysVarHisto(" << name_ << "):: in makeTH1()." << std::endl;


  if (saveHist_ && currDir_) {
    if (verboseLevel_ > 5)
      std::cout << "PhysVarHisto(" << name_ << ")::makeTH1:"
		<< " making new histo." << std::endl;

    std::string name, title;
    std::stringstream partNum;
    partNum << histos_.size()+1;

    name  = name_  + "_" + partNum.str();
    title = title_ + " #" + partNum.str()+ ";" + units_ ;


    TH1D * h = currDir_->make<TH1D>( name.c_str(), title.c_str(),
				     nbins_, xlow_, xhigh_ );
    // &&& Now decorate: set thickness, bkg color, axis labels, etc.
    // &&& lift the axis labels from RooFit which does a nice job on them

    histos_.push_back( h );
  }
}


// Note: imulti follows Fortran-style indexing,
// so imulti==1 is the 1st element.  Note that the histos_ vector
// wakes up with no elements, so histos_.size() gives 0 in the first
// call to fill().
void
PhysVarHisto::
fill( double x, unsigned int imulti, double weight )
{
  // Once this PhysVarHisto was made, then there's no turning back
  // when it comes to fill() : we cache the value no matter what.

  value_ = x;  // save in case we need to capture it for the ntuple.

  valueColl_.push_back( x );

  if ( value_ext_ ) {
    // Defined, use this one as well
    *( (double*)(value_ext_) ) = value_ ;
  }

  if (verboseLevel_ > 5) {  // very verbose
    std::cout << "PhysVarHisto(" << name_ << ")::fill: "
	      << "value = " << value_ << ". External storage is ";
    if (value_ext_)  std::cout << "defined." << std::endl;
    else         std::cout << "not defined." << std::endl;
  }



  if ( ! saveHist_ ) {
    std::cout << "PhysVarHisto(" << name_ << ")::fill: saveHist is not defined" << std::endl;
    return;
  }

  //--- If the requested histo location in multi-histogram
  //--- mode is larger than our current vector of histograms,
  //--- then grow it up to the location requested.
  //
  while ( imulti > histos_.size() ) {
    if (verboseLevel_ > 4) {
      std::cout << "PhysVarHisto(" << name_ << ")::fill: grow histo list by one."
		<< std::endl;
    }
    //--- Make another TH1 at index == current value of size()
    makeTH1();    // histos_ vector grows by one element
  }
  if ( verboseLevel_ > 4 )
    std::cout << "PhysVarHisto(" << name_ << ")::fill: About to fill " << imulti << std::endl;
  histos_[ imulti-1 ]->Fill( value_, weight );
  if ( verboseLevel_ > 4 )
    std::cout << "PhysVarHisto(" << name_ << ")::fill: Done filling " << imulti << std::endl;
}

