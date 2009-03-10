#ifndef TK_PhysVarHisto_h
#define TK_PhysVarHisto_h 1

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


// STL include files
#include <string>
#include <vector>

// ROOT include files
#include <TH1D.h>
#include <TH1F.h>

// &&& Alert!  Is this a dependence on full framework?
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

namespace pat {

  class PhysVarHisto
  {
  public:
    PhysVarHisto( std::string name,
	       std::string title,
	       int         nbins,
	       double      xlow,
	       double      xhigh,
	       TFileDirectory * currDir = 0,
	       std::string units = "",
	       std::string type  = "D",
	       bool        saveHist = true,
	       bool        saveNtup = false );

    virtual ~PhysVarHisto() { };  //!  Note we don't delete histograms!

    //--- Make one TH1 (may need more than one); all decorations should be done in this call.
    virtual void makeTH1();

    //--- Fill one of the histograms in histos_ vector.
    virtual void fill( double x,
		       unsigned int imulti = 1,
		       double weight = 1.0 );

    //--- Inline accessors.
    inline std::string name() { return name_ ; } //!< ROOT/cfg handle
    inline std::string type() { return type_ ; } //!< type of value_
    inline double value() { return value_ ; }    //!< current value

    template <class T>
      void vec(std::vector<T> & retVec)
    {
      retVec.resize( valueColl_.size() );
      for ( unsigned int i = 0; i < valueColl_.size(); i++ )
	retVec[i] = static_cast<T>( valueColl_[i] );
    } //!< vector of current values in a list

    inline bool   saveHist() { return saveHist_ ; }
    inline bool   saveNtup() { return saveNtup_ ; }

    inline void   setSaveHist(bool flag) { saveHist_ = flag; } //!< save into a histogram
    inline void   setSaveNtup(bool flag) { saveNtup_ = flag; } //!< save into a ntuple

    inline void   setTFileDirectory(TFileDirectory * dir) { currDir_ = dir; };

    inline void   clearVec() { valueColl_.clear(); }

  private:
    //--- Stuff needed to book one histogram
    TFileDirectory * currDir_ ;  //!< ROOT thingy that makes/manages histograms
    std::string name_ ;   //!< ROOT handle, but used in cfg files as well...
    std::string type_ ;   //!< stuff for TBranch() constructor (e.g. "F4")
    std::string title_ ;  //!< nice, descriptive title
    int         nbins_ ;  //!< num of bins
    double      xlow_ ;   //!< min value
    double      xhigh_ ;  //!< max value
    std::string units_ ;  //!< "GeV/c^{2}" etc. for axis labels

    //--- Cache to make histograms
    std::vector<TH1D *> histos_ ;   // maybe use a base class TH1* ?
    // &&& Should we template all this?

    //--- Internal cache
    double      value_ ;        // our own cache
    void *      value_ext_ ;    // cache is in a struct elsewhere, for TBranch
    std::vector<double> valueColl_; // our own cache of a list of values

    //--- Flags to control behavior
    // bool        active_ ;    // no clear use case to have this flag...
    bool        saveHist_ ;     // save info into a histogram
    bool        saveNtup_ ;     // save info into a ntuple

    int         verboseLevel_ ; // how much verbosity
  };

}

#endif
