// -*- C++ -*-

#if !defined(FWLiteCont_H)
#define FWLiteCont_H

#include <map>
#include <string>

#include "TH1.h"
#include "TFile.h"

#include "DataFormats/FWLite/interface/EventBase.h"
#include "PhysicsTools/FWLite/interface/TH1Store.h"

class FWLiteCont : public fwlite::EventBase
{
   public:

      //////////////////////
      // Public Constants //
      //////////////////////

      typedef std::map< std::string, std::string >   SSMap;

      /////////////
      // friends //
      /////////////
      // tells particle data how to print itself out
      friend std::ostream& operator<< (std::ostream& o_stream, 
                                       const FWLiteCont &rhs);

      //////////////////////////
      //            _         //
      // |\/|      |_         //
      // |  |EMBER | UNCTIONS //
      //                      //
      //////////////////////////

      /////////////////////////////////
      // Constructors and Destructor //
      /////////////////////////////////
      FWLiteCont();
      ~FWLiteCont();

       ////////////////
      // One Liners //
      ////////////////

      //////////////////////////////
      // Regular Member Functions //
      //////////////////////////////

      // adds a histogram pointer to the map
      void add (TH1 *histPtr);

      // given a string, returns corresponding histogram pointer
      TH1* hist (const std::string &name);

      // write all histograms to a root file
      void write (const std::string &filename) const;
      void write (TFile *filePtr) const;

      // implement the two functions needed to make this an EventBase.
      bool getByLabel (const std::type_info& iInfo,
                       const char* iModuleLabel,
                       const char* iProductInstanceLabel,
                       const char* iProcessLabel,
                       void* oData) const;

      const std::string getBranchNameFor (const std::type_info& iInfo,
                                          const char* iModuleLabel,
                                          const char* iProductInstanceLabel,
                                          const char* iProcessLabel) const;

      const FWLiteCont& operator++();
      
      /////////////////////////////
      // Static Member Functions //
      /////////////////////////////


  private:

      /////////////////////////
      // Private Member Data //
      /////////////////////////

      fwlite::EventBase *m_eventBasePtr;
      TH1Store           m_histStore;

};


#endif // FWLiteCont_H
