// -*- C++ -*-

#if !defined(TH1Store_H)
#define TH1Store_H

#include <map>
#include <string>

#include "TH1.h"
#include "TFile.h"

class TH1Store
{
   public:

      //////////////////////
      // Public Constants //
      //////////////////////

      typedef std::map< std::string, TH1* > STH1PtrMap;
      typedef STH1PtrMap::iterator          STH1PtrMapIter;
      typedef STH1PtrMap::const_iterator    STH1PtrMapConstIter;

      /////////////
      // friends //
      /////////////
      // tells particle data how to print itself out
      friend std::ostream& operator<< (std::ostream& o_stream, 
                                       const TH1Store &rhs);

      //////////////////////////
      //            _         //
      // |\/|      |_         //
      // |  |EMBER | UNCTIONS //
      //                      //
      //////////////////////////

      /////////////////////////////////
      // Constructors and Destructor //
      /////////////////////////////////
      TH1Store();
      ~TH1Store();

       ////////////////
      // One Liners //
      ////////////////
      // Whether or not to delete histogram pointers on destruction
      void setDeleteOnDestruction (bool deleteOnDestruction = true) 
      { m_deleteOnDestruction = deleteOnDestruction; }

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

      /////////////////////////////
      // Static Member Functions //
      /////////////////////////////

      // turn on verbose messages (e.g., printing out histogram names
      // when being made)
      static void setVerbose (bool verbose = true)
      { sm_verbose = verbose; }

  private:

      /////////////////////////
      // Private Member Data //
      /////////////////////////

      bool       m_deleteOnDestruction;
      STH1PtrMap m_ptrMap;
      
      ////////////////////////////////
      // Private Static Member Data //
      ////////////////////////////////

      static bool sm_verbose;

};


#endif // TH1Store_H
