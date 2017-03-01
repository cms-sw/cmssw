#ifndef FWCore_Services_ProcInfoFetcher_h
#define FWCore_Services_ProcInfoFetcher_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     ProcInfoFetcher
// 
/**\class ProcInfoFetcher ProcInfoFetcher.h FWCore/Services/interface/ProcInfoFetcher.h

 Description:Class used to fetch process information

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun May  6 11:14:28 CDT 2012
//

// system include files

// user include files

// forward declarations
namespace edm {
  namespace service {
    struct ProcInfo
    {
      ProcInfo():vsize(),rss() {}
      ProcInfo(double sz, double rss_sz): vsize(sz),rss(rss_sz) {}
      
      bool operator==(const ProcInfo& p) const
      { return vsize==p.vsize && rss==p.rss; }
      
      bool operator>(const ProcInfo& p) const
      { return vsize>p.vsize || rss>p.rss; }
      
      // see proc(4) man pages for units and a description
      double vsize;   // in MB (used to be in pages?)
      double rss;     // in MB (used to be in pages?)
    };
    
    class ProcInfoFetcher {
    public:
      ProcInfoFetcher();
      ~ProcInfoFetcher();
      ProcInfoFetcher(ProcInfoFetcher const&) = delete;
      ProcInfoFetcher& operator=(ProcInfoFetcher const&) = delete;
      
      ProcInfo fetch() const;
    private:
      double pg_size_;
      int fd_;
      mutable char buf_[500];
    };
  }
}
#endif
