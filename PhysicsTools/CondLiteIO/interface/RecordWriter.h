#ifndef PhysicsTools_CondLiteIO_RecordWriter_h
#define PhysicsTools_CondLiteIO_RecordWriter_h
// -*- C++ -*-
//
// Package:     CondLiteIO
// Class  :     RecordWriter
// 
/**\class RecordWriter RecordWriter.h PhysicsTools/CondLiteIO/interface/RecordWriter.h

 Description: Used to write the contents of an EventSetup Record to a TFile

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  9 17:00:53 CST 2009
// $Id: RecordWriter.h,v 1.3 2010/02/19 21:13:46 chrjones Exp $
//

// system include files
#include <map>
// user include files
#include "FWCore/Utilities/interface/TypeIDBase.h"
#include "DataFormats/Provenance/interface/ESRecordAuxiliary.h"

// forward declarations
class TFile;
class TBranch;
class TTree;

namespace fwlite {
class RecordWriter
{

   public:
      RecordWriter(const char* iName, TFile* iFile);
      virtual ~RecordWriter();

      struct DataBuffer {
         const void* pBuffer_;
         TBranch* branch_;
         edm::TypeIDBase trueType_;
      };
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void update(const void* iData, const std::type_info& iType, const char* iLabel);
      
      //call update before calling write
      void fill(const edm::ESRecordAuxiliary&);

      void write();

   private:
      RecordWriter(const RecordWriter&); // stop default

      const RecordWriter& operator=(const RecordWriter&); // stop default

      // ---------- member data --------------------------------
      TTree* tree_;
      edm::ESRecordAuxiliary aux_;
      edm::ESRecordAuxiliary* pAux_;
      TBranch* auxBranch_;
      std::map<std::pair<edm::TypeIDBase,std::string>, DataBuffer> idToBuffer_;
};
}

#endif
