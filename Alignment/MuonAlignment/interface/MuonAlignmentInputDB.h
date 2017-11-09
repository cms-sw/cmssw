#ifndef Alignment_MuonAlignment_MuonAlignmentInputDB_h
#define Alignment_MuonAlignment_MuonAlignmentInputDB_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputDB
// 
/**\class MuonAlignmentInputDB MuonAlignmentInputDB.h Alignment/MuonAlignment/interface/MuonAlignmentInputDB.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 17:30:40 CST 2008
// $Id: MuonAlignmentInputDB.h,v 1.1 2008/03/15 20:26:46 pivarski Exp $
//

// system include files

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"

// forward declarations

class MuonAlignmentInputDB: public MuonAlignmentInputMethod {
   public:
      MuonAlignmentInputDB();
      MuonAlignmentInputDB(std::string dtLabel, std::string cscLabel, bool getAPEs);
      ~MuonAlignmentInputDB() override;

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const override;

   private:
      MuonAlignmentInputDB(const MuonAlignmentInputDB&) = delete; // stop default

      const MuonAlignmentInputDB& operator=(const MuonAlignmentInputDB&) = delete; // stop default

      // ---------- member data --------------------------------

      std::string m_dtLabel, m_cscLabel;
      bool m_getAPEs;
};


#endif
