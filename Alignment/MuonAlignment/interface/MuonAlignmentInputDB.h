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
// $Id: MuonAlignmentInputDB.h,v 1.2 2009/10/07 20:46:38 pivarski Exp $
//

// system include files

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"

// forward declarations

class MuonAlignmentInputDB: public MuonAlignmentInputMethod {
   public:
      MuonAlignmentInputDB();
      MuonAlignmentInputDB(std::string dtLabel, std::string cscLabel, bool getAPEs);
      virtual ~MuonAlignmentInputDB();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

   private:
      MuonAlignmentInputDB(const MuonAlignmentInputDB&); // stop default

      const MuonAlignmentInputDB& operator=(const MuonAlignmentInputDB&); // stop default

      // ---------- member data --------------------------------

      std::string m_dtLabel, m_cscLabel;
      bool m_getAPEs;
};


#endif
