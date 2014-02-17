#ifndef Alignment_MuonAlignment_MuonAlignmentInputSurveyDB_h
#define Alignment_MuonAlignment_MuonAlignmentInputSurveyDB_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputSurveyDB
// 
/**\class MuonAlignmentInputSurveyDB MuonAlignmentInputSurveyDB.h Alignment/MuonAlignment/interface/MuonAlignmentInputSurveyDB.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri Mar  7 16:13:19 CST 2008
// $Id: MuonAlignmentInputSurveyDB.h,v 1.1 2008/03/15 20:26:46 pivarski Exp $
//

// system include files

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"

// forward declarations

class MuonAlignmentInputSurveyDB: public MuonAlignmentInputMethod {
   public:
      MuonAlignmentInputSurveyDB();
      MuonAlignmentInputSurveyDB(std::string dtLabel, std::string cscLabel);
      virtual ~MuonAlignmentInputSurveyDB();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

   private:
      MuonAlignmentInputSurveyDB(const MuonAlignmentInputSurveyDB&); // stop default

      const MuonAlignmentInputSurveyDB& operator=(const MuonAlignmentInputSurveyDB&); // stop default

      void addSurveyInfo_(Alignable* ali,
			  unsigned int* theSurveyIndex,
			  const Alignments* theSurveyValues,
			  const SurveyErrors* theSurveyErrors) const;

      // ---------- member data --------------------------------

      std::string m_dtLabel, m_cscLabel;
};


#endif
