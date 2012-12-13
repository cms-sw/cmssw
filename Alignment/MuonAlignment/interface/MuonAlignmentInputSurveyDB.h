#ifndef Alignment_MuonAlignment_MuonAlignmentInputSurveyDB_h
#define Alignment_MuonAlignment_MuonAlignmentInputSurveyDB_h

//
// Original Author:  Jim Pivarski
//         Created:  Fri Mar  7 16:13:19 CST 2008
//
// $Id: MuonAlignmentInputSurveyDB.h,v 1.1 2008/03/15 20:26:46 pivarski Exp $
//

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"


class MuonAlignmentInputSurveyDB: public MuonAlignmentInputMethod
{
public:

  MuonAlignmentInputSurveyDB();
  MuonAlignmentInputSurveyDB(std::string dtLabel, std::string cscLabel);

  virtual ~MuonAlignmentInputSurveyDB();

  virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

private:

  MuonAlignmentInputSurveyDB(const MuonAlignmentInputSurveyDB&); // stop default

  const MuonAlignmentInputSurveyDB& operator=(const MuonAlignmentInputSurveyDB&); // stop default

  void addSurveyInfo_(Alignable* ali,
                      unsigned int* theSurveyIndex,
                      const Alignments* theSurveyValues,
                      const SurveyErrors* theSurveyErrors) const;

  std::string m_dtLabel, m_cscLabel;
};

#endif
