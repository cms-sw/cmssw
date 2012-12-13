#ifndef Alignment_MuonAlignment_MuonAlignmentInputDB_h
#define Alignment_MuonAlignment_MuonAlignmentInputDB_h

/**\class MuonAlignmentInputDB

 Class for using database as an input for muon alignment

*/

//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 17:30:40 CST 2008
//
// $Id: MuonAlignmentInputDB.h,v 1.2 2009/10/07 20:46:38 pivarski Exp $
//

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"


class MuonAlignmentInputDB: public MuonAlignmentInputMethod
{
public:

  MuonAlignmentInputDB();
  MuonAlignmentInputDB(std::string dtLabel, std::string cscLabel, bool getAPEs);

  virtual ~MuonAlignmentInputDB();

  virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

private:

  MuonAlignmentInputDB(const MuonAlignmentInputDB&); // stop default

  const MuonAlignmentInputDB& operator=(const MuonAlignmentInputDB&); // stop default

  std::string m_dtLabel, m_cscLabel;
  bool m_getAPEs;
};

#endif
