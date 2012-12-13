//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 17:30:46 CST 2008
//
// $Id: MuonAlignmentInputSurveyDB.cc,v 1.2 2008/03/20 21:39:26 pivarski Exp $
//

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputSurveyDB.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorRcd.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"


MuonAlignmentInputSurveyDB::MuonAlignmentInputSurveyDB()
   : m_dtLabel(""), m_cscLabel("") {}


MuonAlignmentInputSurveyDB::MuonAlignmentInputSurveyDB(std::string dtLabel, std::string cscLabel)
   : m_dtLabel(dtLabel), m_cscLabel(cscLabel) {}


MuonAlignmentInputSurveyDB::~MuonAlignmentInputSurveyDB() {}


AlignableMuon *MuonAlignmentInputSurveyDB::newAlignableMuon(const edm::EventSetup& iSetup) const
{
  boost::shared_ptr<DTGeometry> dtGeometry = idealDTGeometry(iSetup);
  boost::shared_ptr<CSCGeometry> cscGeometry = idealCSCGeometry(iSetup);

  edm::ESHandle<Alignments> dtSurvey;
  edm::ESHandle<SurveyErrors> dtSurveyError;
  edm::ESHandle<Alignments> cscSurvey;
  edm::ESHandle<SurveyErrors> cscSurveyError;
  iSetup.get<DTSurveyRcd>().get(m_dtLabel, dtSurvey);
  iSetup.get<DTSurveyErrorRcd>().get(m_dtLabel, dtSurveyError);
  iSetup.get<CSCSurveyRcd>().get(m_cscLabel, cscSurvey);
  iSetup.get<CSCSurveyErrorRcd>().get(m_cscLabel, cscSurveyError);

  AlignableMuon *output = new AlignableMuon(&(*dtGeometry), &(*cscGeometry));

  unsigned int theSurveyIndex  = 0;
  const Alignments *theSurveyValues = &*dtSurvey;
  const SurveyErrors *theSurveyErrors = &*dtSurveyError;
  std::vector<Alignable*> barrels = output->DTBarrel();
  for (std::vector<Alignable*>::const_iterator iter = barrels.begin();  iter != barrels.end();  ++iter)
  {
    addSurveyInfo_(*iter, &theSurveyIndex, theSurveyValues, theSurveyErrors);
  }

  theSurveyIndex  = 0;
  theSurveyValues = &*cscSurvey;
  theSurveyErrors = &*cscSurveyError;
  std::vector<Alignable*> endcaps = output->CSCEndcaps();
  for (std::vector<Alignable*>::const_iterator iter = endcaps.begin();  iter != endcaps.end();  ++iter)
  {
    addSurveyInfo_(*iter, &theSurveyIndex, theSurveyValues, theSurveyErrors);
  }

  return output;
}

// This function was copied (with minimal modifications) from
// Alignment/CommonAlignmentProducer/plugins/AlignmentProducer.cc
// (version CMSSW_5_0_0_pre5), guaranteed to work the same way
// unless AlignmentProducer.cc's version changes!
void MuonAlignmentInputSurveyDB::addSurveyInfo_(Alignable* ali,
                                                unsigned int *theSurveyIndex,
                                                const Alignments* theSurveyValues,
                                                const SurveyErrors* theSurveyErrors) const
{
  const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo_(comp[i], theSurveyIndex, theSurveyValues, theSurveyErrors);

  const SurveyError& error = theSurveyErrors->m_surveyErrors[*theSurveyIndex];

  if ( ali->id() != error.rawId() || ali->alignableObjectId() != error.structureType() )
  {
    throw cms::Exception("DatabaseError") << "Error reading survey info from DB. Mismatched id!";
  }

  const CLHEP::Hep3Vector&  pos = theSurveyValues->m_align[*theSurveyIndex].translation();
  const CLHEP::HepRotation& rot = theSurveyValues->m_align[*theSurveyIndex].rotation();

  AlignableSurface surf( align::PositionType( pos.x(), pos.y(), pos.z() ),
                         align::RotationType( rot.xx(), rot.xy(), rot.xz(),
                                              rot.yx(), rot.yy(), rot.yz(),
                                              rot.zx(), rot.zy(), rot.zz() ) );
  surf.setWidth( ali->surface().width() );
  surf.setLength( ali->surface().length() );

  ali->setSurvey( new SurveyDet( surf, error.matrix() ) );

  (*theSurveyIndex)++;
}
