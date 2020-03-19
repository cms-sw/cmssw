#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIO.h"

//--------------------------------------------------------------------------------------------------
// write one set of original parameters
int AlignmentParametersIO::writeOneOrigRigidBody(Alignable* ali) {
  AlignmentParameters* par = ali->alignmentParameters();
  AlignmentParameters* parBack = (par ? par->clone(par->parameters(), par->covariance()) : nullptr);

  ali->setAlignmentParameters(new RigidBodyAlignmentParameters(ali, true));
  int iret = this->writeOne(ali);

  ali->setAlignmentParameters(parBack);  // deletes the above created RigidBodyAlignmentParameters

  return iret;
}

//-----------------------------------------------------------------------------
// write many parameters
int AlignmentParametersIO::write(const align::Alignables& alivec, bool validCheck) {
  int icount = 0;
  for (align::Alignables::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
    if ((*it)->alignmentParameters()->isValid() || !(validCheck)) {
      icount++;
      int iret = writeOne(*it);
      if (iret != 0)
        return iret;
    }
  }
  edm::LogInfo("Alignment") << "@SUB=AlignmentParametersIO::write"
                            << "Wrote " << icount << " out of " << alivec.size() << " parameters";
  return 0;
}

//-----------------------------------------------------------------------------
// write many original parameters
int AlignmentParametersIO::writeOrigRigidBody(const align::Alignables& alivec, bool validCheck) {
  int icount = 0;
  for (align::Alignables::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
    if (!validCheck || (*it)->alignmentParameters()->isValid()) {
      ++icount;
      int iret = this->writeOneOrigRigidBody(*it);
      if (iret != 0)
        return iret;
    }
  }
  edm::LogInfo("Alignment") << "@SUB=AlignmentParametersIO::writeOrigRigidBody"
                            << "Wrote " << icount << " out of " << alivec.size() << " original parameters.";
  return 0;
}

//-----------------------------------------------------------------------------
// read many parameters

align::Parameters AlignmentParametersIO::read(const align::Alignables& alivec, int& ierr) {
  align::Parameters retvec;
  int ierr2;
  int icount = 0;
  for (align::Alignables::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
    AlignmentParameters* ad = readOne(*it, ierr2);
    if (ad != nullptr && ierr2 == 0) {
      retvec.push_back(ad);
      icount++;
    }
  }
  edm::LogInfo("Alignment") << "@SUB-AlignmentParametersIO::write"
                            << "Read " << icount << " out of " << alivec.size() << " parameters";
  return retvec;
}
