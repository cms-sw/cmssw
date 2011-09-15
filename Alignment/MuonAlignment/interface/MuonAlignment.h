#ifndef Alignment_MuonAlignment_MuonAlignment_H
#define Alignment_MuonAlignment_MuonAlignment_H

/** \class MuonAlignment
 *  The MuonAlignment helper class for alignment jobs
 *
 *  $Date: 2008/03/26 22:02:51 $
 *  $Revision: 1.13 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include <vector>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include <FWCore/Framework/interface/Frameworkfwd.h> 
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"

class MuonAlignment {

  public:

      MuonAlignment( const edm::EventSetup& iSetup );

      MuonAlignment( const edm::EventSetup& iSetup, const MuonAlignmentInputMethod& input );

     ~MuonAlignment() { delete theAlignableMuon; delete theAlignableNavigator; }
      
      AlignableMuon* getAlignableMuon() { return theAlignableMuon; }

      AlignableNavigator* getAlignableNavigator() { return theAlignableNavigator; }


      void moveAlignableLocalCoord( DetId& , std::vector<float>& , std::vector<float>& );
      void moveAlignableGlobalCoord( DetId& , std::vector<float>& , std::vector<float>& );

      void recursiveList(std::vector<Alignable*> alignables, std::vector<Alignable*> &theList);
      void recursiveMap(std::vector<Alignable*> alignables, std::map<align::ID, Alignable*> &theMap);
      void recursiveStructureMap(std::vector<Alignable*> alignables, std::map<std::pair<align::StructureType, align::ID>, Alignable*> &theMap);

      void copyAlignmentToSurvey(double shiftErr, double angleErr);
      void fillGapsInSurvey(double shiftErr, double angleErr);
      void copySurveyToAlignment();

      void writeXML(const edm::ParameterSet &iConfig, const edm::EventSetup &iSetup);

      void saveDTSurveyToDB();
      void saveCSCSurveyToDB();
      void saveSurveyToDB();

      void saveDTtoDB();
      void saveCSCtoDB();
      void saveToDB();


  private:
      void init();
      void recursiveCopySurveyToAlignment(Alignable *alignable);

      std::string theDTAlignRecordName, theDTErrorRecordName;
      std::string theCSCAlignRecordName, theCSCErrorRecordName;
      std::string theDTSurveyRecordName, theDTSurveyErrorRecordName;
      std::string theCSCSurveyRecordName, theCSCSurveyErrorRecordName;
 
      std::vector<float> displacements;

      std::vector<float> rotations;

      AlignableMuon* theAlignableMuon;

      AlignableNavigator* theAlignableNavigator;


};

#endif //MuonAlignment_H
