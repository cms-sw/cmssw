#ifndef Alignment_MuonAlignment_MuonAlignment_H
#define Alignment_MuonAlignment_MuonAlignment_H

/** \class MuonAlignment
 *  The MuonAlignment helper class for alignment jobs
 *
 *  $Date: 2006/11/22 13:15:25 $
 *  $Revision: 1.5 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

class MuonAlignment{

  public:

      MuonAlignment( const edm::EventSetup& setup );

     ~MuonAlignment() { delete theAlignableMuon; delete theAlignableNavigator; }
      
      AlignableMuon* getAlignableMuon() { return theAlignableMuon; }

      AlignableNavigator* getAlignableNavigator() { return theAlignableNavigator; }


      void moveAlignableLocalCoord( DetId& , std::vector<float>& , std::vector<float>& );

      void moveAlignableGlobalCoord( DetId& , std::vector<float>& , std::vector<float>& );


      void saveToDB();


  private:

      std::string theDTAlignRecordName, theDTErrorRecordName;
      std::string theCSCAlignRecordName, theCSCErrorRecordName;
 
      std::vector<float> displacements;

      std::vector<float> rotations;

      AlignableMuon* theAlignableMuon;

      AlignableNavigator* theAlignableNavigator;


};

#endif //MuonAlignment_H
