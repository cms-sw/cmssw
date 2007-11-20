#ifndef Alignment_MuonAlignment_MuonAlignment_H
#define Alignment_MuonAlignment_MuonAlignment_H

/** \class MuonAlignment
 *  The MuonAlignment helper class for alignment jobs
 *
 *  $Date: 2007/01/26 19:39:41 $
 *  $Revision: 1.6 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

class MuonAlignment{
  typedef std::map<DetId, Alignable*> MapType;
  typedef std::pair<DetId, Alignable*> PairType;
  public:

      MuonAlignment( const edm::EventSetup& setup );

     ~MuonAlignment() { delete theAlignableMuon;  }
      
      AlignableMuon* getAlignableMuon() { return theAlignableMuon; }

      void moveAlignableLocalCoord( DetId& , std::vector<float>& , std::vector<float>& );

      void moveAlignableGlobalCoord( DetId& , std::vector<float>& , std::vector<float>& );


      void saveToDB();


  private:

      std::string theDTAlignRecordName, theDTErrorRecordName;
      std::string theCSCAlignRecordName, theCSCErrorRecordName;
 
      std::vector<float> displacements;

      std::vector<float> rotations;

    void recursiveGetId( Alignable* alignable );

      AlignableMuon* theAlignableMuon;

  MapType theAlignableMap;


};

#endif //MuonAlignment_H
