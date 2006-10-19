#ifndef Alignment_MuonAlignment_MuonAlignment_H
#define Alignment_MuonAlignment_MuonAlignment_H

/** \class MuonAlignment
 *  The MuonAlignment helper class for alignment jobs
 *
 *  $Date: 2006/10/11 14:59:55 $
 *  $Revision: 1.3 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

class MuonAlignment{

  public:

      MuonAlignment( const edm::EventSetup& setup );

     ~MuonAlignment() { delete theAlignableMuon; }
      
      AlignableMuon* getAlignableMuon() { return theAlignableMuon; }

      void moveAlignableDTChamber( int rawId, std::vector<float> localDisplacements, std::vector<float> localRotations  );

      void moveAlignableCSCChamber( int rawId, std::vector<float> localDisplacements, std::vector<float> localRotations  );

      void saveToDB();
      int rawid;

      std::vector<float> local_displacements;

      std::vector<float> local_rotations;


      AlignableMuon* theAlignableMuon;


  private:

};

#endif //MuonAlignment_H
