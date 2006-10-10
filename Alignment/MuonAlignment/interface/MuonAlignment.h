#ifndef Alignment_MuonAlignment_MuonAlignment_H
#define Alignment_MuonAlignment_MuonAlignment_H

/** \class MuonAlignment
 *  The MuonAlignment helper class for alignment jobs
 *
 *  $Date: 2006/10/10 09:30:50 $
 *  $Revision: 1.10 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


class MuonAlignment {

  public:

      MuonAlignment(const edm::EventSetup& setup);

     ~MuonAlignment() { delete theAlignableMuon; }

      void moveDTChamber( int rawId, std::vector<float> localDisplacements, std::vector<float> localRotations  );

      void moveCSCChamber( int rawId, std::vector<float> localDisplacements, std::vector<float> localRotations  );

      void saveToDB();


  private:

      // ----------member data ---------------------------
      AlignableMuon* theAlignableMuon;
      int rawid;
      std::vector<float> local_displacements;
      std::vector<float> local_rotations;

};

#endif //MuonAlignment_H
