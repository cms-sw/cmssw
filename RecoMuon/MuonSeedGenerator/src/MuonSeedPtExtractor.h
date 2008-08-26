#ifndef RecoMuon_MuonSeedGenerator_MuonSeedPtExtractor_H
#define RecoMuon_MuonSeedGenerator_MuonSeedPtExtractor_H

/** \class MuonSeedPtExtractor
 */

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h" 


namespace edm {class ParameterSet;}

class MuonSeedPtExtractor {

 public:
  /// Constructor with Parameter set and MuonServiceProxy
  MuonSeedPtExtractor(const edm::ParameterSet&);

  /// Destructor
  ~MuonSeedPtExtractor();


  std::vector<double> pT_extract(MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstHit,
                                 MuonTransientTrackingRecHit::ConstMuonRecHitPointer secondHit) const;

 std::vector<double> getPt(const std::vector<double> & vPara, double eta, double dPhi ) const; 

 std::vector<double>  dt12() const { return DT12; }
 std::vector<double>  dt13() const { return DT13; }
 std::vector<double>  dt14() const { return DT14; }
 std::vector<double>  dt23() const { return DT23; }
 std::vector<double>  dt24() const { return DT24; }
 std::vector<double>  dt34() const { return DT34; }
 
 std::vector<double>  csc01() const { return CSC01; }
 std::vector<double>  csc12() const { return CSC12; }
 std::vector<double>  csc02() const { return CSC02; }
 std::vector<double>  csc13() const { return CSC13; }
 std::vector<double>  csc03() const { return CSC03; }
 std::vector<double>  csc14() const { return CSC14; }
 std::vector<double>  csc23() const { return CSC23; }
 std::vector<double>  csc24() const { return CSC24; }
 std::vector<double>  csc34() const { return CSC34; }

 std::vector<double>  ol1213() const { return OL1213; }
 std::vector<double>  ol1222() const { return OL1222; }
 std::vector<double>  ol1232() const { return OL1232; }
 std::vector<double>  ol2213() const { return OL2213; }
 std::vector<double>  ol2222() const { return OL2222; }

 std::vector<double>  sme11() const { return SME11; }
 std::vector<double>  sme12() const { return SME12; }
 std::vector<double>  sme13() const { return SME13; }
 std::vector<double>  sme21() const { return SME21; }
 std::vector<double>  sme22() const { return SME22; }
 std::vector<double>  sme31() const { return SME31; }
 std::vector<double>  sme32() const { return SME32; }
 std::vector<double>  sme41() const { return SME41; } 

 std::vector<double>  smb10() const { return SMB10; }
 std::vector<double>  smb11() const { return SMB11; }
 std::vector<double>  smb12() const { return SMB12; }
 std::vector<double>  smb20() const { return SMB20; }
 std::vector<double>  smb21() const { return SMB21; }
 std::vector<double>  smb22() const { return SMB22; }
 std::vector<double>  smb30() const { return SMB30; }
 std::vector<double>  smb31() const { return SMB31; }
 std::vector<double>  smb32() const { return SMB32; }

 private:
  int stationCode(MuonTransientTrackingRecHit::ConstMuonRecHitPointer hit) const;
  // seed parameters vectors
  std::vector<double> DT12;
  std::vector<double> DT13;
  std::vector<double> DT14;
  std::vector<double> DT23;
  std::vector<double> DT24;
  std::vector<double> DT34;

  std::vector<double> CSC01;
  std::vector<double> CSC12;
  std::vector<double> CSC02;
  std::vector<double> CSC13;
  std::vector<double> CSC03;
  std::vector<double> CSC14;
  std::vector<double> CSC23;
  std::vector<double> CSC24;
  std::vector<double> CSC34;

  std::vector<double> OL1213;
  std::vector<double> OL1222;
  std::vector<double> OL1232;
  std::vector<double> OL2213;
  std::vector<double> OL2222;

  std::vector<double> SME11;
  std::vector<double> SME12;
  std::vector<double> SME13;
  std::vector<double> SME21;
  std::vector<double> SME22;
  std::vector<double> SME31;
  std::vector<double> SME32;
  std::vector<double> SME41;

  std::vector<double> SMB10;
  std::vector<double> SMB11;
  std::vector<double> SMB12;
  std::vector<double> SMB20;
  std::vector<double> SMB21;
  std::vector<double> SMB22;
  std::vector<double> SMB30;
  std::vector<double> SMB31;
  std::vector<double> SMB32;


};
#endif
