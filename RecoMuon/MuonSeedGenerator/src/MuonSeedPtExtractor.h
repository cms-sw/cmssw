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
                                       MuonTransientTrackingRecHit::ConstMuonRecHitPointer secondHit);

 std::vector<double> getPt(std::vector<double> vPara, double eta, double dPhi ); 

 std::vector<double>  dt12(){ return DT12; }
 std::vector<double>  dt13(){ return DT13; }
 std::vector<double>  dt14(){ return DT14; }
 std::vector<double>  dt23(){ return DT23; }
 std::vector<double>  dt24(){ return DT24; }
 std::vector<double>  dt34(){ return DT34; }
 
 std::vector<double>  csc01(){ return CSC01; }
 std::vector<double>  csc12(){ return CSC12; }
 std::vector<double>  csc02(){ return CSC02; }
 std::vector<double>  csc13(){ return CSC13; }
 std::vector<double>  csc03(){ return CSC03; }
 std::vector<double>  csc14(){ return CSC14; }
 std::vector<double>  csc23(){ return CSC23; }
 std::vector<double>  csc24(){ return CSC24; }
 std::vector<double>  csc34(){ return CSC34; }

 std::vector<double>  ol1213(){ return OL1213; }
 std::vector<double>  ol1222(){ return OL1222; }
 std::vector<double>  ol1232(){ return OL1232; }
 std::vector<double>  ol2213(){ return OL2213; }
 std::vector<double>  ol2222(){ return OL2222; }

 std::vector<double>  sme11(){ return SME11; }
 std::vector<double>  sme12(){ return SME12; }
 std::vector<double>  sme13(){ return SME13; }
 std::vector<double>  sme21(){ return SME21; }
 std::vector<double>  sme22(){ return SME22; }
 std::vector<double>  sme31(){ return SME31; }
 std::vector<double>  sme32(){ return SME32; }
 std::vector<double>  sme41(){ return SME41; } 

 std::vector<double>  smb10(){ return SMB10; }
 std::vector<double>  smb11(){ return SMB11; }
 std::vector<double>  smb12(){ return SMB12; }
 std::vector<double>  smb20(){ return SMB20; }
 std::vector<double>  smb21(){ return SMB21; }
 std::vector<double>  smb22(){ return SMB22; }
 std::vector<double>  smb30(){ return SMB30; }
 std::vector<double>  smb31(){ return SMB31; }
 std::vector<double>  smb32(){ return SMB32; }

 private:
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
