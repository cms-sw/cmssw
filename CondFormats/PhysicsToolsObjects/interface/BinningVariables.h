#ifndef BinningVariables_h
#define BinningVariables_h


#include <map>

class BinningVariables {
 public:
  enum  BinningVariablesType{
// Jets
JetEta=1, JetEt=2, JetPhi=3, JetNTracks=4, JetAbsEta=5,
// Muons 
MuonPt=1001, MuonCharge=1002,MuonEta=1003, MuonPhi=1004,
// Continiuos Discriminator
Discriminator = 2001};
};


#endif
