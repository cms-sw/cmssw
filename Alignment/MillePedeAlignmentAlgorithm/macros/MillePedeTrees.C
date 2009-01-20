// Original Author: Gero Flucke
// last change    : $Date: 2009/01/20 18:05:45 $
// by             : $Author: flucke $

#include "TTree.h"
#include "TFriendElement.h"
// in header #include "TString.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TError.h"

#include <iostream>

#include "MillePedeTrees.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// implementations
//
////////////////////////////////////////////////////////////////////////////////////////////////////

MillePedeTrees::MillePedeTrees(const char *fileName, Int_t iter, const char *treeNameAdd)
  : fTree(NULL), fOrgPos("AlignablesOrgPos_0"),
    fMisPos("AlignablesAbsPos_0"), fMisRelPos("AlignablesRelPos_0"),
    fMisPar("AlignmentParameters_0"), 
    fPos(Form("AlignablesAbsPos_%d", iter)), fRelPos(Form("AlignablesRelPos_%d", iter)),
    fPar(Form("AlignmentParameters_%d", iter)), fMp(Form("MillePedeUser_%d", iter)),
    fUseSignedR(false)
{
  fTree = this->CreateTree(fileName, treeNameAdd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
MillePedeTrees::~MillePedeTrees()
{
  delete fTree->GetCurrentFile(); // deletes everything in file: this tree and its friends etc.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TTree* MillePedeTrees::CreateTree(const char *fileName, const TString &treeNameAdd)
{
  TFile *file = TFile::Open(fileName);
  if (!file) return NULL;

  TString *allTreeNames[] = {&fOrgPos, &fPos, &fMisPos, &fMisRelPos, &fMisPar, &fRelPos,
                             &fPar, &fMp};
  const unsigned int nTree = sizeof(allTreeNames) / sizeof(allTreeNames[0]);

  unsigned int iTree = 0;
  TTree *mainTree = NULL;
  do {
    file->GetObject(allTreeNames[iTree]->Data(), mainTree);
    if (!mainTree) {
      ::Error("MillePedeTrees::CreateTree",
              "no tree %s in %s", allTreeNames[iTree]->Data(), fileName);
    } 
    *(allTreeNames[iTree]) += treeNameAdd; // Yes, we really change the data members!
    if (mainTree && !treeNameAdd.IsNull()) {
      mainTree->SetName(*(allTreeNames[iTree]));
    }
    ++iTree;
  } while (!mainTree && iTree < nTree);

  if (mainTree) {
    for (unsigned int jTree = iTree; jTree < nTree; ++jTree) {
      const TString newName(*(allTreeNames[jTree]) + treeNameAdd);
// either by really renaming trees: 
//       TTree *tree = NULL;
//       file->GetObject(allTreeNames[jTree]->Data(), tree);
//       if (!tree) {
//         ::Error("MillePedeTrees::CreateTree",
//                 "no tree %s in %s", allTreeNames[jTree]->Data(), fileName);
//       } else {
//         tree->SetName(newName);
//         mainTree->AddFriend(tree, "", true); // no alias, but warn if different lengths
//       }
// or by setting an alias:
      TFriendElement *fEle = mainTree->AddFriend(newName + " = " + *(allTreeNames[jTree]));
      if (!fEle || !fEle->GetTree()) {
        ::Error("MillePedeTrees::CreateTree","no %s as friend tree",allTreeNames[jTree]->Data());
       } 
      *(allTreeNames[jTree]) = newName; // Yes, we really change the data members!
    }
    mainTree->SetEstimate(mainTree->GetEntries()); // for secure use of GetV1() etc.  
  }

  return mainTree;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TH1* MillePedeTrees::Draw(const char* exp, const char* selection, const char *hDef, Option_t* opt)
{

  TString def(hDef);
  if (def.Length()) def.Prepend(">>");

  fTree->Draw(exp + def, selection, opt);

  return fTree->GetHistogram();
//   TH1 *result = NULL;
//   const TString name(histDef.Remove(histDef.First('(')));
//   gDirectory->GetObject(name, result);
//   return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TH1 *MillePedeTrees::CreateHist(const char *exp, const char *selection, const char *hDef,
                                Option_t* opt)
{
  TH1 *h = this->Draw(exp, selection, hDef, "goff");

  TH1 *hResult = static_cast<TH1*>(h->Clone(Form("%sC", h->GetName())));
  if (opt) hResult->SetOption(opt);

  return hResult;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
TProfile *MillePedeTrees::CreateHistProf(const char *expX, const char *expY, const char *selection,
                                         const char *hDef, Option_t* opt)
{

  const TString combExpr(Form("%s:%s", expY, expX));
  TH1 *h = this->Draw(combExpr, selection, hDef, "goff prof");

  TProfile *hResult = static_cast<TProfile*>(h->Clone(Form("%sClone", h->GetName())));
  if (opt) hResult->SetOption(opt);

  return hResult;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TH2 *MillePedeTrees::CreateHist2D(const char *expX, const char *expY, const char *selection,
                                  const char *hDef, Option_t* opt)
{
  const TString combExpr(Form("%s:%s", expY, expX));
  TH1 *h = this->CreateHist(combExpr, selection, hDef, opt);

  return static_cast<TH2*>(h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TGraph *MillePedeTrees::CreateGraph(const char *expX, const char *expY, const char *sel, Option_t *)
{
  TH2 *h = this->CreateHist2D(expX, expY, sel, NULL, "goff");
  if (!h) return NULL;

  ::Error("MillePedeTrees::CreateGraph", "Not yet implemented.");

  return NULL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// void MillePedeTrees::ScanDiffAbove(UInt_t iPar, float absDiff)
// {
//
//   TString sel(//Fixed(iPar, false) += AndL() += 
//               Abs(DiffPar(ParT(), MisParT(), iPar)) += Form(">%f", absDiff));
//   cout << "MillePedeTrees::ScanDiffAbove: " << sel << std::endl;
// //   fTree->Scan(MpT() += "Label:" + OrgPosT() += XPos() += ":" + OrgPosT() += YPos(), sel);
//   fTree->Scan("", MpT() += "Label");
//
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::RPos2(const TString &tree) const
{

  const TString x(tree + XPos());
  const TString y(tree + YPos());

  return Parenth((x + Mal() += x) + Plu() += (y + Mal() += y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::RPos(const TString &tree) const
{
  const TString r(Sqrt(RPos2(tree)));

  if (fUseSignedR) {
    const TString y(tree + Pos(2));
    return r + Mal() += Parenth(Fun("TMath::Abs", y) += Div() += y);
  } else {
    return r;
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Phi(const TString &tree) const
{
  return Fun("TMath::ATan2", tree + YPos() += "," + tree + XPos());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::OrgPos(const TString &pos) const
{
 // pos x, y, z, rphi, phi,... on original position
  const TString tree(OrgPosT());
  if (pos == "x") {
    return tree + Pos(0);
  } else if (pos == "y") {
    return tree + Pos(1);
  } else if (pos == "z") {
    return tree + Pos(2);
  } else if (pos == "r") {
    return RPos(tree);
  } else if (pos == "phi") {
    return Phi(tree);
  } else {
    ::Error("MillePedeTrees::OrgPos", "unknown position %s, try x,y,z,r,phi", pos.Data());
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::PhiSwaps(double swapAround, const TString &tree1, const TString &tree2) const
{
  // 'true'/1 if one phi is above, one below 'swapAround'
  return 
    Parenth((Phi(tree1) += Form(">%f", swapAround)) += 
            AndL() += Phi(tree2) += Form("<%f", swapAround))
    += OrL()
    += Parenth((Phi(tree1) += Form("<%f", swapAround)) += 
               AndL() += Phi(tree2) += Form(">%f", swapAround));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Theta(const TString &tree) const
{

  // protect against possible sign in RPos by using Sqrt(RPos2(tree))
  return Fun("TMath::ATan2", Sqrt(RPos2(tree)) += "," + tree + ZPos()) // theta, cf. TVector3::Theta
    += "*(" + XPos() += "!=0.&&" + YPos() += "!=0.&&" + ZPos() += "!=0.)"; // guard against |p|=0
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Alpha(const TString &tree, bool betaMpiPpi) const
{
  // a la AlignmentTransformations::eulerAngles

//   double rot[3][3];
//   rot[0][0] = Rot[0]; // Rot from tree
//   rot[0][1] = Rot[1];
//   rot[0][2] = Rot[2];
//   rot[1][0] = Rot[3];
//   rot[1][1] = Rot[4];
//   rot[1][2] = Rot[5];
//   rot[2][0] = Rot[6];
//   rot[2][1] = Rot[7];
//   rot[2][2] = Rot[8];

  TString euler1("TMath::ASin(Rot[6])");
  if (!betaMpiPpi) euler1.Prepend("TMath::Pi() - ");

  TString euler0("TMath::ATan(-Rot[7]/Rot[8]) + (TMath::Pi() * (TMath::Cos(");
  euler0 += euler1;
  euler0 += ") * Rot[8] <= 0))";

  TString result(Form("(TMath::Abs(Rot[6] - 1.0) >= 1.e-6) * (%s)", euler0.Data()));
  result.ReplaceAll("Rot[", tree + "Rot[");

  return result;

//   if (TMath::Abs(Rot[6] - 1.0) > 1.e-6) { // If angle1 is not +-PI/2

//       if (flag == 0) // assuming -PI/2 < angle1 < PI/2 
//         euler[1] = TMath::ASin(Rot[6]); // New beta sign convention
//       else // assuming angle1 < -PI/2 or angle1 >PI/2
//         euler[1] = TMath::Pi() - TMath::ASin(Rot[6]); // New beta sign convention
      
//       if (TMath::Cos(euler[1]) * Rot[8] > 0)
//         euler[0] = TMath::ATan(-Rot[7]/Rot[8]);
//       else
//         euler[0] = TMath::ATan(-Rot[7]/Rot[8]) + TMath::Pi();
      
//       if (TMath::Cos(euler[1]) * Rot[0] > 0)
//         euler[2] = TMath::ATan(-Rot[3]/Rot[0]);
//       else
//         euler[2] = TMath::ATan(-Rot[3]/Rot[0]) + TMath::Pi();
//   } else { // if angle1 == +-PI/2
//     euler[1] = TMath::PiOver2(); // chose positve Solution 
//     if(Rot[8] > 0) {
//       euler[2] = TMath::ATan(Rot[5]/Rot[4]);
//       euler[0] = 0;
//     }
//   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Beta(const TString &tree, bool betaMpiPpi) const
{
  TString euler1("TMath::ASin(Rot[6])");
  if (!betaMpiPpi) euler1.Prepend("TMath::Pi() - ");

  TString result(Form("(TMath::Abs(Rot[6] - 1.0) >= 1.e-6) * (%s)", euler1.Data()));
  result += " + (TMath::Abs(Rot[6] - 1.0) < 1.e-6) * TMath::PiOver2()"; // choose positive sol.

  result.ReplaceAll("Rot[", tree + "Rot[");
  return result;

  /*
  if (TMath::Abs(Rot[6] - 1.0) > 1.e-6) { // If angle1 is not +-PI/2

      if (flag == 0) // assuming -PI/2 < angle1 < PI/2 
        euler[1] = TMath::ASin(Rot[6]); // New beta sign convention
      else // assuming angle1 < -PI/2 or angle1 >PI/2
        euler[1] = TMath::Pi() - TMath::ASin(Rot[6]); // New beta sign convention
      
      if (TMath::Cos(euler[1]) * Rot[8] > 0)
        euler[0] = TMath::ATan(-Rot[7]/Rot[8]);
      else
        euler[0] = TMath::ATan(-Rot[7]/Rot[8]) + TMath::Pi();
      
      if (TMath::Cos(euler[1]) * Rot[0] > 0)
        euler[2] = TMath::ATan(-Rot[3]/Rot[0]);
      else
        euler[2] = TMath::ATan(-Rot[3]/Rot[0]) + TMath::Pi();
  } else { // if angle1 == +-PI/2
    euler[1] = TMath::PiOver2(); // chose positve Solution 
    if(Rot[8] > 0) {
      euler[2] = TMath::ATan(Rot[5]/Rot[4]);
      euler[0] = 0;
    }
  }
  */
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Gamma(const TString &tree, bool betaMpiPpi) const
{
  TString euler1("TMath::ASin(Rot[6])");
  if (!betaMpiPpi) euler1.Prepend("TMath::Pi() - ");

  TString euler2("TMath::ATan(-Rot[3]/Rot[0]) + (TMath::Pi() * (TMath::Cos(");
  euler2 += euler1;
  euler2 += ") * Rot[0] <= 0))";

  TString result(Form("(TMath::Abs(Rot[6] - 1.0) >= 1.e-6) * (%s)", euler2.Data()));
  result += "+ (TMath::Abs(Rot[6] - 1.0) < 1.e-6) * (TMath::ATan(Rot[5]/Rot[4]))";
  result.ReplaceAll("Rot[", tree + "Rot[");

  return result;

  /*
  if (TMath::Abs(Rot[6] - 1.0) > 1.e-6) { // If angle1 is not +-PI/2

      if (flag == 0) // assuming -PI/2 < angle1 < PI/2 
        euler[1] = TMath::ASin(Rot[6]); // New beta sign convention
      else // assuming angle1 < -PI/2 or angle1 >PI/2
        euler[1] = TMath::Pi() - TMath::ASin(Rot[6]); // New beta sign convention
      
      if (TMath::Cos(euler[1]) * Rot[8] > 0)
        euler[0] = TMath::ATan(-Rot[7]/Rot[8]);
      else
        euler[0] = TMath::ATan(-Rot[7]/Rot[8]) + TMath::Pi();
      
      if (TMath::Cos(euler[1]) * Rot[0] > 0)
        euler[2] = TMath::ATan(-Rot[3]/Rot[0]);
      else
        euler[2] = TMath::ATan(-Rot[3]/Rot[0]) + TMath::Pi();
  } else { // if angle1 == +-PI/2
    euler[1] = TMath::PiOver2(); // chose positve Solution 
    if(Rot[8] > 0) {
      euler[2] = TMath::ATan(Rot[5]/Rot[4]);
      euler[0] = 0;
    }
  }
  */
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::ParSi(const TString &tree, UInt_t iParam) const
{
  TString index(Form("%sparSize*%d", tree.Data(), iParam));
  if (iParam > 1) {
    UInt_t aParNum = 1;
    UInt_t reducer = 0;
    while (aParNum < iParam) {
      reducer += aParNum;
      ++aParNum;
    }
    index += Form("-%d", reducer);
  }

  return Sqrt((tree + "Cov") += Bracket(index));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::DelPos(UInt_t ui, const TString &tree1, const TString &tree2) const
{
  return tree1 + Pos(ui) += (Min() += tree2) += Pos(ui);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::DelR(const TString &tree1, const TString &tree2) const
{
  return RPos(tree1) += Min() += RPos(tree2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::DelRphi(const TString &tree1, const TString &tree2) const
{
  // distance vector in rphi/xy plane times unit vector e_phi = (-y/r, x/r)
  // tree2 gives reference for e_phi 
  const TString deltaX = Parenth(tree1 + XPos() += (Min() += tree2) += XPos());
  const TString deltaY = Parenth(tree1 + YPos() += (Min() += tree2) += YPos());
  // (delta_x * (-y) + delta_y * x) / r
  // protect against possible sign of RPos:
  return Parenth(Parenth(deltaX + Mal() += ("-" + tree2) += YPos()
			 += Plu() += deltaY + Mal() += tree2 + XPos()
			 ) += Div() += Sqrt(RPos2(tree2)) //RPos(tree2)
		 );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TString MillePedeTrees::DelRphi_b(const TString &tree1, const TString &tree2) const
// {
//   // distance vector in rphi/xy plane times unit vector e_phi = (-sin phi, cos phi)
//   // tree2 gives reference for e_phi 
//   const TString deltaX = Parenth(tree1 + XPos() += (Min() += tree2) += XPos());
//   const TString deltaY = Parenth(tree1 + YPos() += (Min() += tree2) += YPos());
//
//   return Parenth(deltaX + Mal() 
// 		 += Fun("-TMath::Sin", Phi(tree2))
// 		 += Plu() 
// 		 += deltaY + Mal() 
// 		 += Fun("TMath::Cos", Phi(tree2)));
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::DelPhi(const TString &tree1, const TString &tree2) const
{
  return Fun("TVector2::Phi_mpi_pi", Phi(tree1) += Min() += Phi(tree2));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Valid(UInt_t iParam) const
{
  return (MpT() += "IsValid") += Bracket(iParam);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Fixed(UInt_t iParam, bool isFixed) const
{
  return (isFixed ? Parenth(PreSi(iParam) += "<0.") : "!" + Parenth(PreSi(iParam) += "<0."));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::AnyFreePar() const
{
  TString result("(");
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { 
    result += Fixed(iPar, false);
    if (iPar != kNpar - 1) result += OrL();
    else result += ")";
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Label(UInt_t iParam) const
{
//   return (MpT() += "Label") += Bracket(iParam);
  return Parenth((MpT() += "Label + ") += Int(iParam)); 
}


////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Cor(UInt_t iParam) const
{
  return (MpT() += "GlobalCor") += Bracket(iParam);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Diff(UInt_t iParam) const
{
  return (MpT() += "DiffBefore") += Bracket(iParam);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::PreSi(UInt_t iParam) const
{
  return (MpT() += "PreSigma") += Bracket(iParam);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::ParSi(UInt_t iParam) const
{
  return (MpT() += "Sigma") += Bracket(iParam);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::ParSiOk(UInt_t iParam) const
{
  // cf. default value in MillePedeVariables::setAllDefault
  return Parenth(ParSi(iParam) += " != -1.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Name(UInt_t iParam) const
{
  switch (iParam) {
  case 0: return "u";
  case 1: return "v";
  case 2: return "w";
  case 3: return "#alpha";
  case 4: return "#beta";
  case 5: return "#gamma";
  default:
    ::Error("MillePedeTrees::Name", "unknown parameter %d", iParam);
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::ToMumMuRad(const TString &pos) const
{
  if (pos == "r" || pos == "rphi" || pos == "x" || pos == "y" || pos == "z") {
    return "*10000"; // cms to mum
  } else if (pos == "phi") {
    return "*1000000"; // rad to murad
  } else {
    ::Error("MillePedeTrees::ToMumMuRad", "unknown position %s", pos.Data());
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Name(const TString &pos) const
{
  if (pos == "r" || pos == "x" || pos == "y" || pos == "z") {
    return pos;
  } else if (pos == "phi") {
    return "#phi";
  } else if (pos == "rphi") {
    return "r#phi";
  } else {
    ::Error("MillePedeTrees::Name", "unknown position %s", pos.Data());
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::DelName(const TString &pos) const
{
  if (pos == "rphi") {
    return "r#Delta#phi";
  } else {
    return "#Delta"+Name(pos);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::Unit(const TString &pos) const
{
  if (pos == "r" || pos == "rphi" || pos == "x" || pos == "y" || pos == "z") {
    return " [#mum]";
  } else if (pos == "phi") {
    return " [#murad]";
  } else {
    ::Error("MillePedeTrees::Unit", "unknown position %s", pos.Data());
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::DeltaPos(const TString &pos, const TString &tree) const
{
  if (pos == "r") {
    return DelR(tree);
  } else if(pos == "rphi") {
    return DelRphi(tree);
  } else if(pos == "x") {
    return DelPos(0, tree);
  } else if(pos == "y") {
    return DelPos(1, tree);
  } else if(pos == "z") {
    return DelPos(2, tree);
  } else if (pos == "phi") {
    return DelPhi(tree);
  } else {
    ::Error("MillePedeTrees::Delta", "unknown position %s", pos.Data());
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::SelIs1D() const
{
  return Parenth("!" + SelIs2D());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString MillePedeTrees::SelIs2D() const
{
  // currently check simply radius
  // protect against possible sign of RPos:
  const TString r(Sqrt(RPos2(OrgPosT()))); // RPos(OrgPosT()));
  //  return Parenth((r + "<40 || (") += (r + ">60 && ") += (r + "<75)"));
  return Parenth((r + "<40 || (") += (r + ">57.5 && ") += (r + "<75)"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TString MillePedeTrees::RemoveHistName(TString &option) const
// {
//   // Removing histname from option (as usable in tree->Draw("...>>histname",...))
//   // and returning it. The histname is dentified by what is behind the last ';',
//   // e.g. option = "<anOption>;hist(100, -200, 200)"
//   // If there is no ';', no default histname is returned
//  
//   const Ssiz_t token = option.Last(';');
//   if (token == kNPOS) {
//     return "";
//   } else {
//     const TString result(option(token+1, option.Length()-token-1));
//     option.Remove(token);
// //     return ">>" + result;
//     return result;
//   }
// }
