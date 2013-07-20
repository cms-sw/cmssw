// Original Author: Gero Flucke
// last change    : $Date: 2013/03/07 11:27:40 $
// by             : $Author: flucke $

#ifndef MILLEPEDETREES_H
#define MILLEPEDETREES_H

#include "TString.h"

class TTree;
class TGraph;
class TH1;
class TH2;
class TProfile;

class MillePedeTrees
{
 public:
  explicit MillePedeTrees(const char *fileName, Int_t iter = 1, const char *treeNameAdd = "");
  virtual ~MillePedeTrees();

  TH1* Draw(const char *exp, const char *selection, const char *hDef = "", Option_t *opt = "");
  TH1* CreateHist(const char *exp, const char *select, const char *hDef = "", Option_t *opt = "");
  TH2* CreateHist2D(const char *expX, const char *expY, const char *select,
		    const char *hDef = "", Option_t *opt = "");
  TProfile* CreateHistProf(const char *expX, const char *expY, const char *select,
			   const char *hDef = "", Option_t *opt = "");
  TGraph* CreateGraph(const char *expX, const char *expY, const char *select, Option_t *option="");

/*   void ScanDiffAbove(UInt_t iPar, float absDiff); */

  // values: absolute positions and orientations; need a 'position tree' as argument or in front
  TString Pos() const { return "Pos";}
  TString Pos(UInt_t ui) const { return Pos() += Bracket(ui);}
  TString XPos() const { return Pos(0);}
  TString YPos() const { return Pos(1);}
  TString ZPos() const { return Pos(2);}
  TString RPos2(const TString &tree) const; // x^2+y^2
/*   TString RPos(const TString &tree) const { return Sqrt(RPos2(tree));} // sqrt(x^2+y^2) */
  TString RPos(const TString &tree) const; // sqrt(x^2+y^2) - y-sign if selected by UseSignedR()
  // (un)set whether RPos should be signed like y, return old setting
  bool SetUseSignedR(bool use = true) {bool buf = fUseSignedR; fUseSignedR = use; return buf;}//true: radius gets sign of y
  bool SetBowsParameters(bool use = true) {bool buf = fBowsParameters; fBowsParameters = use; return buf;}//true: bows param. for pede
  bool SetSurfDefDeltaBows(bool deltaBows) {const bool buf = fSurfDefDeltaBows; fSurfDefDeltaBows = deltaBows; return buf;} // take care: problems for false if drawing 1-sensor modules!
  
  TString Phi(const TString &tree) const;
  TString OrgPos(const TString &pos) const; // pos x, y, z, r, phi,... on original position

  TString PhiSwaps(double swapAround, const TString &tree1, const TString &tree2) const;
  TString Theta(const TString &tree) const;
  TString Alpha(const TString &tree, bool betaMpiPpi) const;
  TString Beta (const TString &tree, bool betaMpiPpi) const;
  TString Gamma(const TString &tree, bool betaMpiPpi) const;

  // values: alignment parameters; need a 'parameter tree' or MpT() as argument or in front
  TString Par() const { return "Par";}
  TString Par(UInt_t ui) const { return Par() += Bracket(ui);}
  TString XPar() const { return Par(0);}
  TString YPar() const { return Par(1);}
  TString ZPar() const { return Par(2);}
  TString Alpha() const { return Par(3);}
  TString Beta() const { return Par(4);}
  TString Gamma() const { return Par(5);}
  TString DiffPar(const TString &t1, const TString &t2, UInt_t iPar) const {
    return Parenth(t1 + Par(iPar) += Min() += t2 + Par(iPar));}
  // values: alignment parameter errors from parameter tree (to be given as tree name)
  TString ParSi(const TString &tree, UInt_t ui) const;
  TString XParSi(const TString &tree) const  { return ParSi(tree, 0);}
  TString YParSi(const TString &tree) const { return ParSi(tree, 1);}
  TString ZParSi(const TString &tree) const { return ParSi(tree, 2);}
  TString AlphaSi(const TString &tree) const { return ParSi(tree, 3);}
  TString BetaSi(const TString &tree) const { return ParSi(tree, 4);}
  TString GammaSi(const TString &tree) const { return ParSi(tree, 5);}

  // values: Delta positions
  TString DelPos(UInt_t ui, const TString &tree1, const TString &tree2) const;
  TString DelR(const TString &tree1, const TString &tree2) const;
  TString DelRphi(const TString &tree1, const TString &tree2) const;
  // TString DelRphi_b(const TString &tree1, const TString &tree2) const; // version b
  TString DelPhi(const TString &tree1, const TString &tree2) const;
  // values: Delta positions wrt. OrgPosT()
  TString DelPos(UInt_t ui, const TString &tree) const { return DelPos(ui, tree, OrgPosT());}
  TString DelR(const TString &tree) const { return DelR(tree, OrgPosT());}
  TString DelRphi(const TString &tree) const { return DelRphi(tree, OrgPosT());}
  TString DelPhi(const TString &tree) const { return DelPhi(tree, OrgPosT());}
  // see also TString DeltaPos(const TString &pos, const TString &tree) const;
  
  // '25' and '0x7' DataFormats/DetId/interface/DetId.h:
  TString SubDetId() const { return "(" + OrgPosT() += "Id>>25)&0x7";}
  TString AlignableTypeId() const { return OrgPosT() += "ObjId";}
  TString HieraLev(const TString &tree, Int_t level) const {
    return Parenth(tree + "HieraLevel==" + Int(level));}
  TString HieraLev(Int_t level) const {return HieraLev(ParT(), level);}
  // values: from pede, do not add tree in front
  TString Valid(UInt_t iParam) const;
  TString Fixed(UInt_t iParam, bool isFixed = true) const;
  TString AnyFreePar() const;
  TString Label(UInt_t iParam) const;
  TString Cor(UInt_t iParam) const;
  TString Diff(UInt_t iParam) const;
  TString PreSi(UInt_t iParam) const;
  TString ParSi(UInt_t iParam) const;
  TString ParSiOk(UInt_t iParam) const;
  TString XParSi() const  { return ParSi(0);}
  TString YParSi() const { return ParSi(1);}
  TString ZParSi() const { return ParSi(2);}
  TString AlphaSi() const { return ParSi(3);}
  TString BetaSi() const { return ParSi(4);}
  TString GammaSi() const { return ParSi(5);}
  TString HitsX() const { return MpT() += "HitsX";}
  TString HitsY() const { return MpT() += "HitsY";}
  TString DeformValue(UInt_t i, const TString &whichOne) const;//start,result,diff
  TString NumDeformValues(const TString &whichOne) const; //start,result,diff

  // symbols
  TString Dot() const { return ".";}
  TString Plu() const { return "+";}
  TString Min() const { return "-";}
  TString Mal() const { return "*";} // fixme? german...
  TString Div() const { return "/";}
  TString AndL() const {return "&&";} // logical and (not '&')
  TString OrL() const {return "||";}  // logical or (not '|')
  // numbers
  TString Int(Int_t i) const { return Form("%d", i);}
  TString Int(UInt_t ui) const { return Form("%u", ui);}
  TString Flt(Float_t f) const { return Form("%f", f);}
  // brackets and parentheses
  TString Bra() const { return "[";}
  TString Ket() const { return "]";}
  TString Bracket(UInt_t ui) const { return Bra() += Int(ui) += Ket();}
  TString Bracket(Int_t i) const { return Bra() += Int(i) += Ket();}
  TString Bracket(const char *s) const { return (Bra() += s) += Ket();}
  TString Paren() const { return "(";}
  TString Thesis() const { return ")";}
  TString Parenth(const char *s) const { return (Paren() += s) += Thesis();}
  TString Abs(const char *s) const {return Fun("TMath::Abs", s);}
  // functions
  TString Fun(const char *fun, const char *s) const { return fun + Parenth(s);}
  TString Sqrt(const char *s) const { return Fun("sqrt", s);}

  // units and names for params
  TString ToMumMuRad(UInt_t iParam) const { return (iParam < 3 ? "*10000" : "*1000000");}
  TString ToMumMuRadPede(UInt_t iParam) const;
  TString ToMumMuRadSurfDef(UInt_t iParam) const;
  TString Name(UInt_t iParam) const;
  TString NamePede(UInt_t iParam) const;
  TString NameSurfDef(UInt_t iParam) const;
  TString DelName(UInt_t iParam) const { return "#Delta"+Name(iParam);}
  TString DelNameU(UInt_t iParam) const { return DelName(iParam) += Unit(iParam);}
  TString Unit(UInt_t iParam) const { return (iParam < 3 
					      ? " [#mum]" 
					      : (iParam < kNpar ? " [#murad]" : ""));}
  TString UnitPede(UInt_t iParam) const;
  TString UnitSurfDef(UInt_t iParam) const;
  // units and names for position strings (r, rphi, phi, x, y, z)
  TString ToMumMuRad(const TString &pos) const;
  TString Name(const TString &pos) const;
  TString NamePos(UInt_t iPos) const; //0-2 are x, y, z
  TString DelName(const TString &pos) const;// { return "#Delta"+Name(pos);}
  TString DelNameU(const TString &pos) const { return DelName(pos) += Unit(pos);}
  TString Unit(const TString &pos) const;

  TString DeltaPos(const TString &pos, const TString &tree /* = PosT()*/) const;// delta position to OrgPosT, depending on pos DelPhi,DelRphi,DelPos(0,..) etc.
  TString SelIs1D() const;
  TString SelIs2D() const;

  // tree names
  TString OrgPosT() const { return fOrgPos + Dot();} // nominal global positions
  TString MisPosT() const { return fMisPos + Dot();} // misaligned global positions
  TString MisParT() const { return fMisPar + Dot();} // misalignment
  TString PosT() const { return fPos + Dot();} // aligned global positions
  TString ParT() const { return fPar + Dot();} // remaining misalignment
  TString MpT() const { return fMp + Dot();} // MP tree (parameter, hits,...)

  TTree* GetMainTree() {return fTree;} // use with care...

  enum {kLocX = 0, kLocY, kLocZ, kNpar = 6}; // number of parameters we have...
 protected:

 private: 
  MillePedeTrees();
  // utils
  TTree* CreateTree(const char *fileName, const TString &treeNameAdd);
/*   TString RemoveHistName(TString &option) const; */

  // data members
  TTree   *fTree;

  // tree names
  TString fOrgPos;    // absolute original positions from xml/input DB
  TString fMisPos;    // absolute positions with misalignment applied
  TString fMisPar;    // misalignment parameters
  TString fPos;       // positions after alignment
  TString fPar;       // remaining misalign paramters after alignment (was: alignment parameters)
  TString fMp;        // specials for/from MillePede

  // special seetings
  bool fUseSignedR;  // if true, Rpos will have sign of y
  bool fBowsParameters; //true: pede parameter names and titles to 'bows', false: rigid body
  bool fSurfDefDeltaBows; // true: SurfaceDeformation values as is, otherwise bowMean+Delta and bowMean-Delta
};
#endif
