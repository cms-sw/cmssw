#ifndef PLOTMILLEPEDEIOV_H
#define PLOTMILLEPEDEIOV_H
// Original Author: Gero Flucke
// last change    : $Date: 2012/06/25 13:21:53 $
// by             : $Author: flucke $
//
// PlotMillePedeIOV is a class to plot the IOV dependence of pede parameters
// in case of run-dependend alignment, e.g. for large pixel structures.
// It makes internal use of PlotMillePede and also interfaces
// various selection settings of that class, see PlotMillePede.h.
// (Missing selection possibilities should be simple to add...)
//
// Management of the created histograms is done using the ('ancient')
// class GFHistManager, see description in PlotMillePede.h.
//
// By calling new PlotMillePedeIOV::Draw..(..) commands, usually previously
// drawn canvases are deleted. But if the option 'add' is given, canvases
// from previous Draw-commands are kept (in fact they are re-drawn).
//
// EXAMPLE I: full scale alignment with time dependent large pixel structures
// 
// PlotMillePedeIOV i("treeFile_merge.root");
// i.SetSubDetId(1);      // Select BPIX.
// i.DrawPedeParam("x");
// i.SetSubDetId(2);      // Select FPIX...
// i.AddAdditionalSel("z", 0,100); // ...but only positive z.
// i.DrawPedeParam("add x z");
// i.ClearAdditionalSel();           // Remove selection on z >= 0... 
// i.AddAdditionalSel("z", -100, 0); // ... and request negative z.
// i.DrawPedeParam("add x z");
//
//
// EXAMPLE II: time dependent alignment of large pixel structures
//             using inversion (i.e. errors available), no hierarchy involved
// 
// PlotMillePedeIOV i("treeFile_merge.root");
// i.SetHieraLevel(0); // no hierarchy => lowest level (default is 1, see ctr.)
// i.SetSubDetId(1);      // Select BPIX.
// i.DrawPedeParam("x");
// i.DrawPedeParam("x val"); // same but without error bars
// i.DrawPedeParam("x err add"); // now draw errors vs IOV
// i.SetSubDetId(2);      // Select FPIX...
// i.DrawPedeParam("x val add");
// i.DrawPedeParam("x err add");

#include <vector>

class GFHistManager;
class PlotMillePede;

class PlotMillePedeIOV 
{
 public:
  explicit PlotMillePedeIOV(const char *fileName, Int_t maxIov = -1, Int_t hieraLevel = 1); // maxIov <=0: find out from file!; hieraLev: -1 ignore, 0 lowest level, etc.
  virtual ~PlotMillePedeIOV();

  void DrawPedeParam(Option_t *option = "", unsigned int nNonRigidParam = 0);// "add", any of "x","y","z","id" to add position or DetId in legend, "err" error (not value), "val" skip error bar even if valid 

  void SetTitle(const char *title) {fTitle = title;}
  const TString& GetTitle() const { return fTitle;}
  GFHistManager* GetHistManager() { return fHistManager;}
  PlotMillePede* GetPlotMillePede(unsigned int i) { return (i < fIovs.size() ? fIovs[i] : 0);}


  TString Unique(const char *name) const;
  Int_t PrepareAdd(bool addPlots);
  template<class T>
    void SetLineMarkerStyle(T &object, Int_t num) const;
  
  void SetSubDetId(Int_t subDet);
  void SetSubDetIds(Int_t id1, Int_t id2, Int_t id3 = -1, Int_t id4 = -1, Int_t id5 = -1); // ignores id<n> <= 0
  void SetAlignableTypeId(Int_t alignableTypeId);//detunit=1,det=2,...,TIBString=15,etc. from StructureType.h (-1: all)
  void SetHieraLevel(Int_t hieraLevel); // select hierarchical level (-1: all)
  void SetBowsParameters(bool use = true);//true: bows param. for pede
  void AddAdditionalSel(const TString &xyzrPhiNhit, Float_t min, Float_t max); // min <= x,y,z,r,phi,Nhit < max
  void ClearAdditionalSel();

  struct ParId {
    //parameter identified by id (=DetId), objId (=hieraLevel), parameter
  public:
    ParId(Int_t id, Int_t objId, Int_t par) :
      id_(id), objId_(objId), par_(par) {};
    Int_t id_, objId_, par_;

    bool operator< (const ParId& other) const; // needed for use as Key in std::map
  };
  // end struct ParId

 private:
  GFHistManager *fHistManager;
  std::vector<PlotMillePede*> fIovs;
  TString fTitle;
};

#endif
