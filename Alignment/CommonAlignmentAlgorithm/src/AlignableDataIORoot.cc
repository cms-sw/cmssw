#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableDataIORoot.h"

// ----------------------------------------------------------------------------
// constructor
AlignableDataIORoot::AlignableDataIORoot(PosType p) : 
  AlignableDataIO(p)
{
  if (thePosType == Abs) {
    treename = "AlignablesAbsPos";
    treetxt = "Alignables abs.Pos";
  }
  else if (thePosType == Org) {
    treename = "AlignablesOrgPos";
    treetxt = "Alignables org.Pos";
  }
  else if (thePosType == Rel) {
    treename = "AlignablesRelPos";
    treetxt = "Alignables rel.Pos";
  }
}

// ----------------------------------------------------------------------------
// create root tree branches (for writing)

void AlignableDataIORoot::createBranches(void) 
{
  tree->Branch("Id",    &Id,    "Id/I");
  tree->Branch("ObjId", &ObjId, "ObjId/I");
  tree->Branch("Pos",   &Pos,   "Pos[3]/D");
  tree->Branch("Rot",   &Rot,   "Rot[9]/D");
}

// ----------------------------------------------------------------------------
// set root tree branch addresses (for reading)

void AlignableDataIORoot::setBranchAddresses(void) 
{
  tree->SetBranchAddress("Id",    &Id);
  tree->SetBranchAddress("ObjId", &ObjId);
  tree->SetBranchAddress("Pos",   &Pos);
  tree->SetBranchAddress("Rot",   &Rot);
}

// ----------------------------------------------------------------------------
// find root tree entry based on IDs

int AlignableDataIORoot::findEntry(unsigned int detId,int comp)
{
  if (newopen) { // we're here first time
    edm::LogInfo("Alignment") << "@SUB=AlignableDataIORoot::findEntry"
                              << "Filling map ...";
    treemap.erase(treemap.begin(),treemap.end());
    for (int ev = 0;ev<tree->GetEntries();ev++) {
      tree->GetEntry(ev); 
      treemap[ std::make_pair(Id,ObjId) ] = ev;
    }
    newopen=false;
  }
  
  // now we have filled the map
  treemaptype::iterator imap = treemap.find( std::make_pair(detId,comp) );
  int result=-1;
  if (imap != treemap.end()) result=(*imap).second;
  return result;

}

// ----------------------------------------------------------------------------
int AlignableDataIORoot::writeAbsRaw(const AlignableAbsData &ad)
{
  GlobalPoint pos = ad.pos();
  Surface::RotationType rot = ad.rot();
  Id = ad.id();
  ObjId = ad.objId();
  Pos[0]=pos.x(); Pos[1]=pos.y(); Pos[2]=pos.z();
  Rot[0]=rot.xx(); Rot[1]=rot.xy(); Rot[2]=rot.xz();
  Rot[3]=rot.yx(); Rot[4]=rot.yy(); Rot[5]=rot.yz();
  Rot[6]=rot.zx(); Rot[7]=rot.zy(); Rot[8]=rot.zz();
  tree->Fill();
  return 0;
}

// ----------------------------------------------------------------------------
int AlignableDataIORoot::writeRelRaw(const AlignableRelData &ad)
{
  GlobalVector pos = ad.pos();
  Surface::RotationType rot = ad.rot();
  Id = ad.id();
  ObjId = ad.objId();
  Pos[0]=pos.x(); Pos[1]=pos.y(); Pos[2]=pos.z();
  Rot[0]=rot.xx(); Rot[1]=rot.xy(); Rot[2]=rot.xz();
  Rot[3]=rot.yx(); Rot[4]=rot.yy(); Rot[5]=rot.yz();
  Rot[6]=rot.zx(); Rot[7]=rot.zy(); Rot[8]=rot.zz();
  tree->Fill();
  return 0;
}

// ----------------------------------------------------------------------------
AlignableAbsData AlignableDataIORoot::readAbsRaw(Alignable* ali,int& ierr)
{
  Global3DPoint pos;
  Surface::RotationType rot;

  TrackerAlignableId converter;
  int typeId = converter.alignableTypeId(ali);
  unsigned int id = converter.alignableId(ali);
  int entry = findEntry(id,typeId);
  if(entry!=-1) {
    tree->GetEntry(entry);
    Global3DPoint pos2(Pos[0],Pos[1],Pos[2]);
    Surface::RotationType rot2(Rot[0],Rot[1],Rot[2],
                               Rot[3],Rot[4],Rot[5],
							   Rot[6],Rot[7],Rot[8]);
    pos=pos2;
    rot=rot2;
    ierr=0;
  }
  else ierr=-1;

  return AlignableAbsData(pos,rot,id,typeId);
}

// ----------------------------------------------------------------------------

AlignableRelData AlignableDataIORoot::readRelRaw(Alignable* ali,int& ierr)
{
  Global3DVector pos;
  Surface::RotationType rot;

  TrackerAlignableId converter;
  int typeId = converter.alignableTypeId(ali);
  unsigned int id = converter.alignableId(ali);
  int entry = findEntry(id,typeId);
  if(entry!=-1) {
    tree->GetEntry(entry);
    Global3DVector pos2(Pos[0],Pos[1],Pos[2]);
    Surface::RotationType rot2(Rot[0],Rot[1],Rot[2],
                               Rot[3],Rot[4],Rot[5],
	                       Rot[6],Rot[7],Rot[8]);
    pos=pos2;
    rot=rot2;
    ierr=0;
  }
  else ierr=-1;

  return AlignableRelData(pos,rot,id,typeId);
}
