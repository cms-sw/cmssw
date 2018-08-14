#include "TTree.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  tree->Branch("Id",    &Id,    "Id/i");
  tree->Branch("ObjId", &ObjId, "ObjId/I");
  tree->Branch("Pos",   &Pos,   "Pos[3]/D");
  tree->Branch("Rot",   &Rot,   "Rot[9]/D");

  tree->Branch("NumDeform",   &numDeformationValues_, "NumDeform/i");
  tree->Branch("DeformValues", deformationValues_,    "DeformValues[NumDeform]/F");
}

// ----------------------------------------------------------------------------
// set root tree branch addresses (for reading)

void AlignableDataIORoot::setBranchAddresses(void) 
{
  tree->SetBranchAddress("Id",    &Id);
  tree->SetBranchAddress("ObjId", &ObjId);
  tree->SetBranchAddress("Pos",   &Pos);
  tree->SetBranchAddress("Rot",   &Rot);

  tree->SetBranchAddress("NumDeform",   &numDeformationValues_);
  tree->SetBranchAddress("DeformValues", deformationValues_);
}

// ----------------------------------------------------------------------------
// find root tree entry based on IDs

int AlignableDataIORoot::findEntry(align::ID id, align::StructureType comp)
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
  treemaptype::iterator imap = treemap.find( std::make_pair(id,comp) );
  int result=-1;
  if (imap != treemap.end()) result=(*imap).second;
  return result;

}

// ----------------------------------------------------------------------------
int AlignableDataIORoot::writeAbsRaw(const AlignableAbsData &ad)
{
  const align::GlobalPoint& pos = ad.pos();
  align::RotationType rot = ad.rot();
  Id = ad.id();
  ObjId = ad.objId();
  Pos[0]=pos.x(); Pos[1]=pos.y(); Pos[2]=pos.z();
  Rot[0]=rot.xx(); Rot[1]=rot.xy(); Rot[2]=rot.xz();
  Rot[3]=rot.yx(); Rot[4]=rot.yy(); Rot[5]=rot.yz();
  Rot[6]=rot.zx(); Rot[7]=rot.zy(); Rot[8]=rot.zz();

  const std::vector<double> &deformPars = ad.deformationParameters();
  numDeformationValues_ = (deformPars.size() > kMaxNumPar ? kMaxNumPar : deformPars.size());
  for (unsigned int i = 0; i < numDeformationValues_; ++i) {
    deformationValues_[i] = deformPars[i];
  }

  tree->Fill();
  return 0;
}

// ----------------------------------------------------------------------------
int AlignableDataIORoot::writeRelRaw(const AlignableRelData &ad)
{
  const align::GlobalVector& pos = ad.pos();
  align::RotationType rot = ad.rot();
  Id = ad.id();
  ObjId = ad.objId();
  Pos[0]=pos.x(); Pos[1]=pos.y(); Pos[2]=pos.z();
  Rot[0]=rot.xx(); Rot[1]=rot.xy(); Rot[2]=rot.xz();
  Rot[3]=rot.yx(); Rot[4]=rot.yy(); Rot[5]=rot.yz();
  Rot[6]=rot.zx(); Rot[7]=rot.zy(); Rot[8]=rot.zz();

  const std::vector<double> &deformPars = ad.deformationParameters();
  numDeformationValues_ = (deformPars.size() > kMaxNumPar ? kMaxNumPar : deformPars.size());
  for (unsigned int i = 0; i < numDeformationValues_; ++i) {
    deformationValues_[i] = deformPars[i];
  }

  tree->Fill();
  return 0;
}

// ----------------------------------------------------------------------------
AlignableAbsData AlignableDataIORoot::readAbsRaw(Alignable* ali,int& ierr)
{
  align::GlobalPoint pos;
  align::RotationType rot;

  align::StructureType typeId = ali->alignableObjectId();
  align::ID id = ali->id();
  std::vector<double> deformPars; deformPars.reserve(numDeformationValues_);
  int entry = findEntry(id,typeId);
  if(entry!=-1) {
    tree->GetEntry(entry);
    align::GlobalPoint pos2(Pos[0],Pos[1],Pos[2]);
    align::RotationType rot2(Rot[0],Rot[1],Rot[2],
			     Rot[3],Rot[4],Rot[5],
			     Rot[6],Rot[7],Rot[8]);
    pos=pos2;
    rot=rot2;

    for (unsigned int i = 0; i < numDeformationValues_; ++i) {
      deformPars.push_back((double)deformationValues_[i]);
    }

    ierr=0;
  }
  else ierr=-1;

  return AlignableAbsData(pos,rot,id,typeId,deformPars);
}

// ----------------------------------------------------------------------------

AlignableRelData AlignableDataIORoot::readRelRaw(Alignable* ali,int& ierr)
{
  align::GlobalVector pos;
  align::RotationType rot;

  align::StructureType typeId = ali->alignableObjectId();
  align::ID id = ali->id();
  std::vector<double> deformPars; deformPars.reserve(numDeformationValues_);
  int entry = findEntry(id,typeId);
  if(entry!=-1) {
    tree->GetEntry(entry);
    align::GlobalVector pos2(Pos[0],Pos[1],Pos[2]);
    align::RotationType rot2(Rot[0],Rot[1],Rot[2],
			     Rot[3],Rot[4],Rot[5],
			     Rot[6],Rot[7],Rot[8]);
    pos=pos2;
    rot=rot2;

    for (unsigned int i = 0; i < numDeformationValues_; ++i) {
      deformPars.push_back((double)deformationValues_[i]);
    }

    ierr=0;
  }
  else ierr=-1;

  return AlignableRelData(pos,rot,id,typeId,deformPars);
}
