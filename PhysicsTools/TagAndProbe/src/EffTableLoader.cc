#include "PhysicsTools/TagAndProbe/interface/EffTableLoader.h"
#include "PhysicsTools/TagAndProbe/interface/EffTableReader.h"
#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"

namespace {
  const unsigned nParameters = 6;
}

EffTableLoader::EffTableLoader () 
  : mParameters (0) 
{}

EffTableLoader::EffTableLoader (const std::string& fDataFile) 
  : mParameters (new EffTableReader (fDataFile)) 
{}

EffTableLoader::~EffTableLoader () {
  delete mParameters;
}

int EffTableLoader::GetBandIndex(float fEt, float fEta)const {
  int index=mParameters->bandIndex(fEt, fEta);
  return index;
}
std::vector<float> EffTableLoader::correctionEff (float fEt,float fEta) const {
  int index=mParameters->bandIndex(fEt, fEta);
  EffTableReader::Record rec=mParameters->record(index);
  std::vector<float> param=rec.parameters();
  return param;
}
std::vector<float> EffTableLoader::correctionEff (int index) const {
  EffTableReader::Record rec=mParameters->record(index);
  std::vector<float> param=rec.parameters();
  return param;
}


std::vector<std::pair<float, float> > EffTableLoader::GetCellInfo(int index)const {
  EffTableReader::Record rec=mParameters->record(index);
  std::pair<float, float> PtBin;
  PtBin.first = rec.EtMin();
  PtBin.second= rec.EtMax();
   
  std::pair<float, float> EtaBin;
  EtaBin.first = rec.etaMin();
  EtaBin.second= rec.etaMax();
   
  std::vector<std::pair<float, float> > BinInfo;
  BinInfo.push_back(PtBin);
  BinInfo.push_back(EtaBin);
  return BinInfo ;
}



std::pair<float, float> EffTableLoader::GetCellCenter(int index )const {
  EffTableReader::Record rec=mParameters->record(index);
  std::pair<float, float> BinCenter;
  BinCenter.first = rec.EtMiddle();
  BinCenter.second= rec.etaMiddle();
  return BinCenter ;
}





std::vector<std::pair<float, float> > EffTableLoader::GetCellInfo(float fEt, float fEta)const {
  int index=mParameters->bandIndex(fEt, fEta);
  return (this->GetCellInfo(index)) ;
}





std::pair<float, float> EffTableLoader::GetCellCenter(float fEt, float fEta )const {
  int index=mParameters->bandIndex(fEt, fEta);
  return (this->GetCellCenter(index)); 
}

int EffTableLoader::size(void) {
  return mParameters->size();
}
