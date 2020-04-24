#include "DataFormats/MuonReco/interface/DYTInfo.h"
using namespace reco;

DYTInfo::DYTInfo() 
{
  NStUsed_ = 0;
  DYTEstimators_.assign (4,-1);
  UsedStations_.assign (4, false);
  IdChambers_.assign (4,DetId());
  Thresholds_.assign (4,-1);
}

DYTInfo::~DYTInfo() {}

void DYTInfo::CopyFrom(const DYTInfo &dytInfo)
{
  setNStUsed(dytInfo.NStUsed());
  setDYTEstimators(dytInfo.DYTEstimators());
  setUsedStations(dytInfo.UsedStations());
  setIdChambers(dytInfo.IdChambers());
  setThresholds(dytInfo.Thresholds());
}
