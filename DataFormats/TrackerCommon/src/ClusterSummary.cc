#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ClusterSummary::ClusterSummary() : ClusterSummary(NDEFAULTENUMS) {}

ClusterSummary::ClusterSummary(const int nSelections) : modules(nSelections),nClus(nSelections),clusSize(nSelections),clusCharge(nSelections)
{
  for(int i = 0; i < nSelections; ++i) modules[i] = i;
}

ClusterSummary& ClusterSummary::operator=(const ClusterSummary& rhs)
{
  modules       = rhs.modules       ;
  nClus         = rhs.nClus         ;
  clusSize      = rhs.clusSize      ;
  clusCharge    = rhs.clusCharge    ;
  return *this;
}

// move ctor
ClusterSummary::ClusterSummary(ClusterSummary&& other) : ClusterSummary()
{
    *this = other;
}

ClusterSummary::ClusterSummary(const ClusterSummary& src) :
    modules   (src.getModules()         ),
    nClus     (src.getNClusVector()     ),
    clusSize  (src.getClusSizeVector()  ),
    clusCharge(src.getClusChargeVector())
{}

std::vector<ClusterSummary::CMSTrackerSelection> ClusterSummary::getStandardSelections(){
  std::vector<CMSTrackerSelection> selections(NDEFAULTENUMS);

  selections[STRIP].name = "STRIP";
  selections[STRIP].selection.push_back("0x1e000000-0x1A000000");
  selections[STRIP].selection.push_back("0x1e000000-0x16000000");
  selections[STRIP].selection.push_back("0x1e000000-0x18000000");
  selections[STRIP].selection.push_back("0x1e000000-0x1C000000");

  selections[TOB].name = "TOB";
  selections[TOB].selection.push_back("0x1e000000-0x1A000000");
  selections[TIB].name = "TIB";
  selections[TIB].selection.push_back("0x1e000000-0x16000000");
  selections[TID].name = "TID";
  selections[TID].selection.push_back("0x1e000000-0x18000000");
  selections[TEC].name = "TEC";
  selections[TEC].selection.push_back("0x1e000000-0x1C000000");

  selections[PIXEL].name = "PIXEL";
  selections[PIXEL].selection.push_back("0x1e000000-0x12000000");
  selections[PIXEL].selection.push_back("0x1e000000-0x14000000");

  selections[BPIX].name = "BPIX";
  selections[BPIX].selection.push_back("0x1e000000-0x12000000");
  selections[FPIX].name = "FPIX";
  selections[FPIX].selection.push_back("0x1e000000-0x14000000");
  return selections;
}

std::string ClusterSummary::getVarName(const VariablePlacement var){
  static std::string varVec[] = {"NCLUSTERS","CLUSTERSIZE","CLUSTERCHARGE"};
  return var >= NVARIABLES ? "UNKOWN" : varVec[var];
}

ClusterSummary::CMSTracker ClusterSummary::getSubDetEnum(const std::string name){
  for(int i=0; i<NDEFAULTENUMS; i++)
    if(name == getSubDetName((CMSTracker)i)) return (CMSTracker)i;
  return NDEFAULTENUMS;
}

int ClusterSummary::getModuleLocation ( int mod, bool warn ) const {
  int iM = -1;
  for (auto m : modules){++iM; if (m==mod) break;}

  if(!warn)
    return -1;

    edm::LogWarning("NoModule") << "No information for requested module "<<mod<<". Please check in the Provinence Infomation for proper modules.";
    return -1;
}

void ClusterSummary::cleanAndReset(ClusterSummary& src){
  std::vector<int>  & src_modules     = src.getModules()         ;
  std::vector<int>  & src_nClus       = src.getNClusVector()     ;
  std::vector<int>  & src_clusSize    = src.getClusSizeVector()  ;
  std::vector<float>& src_clusCharge  = src.getClusChargeVector();

  modules   .clear();
  nClus     .clear();
  clusSize  .clear();
  clusCharge.clear();

  modules   .reserve(src_modules.size());
  nClus     .reserve(src_modules.size());
  clusSize  .reserve(src_modules.size());
  clusCharge.reserve(src_modules.size());

  for(unsigned int iM = 0; iM < src_modules.size(); ++iM){
    if(src_nClus[iM] != 0){
      modules   .push_back(src_modules   [iM]);
      nClus     .push_back(src_nClus     [iM]);
      clusSize  .push_back(src_clusSize  [iM]);
      clusCharge.push_back(src_clusCharge[iM]);
    }
    src_nClus     [iM] = 0;
    src_clusSize  [iM] = 0;
    src_clusCharge[iM] = 0;
  }
}
