#ifndef __L1Analysis_L1AnalysisGMT_H__
#define __L1Analysis_L1AnalysisGMT_H__

#include <TTree.h>

namespace L1Analysis
{
  class L1AnalysisGMT
{

  public : 
  void initTree(TTree * ttree);

  public:
  L1AnalysisGMT() {}
  void print();
  bool check();    
  
  // ---- L1AnalysisGMT information.
    int gmtEvBx;
    
    //DTBX Trigger block
    int gmtNdt;
    std::vector<int>   gmtBxdt;
    std::vector<float> gmtPtdt;
    std::vector<int>   gmtChadt;
    std::vector<float> gmtEtadt;
    std::vector<int>   gmtFineEtadt;
    std::vector<float> gmtPhidt;
    std::vector<int>   gmtQualdt;
    std::vector<int>   gmtDwdt;
    std::vector<int>   gmtChdt;

    //CSC Trigger block
    int	gmtNcsc;
    std::vector<int>   gmtBxcsc;
    std::vector<float> gmtPtcsc;
    std::vector<int>   gmtChacsc;
    std::vector<float> gmtEtacsc;
    std::vector<float> gmtPhicsc;
    std::vector<int>   gmtQualcsc;
    std::vector<int>   gmtDwcsc;
   
    //RPCb Trigger
    int	gmtNrpcb ;
    std::vector<int>   gmtBxrpcb;
    std::vector<float> gmtPtrpcb;
    std::vector<int>   gmtCharpcb;
    std::vector<float> gmtEtarpcb;
    std::vector<float> gmtPhirpcb;
    std::vector<int>   gmtQualrpcb;
    std::vector<int>   gmtDwrpcb;
    
    //RPCf Trigger
    int	gmtNrpcf ;
    std::vector<int>   gmtBxrpcf;
    std::vector<float> gmtPtrpcf;
    std::vector<int>   gmtCharpcf;
    std::vector<float> gmtEtarpcf;
    std::vector<float> gmtPhirpcf;
    std::vector<int>   gmtQualrpcf;
    std::vector<int>   gmtDwrpcf;
    		  
    //Global Muon Trigger
    int gmtN;
    std::vector<int>	  gmtCandBx;
    std::vector<float>    gmtPt;
    std::vector<int>	  gmtCha;
    std::vector<float>    gmtEta;
    std::vector<float>    gmtPhi;
    std::vector<int>	  gmtQual;
    std::vector<int>	  gmtDet;
    std::vector<int>	  gmtRank;
    std::vector<int>	  gmtIsol;
    std::vector<int>	  gmtMip;
    std::vector<int>	  gmtDw;
    std::vector<int>	  gmtIdxRPCb;
    std::vector<int>	  gmtIdxRPCf;
    std::vector<int>	  gmtIdxDTBX;
    std::vector<int>	  gmtIdxCSC;
};
}

#endif

#ifdef l1ntuple_cxx

void L1Analysis::L1AnalysisGMT::initTree(TTree * tree)
{
  tree->SetBranchAddress("gmtEvBx",      &gmtEvBx);

  tree->SetBranchAddress("gmtNdt",       &gmtNdt);
  tree->SetBranchAddress("gmtBxdt",      &gmtBxdt);
  tree->SetBranchAddress("gmtPtdt",      &gmtPtdt);
  tree->SetBranchAddress("gmtChadt",     &gmtChadt);
  tree->SetBranchAddress("gmtEtadt",     &gmtEtadt);
  tree->SetBranchAddress("gmtFineEtadt", &gmtFineEtadt);
  tree->SetBranchAddress("gmtPhidt",     &gmtPhidt);
  tree->SetBranchAddress("gmtQualdt",    &gmtQualdt);
  tree->SetBranchAddress("gmtDwdt",      &gmtDwdt);
  tree->SetBranchAddress("gmtChdt",      &gmtChdt);

  tree->SetBranchAddress("gmtNcsc",      &gmtNcsc);
  tree->SetBranchAddress("gmtBxcsc",     &gmtBxcsc);
  tree->SetBranchAddress("gmtPtcsc",     &gmtPtcsc);
  tree->SetBranchAddress("gmtChacsc",    &gmtChacsc);
  tree->SetBranchAddress("gmtEtacsc",    &gmtEtacsc);
  tree->SetBranchAddress("gmtPhicsc",    &gmtPhicsc);
  tree->SetBranchAddress("gmtQualcsc",   &gmtQualcsc);
  tree->SetBranchAddress("gmtDwcsc",     &gmtDwcsc);

  tree->SetBranchAddress("gmtNrpcb",     &gmtNrpcb);
  tree->SetBranchAddress("gmtBxrpcb",    &gmtBxrpcb);
  tree->SetBranchAddress("gmtPtrpcb",    &gmtPtrpcb);
  tree->SetBranchAddress("gmtCharpcb",   &gmtCharpcb);
  tree->SetBranchAddress("gmtEtarpcb",   &gmtEtarpcb);
  tree->SetBranchAddress("gmtPhirpcb",   &gmtPhirpcb);
  tree->SetBranchAddress("gmtQualrpcb",  &gmtQualrpcb);
  tree->SetBranchAddress("gmtDwrpcb",    &gmtDwrpcb);
 
  tree->SetBranchAddress("gmtNrpcf",     &gmtNrpcf);
  tree->SetBranchAddress("gmtBxrpcf",    &gmtBxrpcf);
  tree->SetBranchAddress("gmtPtrpcf",    &gmtPtrpcf);
  tree->SetBranchAddress("gmtCharpcf",   &gmtCharpcf);
  tree->SetBranchAddress("gmtEtarpcf",   &gmtEtarpcf);
  tree->SetBranchAddress("gmtPhirpcf",   &gmtPhirpcf);
  tree->SetBranchAddress("gmtQualrpcf",  &gmtQualrpcf);
  tree->SetBranchAddress("gmtDwrpcf",    &gmtDwrpcf);

  tree->SetBranchAddress("gmtN",         &gmtN);
  tree->SetBranchAddress("gmtCandBx",    &gmtCandBx);
  tree->SetBranchAddress("gmtPt",        &gmtPt);
  tree->SetBranchAddress("gmtCha",       &gmtCha);
  tree->SetBranchAddress("gmtEta",       &gmtEta);
  tree->SetBranchAddress("gmtPhi",       &gmtPhi);
  tree->SetBranchAddress("gmtQual",      &gmtQual);
  tree->SetBranchAddress("gmtDet",       &gmtDet);
  tree->SetBranchAddress("gmtRank",      &gmtRank);
  tree->SetBranchAddress("gmtIsol",      &gmtIsol);
  tree->SetBranchAddress("gmtMip",       &gmtMip);
  tree->SetBranchAddress("gmtDw",        &gmtDw);
  tree->SetBranchAddress("gmtIdxRPCb",   &gmtIdxRPCb); 
  tree->SetBranchAddress("gmtIdxRPCf",   &gmtIdxRPCf);
  tree->SetBranchAddress("gmtIdxDTBX",   &gmtIdxDTBX);
  tree->SetBranchAddress("gmtIdxCSC",    &gmtIdxCSC);
}


void L1Analysis::L1AnalysisGMT::print()
{
}

bool L1Analysis::L1AnalysisGMT::check()
{
  bool test=true;
  /* if (gmtNdt!=gmtBxdt.size()) return false;
  if (gmtNcsc!=gmtBxcsc.size()) return false;
  if (gmtNrpcb!=gmtBxrpcb.size()) return false;
  if (gmtNrpcf!=gmtBxrpcf.size()) return false;
  if (gmtN!=gmtCandBx.size()) return false;*/

  return test;
}



#endif


