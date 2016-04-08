#include "Geometry/VeryForwardRPTopology/interface/RPHepPDTWrapper.h"

HepPDT::ParticleDataTable *RPHepPDTWrapper::datacol = NULL;


void RPHepPDTWrapper::SetBinLabels(TH1 & hist)
{
  if(!datacol)
    init();
    
  int bins_no = hist.GetNbinsX();
  for(int i=1; i<=bins_no; ++i)
  {
    if(hist.GetBinContent(i)>0)
    {
      int pdg_code = (int) hist.GetBinCenter(i);
      std::string name = GetName(pdg_code);
      if(name!="")
      {
        hist.GetXaxis()->SetBinLabel(i,name.c_str());
      }
      else
      {
        name = std::string() + (long int)pdg_code;
        hist.GetXaxis()->SetBinLabel(i,name.c_str());
      }
    }
  }
}
