#ifndef HLTriggerOffline_BJet_VertexPlots_h
#define HLTriggerOffline_BJet_VertexPlots_h

// STL
#include <vector>
#include <string>

// ROOT
#include <TDirectory.h>
#include <TH1F.h>

// CMSSW
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

struct VertexPlots {
  VertexPlots() :
    m_r(0),
    m_z(0)
  { }

  void init(const std::string & name, const std::string & title, unsigned int bins, double zRange, double rRange)
  {
    // access the shared ROOT file via TFileService
    edm::Service<TFileService> fileservice;
    
    // enable sum-of-squares for all plots
    bool sumw2 = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);

    m_r  = fileservice->make<TH1F>((name + "_R").c_str(),  (title + " R position").c_str(), bins, -rRange, rRange);
    m_z  = fileservice->make<TH1F>((name + "_Z").c_str(),  (title + " Z position").c_str(), bins, -zRange, zRange);

    // reset sum-of-squares status
    TH1::SetDefaultSumw2(sumw2);
  }
    
  void fill(const reco::Vertex & vertex)
  {
    m_r->Fill(vertex.position().rho());
    m_z->Fill(vertex.z());
  }

  void save(TDirectory & file)
  {
    m_r->SetDirectory(&file);
    m_z->SetDirectory(&file);
  }
  
  TH1 * m_r;
  TH1 * m_z;
};

#endif // HLTriggerOffline_BJet_VertexPlots_h
