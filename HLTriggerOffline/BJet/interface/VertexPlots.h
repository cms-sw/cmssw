#ifndef HLTriggerOffline_BJet_VertexPlots_h
#define HLTriggerOffline_BJet_VertexPlots_h

// STL
#include <vector>
#include <string>

// ROOT
#include <TDirectory.h>
#include <TH1F.h>

// CMSSW
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"

struct VertexPlots {
  VertexPlots() :
    m_r(0),
    m_z(0)
  { }

  void init(const std::string & name, const std::string & title, unsigned int bins, double zRange, double rRange)
  {
    // enable sum-of-squares for all plots
    bool sumw2 = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);
    // disable directory association for all plots
    bool setdir = TH1::AddDirectoryStatus();
    TH1::AddDirectory(false);

    m_r  = new TH1F((name + "_R").c_str(),  (title + " R position").c_str(), bins, -rRange, rRange);
    m_z  = new TH1F((name + "_Z").c_str(),  (title + " Z position").c_str(), bins, -zRange, zRange);

    // reset sum-of-squares status
    TH1::SetDefaultSumw2(sumw2);
    // reset directory association status
    TH1::AddDirectory(setdir);
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
