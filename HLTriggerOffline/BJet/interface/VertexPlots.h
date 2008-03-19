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
    m_r  = new TH1F((name + "_R").c_str(),  (title + " R position").c_str(), bins, -rRange, rRange);
    m_z  = new TH1F((name + "_Z").c_str(),  (title + " Z position").c_str(), bins, -zRange, zRange);
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
