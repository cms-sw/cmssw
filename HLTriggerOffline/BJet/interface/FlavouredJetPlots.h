#ifndef HLTriggerOffline_BJet_FlavouredJetPlots_h
#define HLTriggerOffline_BJet_FlavouredJetPlots_h

// STL
#include <vector>
#include <string>

// ROOT
#include <TDirectory.h>

// CMSSW
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "HLTriggerOffline/BJet/interface/JetPlots.h"

typedef unsigned int            flavour_t;
typedef std::vector<flavour_t>  flavours_t;
struct FlavouredJetPlots {

  std::vector<flavours_t>   m_flavours;
  std::vector<std::string>  m_labels;
  std::vector<JetPlots>     m_plots;
  
  void init(
      const std::string & name, 
      const std::string & title, 
      const std::vector<flavours_t> & flavours, 
      const std::vector<std::string> & labels,
      unsigned int energyBins, 
      double minEnergy, 
      double maxEnergy, 
      unsigned int geometryBins, 
      double maxEta, 
      bool hasTracks = false
  ) {
    m_flavours = flavours;
    m_labels = labels; m_labels.push_back("other");
    m_plots.resize( m_flavours.size() + 1);
    for (unsigned int i = 0; i <= m_flavours.size(); ++i)
      m_plots[i].init(name + "_" + m_labels[i], title + " (" + m_labels[i] + ")", energyBins, minEnergy, maxEnergy, geometryBins, maxEta, hasTracks);
  }

  void fill(const reco::Jet & jet, flavour_t flavour) {
    bool match = false;
    for (unsigned int i = 0; i < m_flavours.size(); ++i) {
      if (std::find(m_flavours[i].begin(), m_flavours[i].end(), flavour) != m_flavours[i].end()) {
        m_plots[i].fill(jet);
        match = true;
      }
    }
    if (not match)
      m_plots[m_flavours.size()].fill(jet);
  }
  
  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks, flavour_t flavour) {
    bool match = false;
    for (unsigned int i = 0; i < m_flavours.size(); ++i) {
      if (std::find(m_flavours[i].begin(), m_flavours[i].end(), flavour) != m_flavours[i].end()) {
        m_plots[i].fill(jet, tracks);
        match = true;
      }
    }
    if (not match)
      m_plots[m_flavours.size()].fill(jet);
  }
  
  void save(TDirectory & file) {
    for (unsigned int i = 0; i <= m_flavours.size(); ++i)
      m_plots[i].save(file);
  }

  FlavouredJetPlots efficiency(const FlavouredJetPlots & denominator) {
    FlavouredJetPlots efficiency;
    efficiency.m_flavours = m_flavours;
    efficiency.m_labels = m_labels;
    efficiency.m_plots.resize(m_flavours.size() + 1);
    for (unsigned int i = 0; i <= m_flavours.size(); ++i)
      efficiency.m_plots[i] = m_plots[i].efficiency( denominator.m_plots[i] );
    return efficiency;
  }
};

#endif // HLTriggerOffline_BJet_FlavouredJetPlots_h
