#ifndef HLTriggerOffline_BJet_OfflineJetPlots_h
#define HLTriggerOffline_BJet_OfflineJetPlots_h

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

struct OfflineJetPlots {

  std::vector<double>       m_cuts;
  std::vector<std::string>  m_labels;
  std::vector<JetPlots>     m_plots;

  void init(
      const std::string & name, 
      const std::string & title, 
      const std::vector<double> & cuts, 
      const std::vector<std::string> & labels, 
      unsigned int energyBins, 
      double minEnergy, 
      double maxEnergy, 
      unsigned int geometryBins, 
      double maxEta, 
      bool hasTracks = false
  ) {
    m_cuts = cuts;
    m_labels = labels;
    m_plots.resize( m_cuts.size() );
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      m_plots[i].init(name + "_" + m_labels[i], title + " (" + m_labels[i] + ")", energyBins, minEnergy, maxEnergy, geometryBins, maxEta, hasTracks);
  }
  
  void fill(const reco::Jet & jet, double discriminant) {
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      if (discriminant >= m_cuts[i])
        m_plots[i].fill(jet);
  }
  
  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks, double discriminant) {
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      if (discriminant >= m_cuts[i])
        m_plots[i].fill(jet, tracks);
  }

  void save(TDirectory & file) {
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      m_plots[i].save(file);
  }

  OfflineJetPlots efficiency(const OfflineJetPlots & denominator) {
    OfflineJetPlots efficiency;
    efficiency.m_cuts = m_cuts;
    efficiency.m_labels = m_labels;
    efficiency.m_plots.resize(m_cuts.size());
    for (unsigned int i = 0; i < m_cuts.size(); ++i)
      efficiency.m_plots[i] = m_plots[i].efficiency( denominator.m_plots[i] );
    return efficiency;
  }
};

#endif // HLTriggerOffline_BJet_OfflineJetPlots_h
