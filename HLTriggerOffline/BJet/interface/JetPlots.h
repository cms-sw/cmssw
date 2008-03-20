#ifndef HLTriggerOffline_BJet_JetPlots_h
#define HLTriggerOffline_BJet_JetPlots_h

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

static const unsigned int jetEnergyBins   =  100;
static const unsigned int jetGeometryBins =  100;
static const unsigned int vertex1DBins    = 1000;
static const unsigned int vertex3DBins    =  100;

// jet plots (for each step)
struct JetPlots {
  JetPlots()
  { }
  
  // let ROOT handle deleting the histograms
  ~JetPlots()
  { }
  
  void init(const std::string & name, const std::string & title, unsigned int energyBins, double minEnergy, double maxEnergy, unsigned int geometryBins, double maxEta, bool hasTracks = false)
  {
    m_overall       = 0.0;
    m_name          = name;
    m_title         = title;
    m_energyBins    = energyBins;
    m_minEnergy     = minEnergy;
    m_maxEnergy     = maxEnergy;
    m_geometryBins  = geometryBins;
    m_maxEta        = maxEta;
    m_hasTracks     = hasTracks;
    
    bool sumw2 = TH1::GetDefaultSumw2();
    TH1::SetDefaultSumw2(true);
    m_jetEnergy = new TH1F((m_name + "_jets_energy").c_str(), (m_title + " jets energy").c_str(), m_energyBins,    m_minEnergy, m_maxEnergy);
    m_jetET     = new TH1F((m_name + "_jets_ET").c_str(),     (m_title + " jets ET").c_str(),     m_energyBins,    m_minEnergy, m_maxEnergy);
    m_jetEta    = new TH1F((m_name + "_jets_eta").c_str(),    (m_title + " jets eta").c_str(),    m_geometryBins, -m_maxEta,    m_maxEta);
    m_jetPhi    = new TH1F((m_name + "_jets_phi").c_str(),    (m_title + " jets phi").c_str(),    m_geometryBins, -M_PI,        M_PI);
    if (m_hasTracks) {
      m_tracksEnergy = new TH1F((m_name + "_tracks_energy").c_str(), ("Tracks in " + m_title + " jets vs. jet energy").c_str(), m_energyBins,    m_minEnergy, m_maxEnergy);
      m_tracksET     = new TH1F((m_name + "_tracks_ET").c_str(),     ("Tracks in " + m_title + " jets vs. jet ET").c_str(),     m_energyBins,    m_minEnergy, m_maxEnergy);
      m_tracksEta    = new TH1F((m_name + "_tracks_eta").c_str(),    ("Tracks in " + m_title + " jets vs. jet eta").c_str(),    m_geometryBins, -m_maxEta,    m_maxEta);
      m_tracksPhi    = new TH1F((m_name + "_tracks_phi").c_str(),    ("Tracks in " + m_title + " jets vs. jet phi").c_str(),    m_geometryBins, -M_PI,        M_PI);
    }
    TH1::SetDefaultSumw2(sumw2);
  }

  void fill(const reco::Jet & jet) {
    ++m_overall;
    m_jetEnergy->Fill(jet.energy());
    m_jetET->Fill(jet.et());
    m_jetEta->Fill(jet.eta());
    m_jetPhi->Fill(jet.phi());
  }

  void fill(const reco::Jet & jet, const reco::TrackRefVector & tracks) {
    fill(jet);
    if (m_hasTracks) {
      m_tracksEnergy->Fill(jet.energy(), tracks.size());
      m_tracksET->Fill(jet.et(), tracks.size());
      m_tracksEta->Fill(jet.eta(), tracks.size());
      m_tracksPhi->Fill(jet.phi(), tracks.size());
    }
  }

  void save(TDirectory & file) {
    m_jetEnergy->SetDirectory(&file);
    m_jetET->SetDirectory(&file);
    m_jetEta->SetDirectory(&file);
    m_jetPhi->SetDirectory(&file);
    if (m_hasTracks) {
      // normalize the number of tracks to the number of jets (i.e. compute average number of tracks per jet)
      m_tracksEnergy->Divide(m_jetEnergy);
      m_tracksEnergy->SetDirectory(&file);
      m_tracksET->Divide(m_jetET);
      m_tracksET->SetDirectory(&file);
      m_tracksEta->Divide(m_jetEta);
      m_tracksEta->SetDirectory(&file);
      m_tracksPhi->Divide(m_jetPhi);
      m_tracksPhi->SetDirectory(&file);
    }
  }

  JetPlots efficiency(const JetPlots & denominator)
  {
    //std::stringstream out;

    JetPlots efficiency;
    efficiency.init(m_name + "_vs_" + denominator.m_name, m_title + " vs. " + denominator.m_title, m_energyBins, m_minEnergy, m_maxEnergy, m_geometryBins, m_maxEta, false);
    if (denominator.m_overall != 0.0) {
      efficiency.m_overall = m_overall / denominator.m_overall;
      //out << std::setw(57) << std::left << efficiency.m_title << "efficiency: " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << efficiency.m_overall * 100. << "%";
    } else {
      efficiency.m_overall = NAN;
      //out << std::setw(57) << std::left << efficiency.m_title << "efficiency:     NaN";
    }
    //std::cout << out.str() << std::endl;

    efficiency.m_jetEnergy->Divide(m_jetEnergy, denominator.m_jetEnergy, 1., 1., "B");
    efficiency.m_jetEnergy->SetMinimum(0.0);
    efficiency.m_jetEnergy->SetMaximum(1.0);
    efficiency.m_jetET->Divide(m_jetET, denominator.m_jetET, 1., 1., "B");
    efficiency.m_jetET->SetMinimum(0.0);
    efficiency.m_jetET->SetMaximum(1.0);
    efficiency.m_jetEta->Divide(m_jetEta, denominator.m_jetEta, 1., 1., "B");
    efficiency.m_jetEta->SetMinimum(0.0);
    efficiency.m_jetEta->SetMaximum(1.0);
    efficiency.m_jetPhi->Divide(m_jetPhi, denominator.m_jetPhi, 1., 1., "B");
    efficiency.m_jetPhi->SetMinimum(0.0);
    efficiency.m_jetPhi->SetMaximum(1.0);
    return efficiency;
  }
 
  TH1F * m_jetEnergy;
  TH1F * m_jetET;
  TH1F * m_jetEta;
  TH1F * m_jetPhi;
  TH1F * m_tracksEnergy;
  TH1F * m_tracksET;
  TH1F * m_tracksEta;
  TH1F * m_tracksPhi;
  double        m_overall;
  std::string   m_name;
  std::string   m_title;
  unsigned int  m_geometryBins;
  unsigned int  m_energyBins;
  double        m_minEnergy;
  double        m_maxEnergy;
  double        m_maxEta;

  bool          m_hasTracks;
};

#endif // HLTriggerOffline_BJet_JetPlots_h
