#ifndef DPGAnalysis_SiStripTools_DigiCollectionProfile_H
#define DPGAnalysis_SiStripTools_DigiCollectionProfile_H

#include <vector>
#include "DPGAnalysis/SiStripTools/interface/DetIdSelector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSet.h"

class TH1F;
class TProfile;
class TH2F;

namespace edm {
  class ParameterSet;
}

template <class T>
class DigiCollectionProfiler {

 public:
  DigiCollectionProfiler();
  DigiCollectionProfiler(const edm::ParameterSet& iConfig);
  ~DigiCollectionProfiler() {};

  void fill(edm::Handle<T> digis, std::vector<TH1F*>, std::vector<TProfile*>, std::vector<TH2F*>) const;

 private:

  bool m_folded;
  bool m_want1dHisto;
  bool m_wantProfile;
  bool m_want2dHisto;

  std::vector<DetIdSelector> m_selections;

};

template <class T>
DigiCollectionProfiler<T>::DigiCollectionProfiler():
  m_folded(false), m_want1dHisto(false), m_wantProfile(false), m_want2dHisto(false), m_selections() { }

template <class T>
DigiCollectionProfiler<T>::DigiCollectionProfiler(const edm::ParameterSet& iConfig):
  m_folded(iConfig.getUntrackedParameter<bool>("foldedStrips",false)),
  m_want1dHisto(iConfig.getUntrackedParameter<bool>("want1dHisto",true)),
  m_wantProfile(iConfig.getUntrackedParameter<bool>("wantProfile",true)),
  m_want2dHisto(iConfig.getUntrackedParameter<bool>("want2dHisto",false))

{

  std::vector<edm::ParameterSet> selconfigs = iConfig.getParameter<std::vector<edm::ParameterSet> >("selections");
  
  for(std::vector<edm::ParameterSet>::const_iterator selconfig=selconfigs.begin();selconfig!=selconfigs.end();++selconfig) {
    DetIdSelector selection(*selconfig);
    m_selections.push_back(selection);
  }

}

template <class T>
void DigiCollectionProfiler<T>::fill(edm::Handle<T> digis, std::vector<TH1F*> hist, std::vector<TProfile*> hprof, std::vector<TH2F*> hist2d) const {

}


template <>
void DigiCollectionProfiler<edm::DetSetVector<SiStripDigi> >::fill(edm::Handle<edm::DetSetVector<SiStripDigi> > digis, std::vector<TH1F*> hist, std::vector<TProfile*> hprof, std::vector<TH2F*> hist2d) const {

  for(edm::DetSetVector<SiStripDigi>::const_iterator mod = digis->begin();mod!=digis->end();mod++) {

    for(unsigned int isel=0;isel< m_selections.size(); ++isel) {
      
      if(m_selections[isel].isSelected(mod->detId())) {
	TH1F* tobefilled1d=0;
	TProfile* tobefilledprof=0;
	TH2F* tobefilled2d=0;
	
	if(m_want1dHisto) tobefilled1d = hist[isel];
	if(m_wantProfile) tobefilledprof = hprof[isel];
	if(m_want2dHisto) tobefilled2d = hist2d[isel];
	
	for(edm::DetSet<SiStripDigi>::const_iterator digi=mod->begin();digi!=mod->end();digi++) {
	  
	  if(digi->adc()>0) {
	    unsigned int strip = digi->strip();
	    if(m_folded) strip = strip%256;
	    if(tobefilled1d) tobefilled1d->Fill(strip);
	    if(tobefilledprof) tobefilledprof->Fill(strip,digi->adc());
	    if(tobefilled2d) tobefilled2d->Fill(strip,digi->adc());
	  }
	}
      }
    }
  }
}

template <>
void DigiCollectionProfiler<edmNew::DetSetVector<SiStripCluster> >::fill(edm::Handle<edmNew::DetSetVector<SiStripCluster> > digis, std::vector<TH1F*> hist, std::vector<TProfile*> hprof, std::vector<TH2F*> hist2d) const {

  for(edmNew::DetSetVector<SiStripCluster>::const_iterator mod = digis->begin();mod!=digis->end();mod++) {

    for(unsigned int isel=0;isel< m_selections.size(); ++isel) {
      
      if(m_selections[isel].isSelected(mod->detId())) {
	TH1F* tobefilled1d=0;
	TProfile* tobefilledprof=0;
	TH2F* tobefilled2d=0;
	
	if(m_want1dHisto) tobefilled1d = hist[isel];
	if(m_wantProfile) tobefilledprof = hprof[isel];
	if(m_want2dHisto) tobefilled2d = hist2d[isel];
	
	for(edmNew::DetSet<SiStripCluster>::const_iterator clus=mod->begin();clus!=mod->end();clus++) {

	  for(unsigned int digi=0; digi < clus->amplitudes().size() ; ++digi) {
	  
	    if(clus->amplitudes()[digi]>0) {
	      unsigned int strip = clus->firstStrip()+digi;
	      if(m_folded) strip = strip%256;
	      if(tobefilled1d) tobefilled1d->Fill(strip);
	      if(tobefilledprof) tobefilledprof->Fill(strip,clus->amplitudes()[digi]);
	      if(tobefilled2d) tobefilled2d->Fill(strip,clus->amplitudes()[digi]);
	    }
	  }
	}
      }
    }
  }
}

#endif // DPGAnalysis_SiStripTools_DigiCollectionProfile_H
