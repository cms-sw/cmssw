
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class CentralityProvider : public reco::Centrality, public CentralityBins {

 public:
  CentralityProvider(const edm::EventSetup& iSetup);
  ~CentralityProvider(){;}

  int getNbins() const {return table_.size();}
  double centralityValue(const edm::Event& ev) const;
  int getBin(const edm::Event& ev) const {return CentralityBins::getBin(centralityValue(ev));}
  float lowEdge(const edm::Event& ev) const { return lowEdgeOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float NpartMean(const edm::Event& ev) const { return NpartMeanOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float NpartSigma(const edm::Event& ev) const { return NpartSigmaOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float NcollMean(const edm::Event& ev) const { return NcollMeanOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float NcollSigma(const edm::Event& ev)const { return NcollSigmaOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float NhardMean(const edm::Event& ev) const { return NhardMeanOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float NhardSigma(const edm::Event& ev) const { return NhardSigmaOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float bMean(const edm::Event& ev) const { return bMeanOfBin(CentralityBins::getBin(centralityValue(ev)));}
  float bSigma(const edm::Event& ev) const { return bSigmaOfBin(CentralityBins::getBin(centralityValue(ev)));}
  void newRun(const edm::EventSetup& iSetup);
  void refreshIOV(const edm::Event& ev,const edm::EventSetup& iSetup);

 private:
  edm::InputTag tag_;
  std::string centralityLabel_;
  std::string centralityMC_;
  unsigned int prevRun_;
  
};

CentralityProvider::CentralityProvider(const edm::EventSetup& iSetup){
  const edm::ParameterSet &thepset = edm::getProcessParameterSet();
  if(thepset.exists("HeavyIonGlobalParameters")){
    edm::ParameterSet hiPset = thepset.getParameter<edm::ParameterSet>("HeavyIonGlobalParameters");
    tag_ = hiPset.getParameter<edm::InputTag>("centralitySrc");
    centralityLabel_ = hiPset.getParameter<std::string>("centralityVariable");
    if(hiPset.exists("nonDefaultGlauberModel")){
      centralityMC_ = hiPset.getParameter<std::string>("nonDefaultGlauberModel");
      centralityLabel_ += centralityMC_;
    }
  }else{
  }
  newRun(iSetup);
}

void CentralityProvider::refreshIOV(const edm::Event& ev,const edm::EventSetup& iSetup){
  if(ev.id().run() == prevRun_) return;
  newRun(iSetup);
}

void CentralityProvider::newRun(const edm::EventSetup& iSetup){
  edm::ESHandle<CentralityTable> inputDB_;
  iSetup.get<HeavyIonRcd>().get(centralityLabel_,inputDB_);
  int nbinsMax = inputDB_->m_table.size();
  table_.reserve(nbinsMax);
  for(int j=0; j<nbinsMax; j++){
    const CentralityTable::CBin* thisBin;
    thisBin = &(inputDB_->m_table[j]);
    table_[j].bin_edge = thisBin->bin_edge;
    table_[j].n_part_mean = thisBin->n_part.mean;
    table_[j].n_part_var  = thisBin->n_part.var;
    table_[j].n_coll_mean = thisBin->n_coll.mean;
    table_[j].n_coll_var  = thisBin->n_coll.var;
    table_[j].n_hard_mean = thisBin->n_hard.mean;
    table_[j].n_hard_var  = thisBin->n_hard.var;
    table_[j].b_mean = thisBin->b.mean;
    table_[j].b_var = thisBin->b.var;
  }
}

double CentralityProvider::centralityValue(const edm::Event& ev) const{
  
  edm::Handle<reco::Centrality> c;
  ev.getByLabel(tag_,c);
  double var = -99;
  if(centralityLabel_.compare("HFhits") == 0) var = c->EtHFhitSum();
  if(centralityLabel_.compare("PixelHits") == 0) var = c->multiplicityPixel();
  if(centralityLabel_.compare("PixelTracks") == 0) var = c->NpixelTracks();

  return var;
}











