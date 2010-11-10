#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/CodedException.h"

class CentralityProvider : public CentralityBins {

 public:
  CentralityProvider(const edm::EventSetup& iSetup);
  ~CentralityProvider(){;}

  enum VariableType {HFtowers,HFhits,PixelHits,PixelTracks,Tracks,EB,EE,Missing};

  int getNbins() const {return table_.size();}
  double centralityValue(const edm::Event& ev) const;
  int getBin(const edm::Event& ev) const {return CentralityBins::getBin(centralityValue(ev));}
  float lowEdge(const edm::Event& ev) const { return lowEdgeOfBin(getBin(ev));}
  float NpartMean(const edm::Event& ev) const { return NpartMeanOfBin(getBin(ev));}
  float NpartSigma(const edm::Event& ev) const { return NpartSigmaOfBin(getBin(ev));}
  float NcollMean(const edm::Event& ev) const { return NcollMeanOfBin(getBin(ev));}
  float NcollSigma(const edm::Event& ev)const { return NcollSigmaOfBin(getBin(ev));}
  float NhardMean(const edm::Event& ev) const { return NhardMeanOfBin(getBin(ev));}
  float NhardSigma(const edm::Event& ev) const { return NhardSigmaOfBin(getBin(ev));}
  float bMean(const edm::Event& ev) const { return bMeanOfBin(getBin(ev));}
  float bSigma(const edm::Event& ev) const { return bSigmaOfBin(getBin(ev));}
  void newRun(const edm::EventSetup& iSetup);
  void newEvent(const edm::Event& ev,const edm::EventSetup& iSetup);
  void print();
  const CentralityBins* table() const {return this;}
  const reco::Centrality* raw() const {return chandle_.product();}

 private:
  edm::InputTag tag_;
  std::string centralityVariable_;
  std::string centralityLabel_;
  std::string centralityMC_;
  unsigned int prevRun_;
  mutable edm::Handle<reco::Centrality> chandle_;
  VariableType varType_;
};

CentralityProvider::CentralityProvider(const edm::EventSetup& iSetup) : 
   prevRun_(0),
   varType_(Missing)
{
  const edm::ParameterSet &thepset = edm::getProcessParameterSet();
  if(thepset.exists("HeavyIonGlobalParameters")){
    edm::ParameterSet hiPset = thepset.getParameter<edm::ParameterSet>("HeavyIonGlobalParameters");
    tag_ = hiPset.getParameter<edm::InputTag>("centralitySrc");
    centralityVariable_ = hiPset.getParameter<std::string>("centralityVariable");
    if(centralityVariable_.compare("HFtowers") == 0) varType_ = HFtowers;
    if(centralityVariable_.compare("HFhits") == 0) varType_ = HFhits;
    if(centralityVariable_.compare("PixelHits") == 0) varType_ = PixelHits;
    if(centralityVariable_.compare("PixelTracks") == 0) varType_ = PixelTracks;
    if(centralityVariable_.compare("Tracks") == 0) varType_ = Tracks;
    if(centralityVariable_.compare("EB") == 0) varType_ = EB;
    if(centralityVariable_.compare("EE") == 0) varType_ = EE;
    if(varType_ == Missing){
      std::string errorMessage="Requested Centrality variable does not exist : "+centralityVariable_+"\n" +
	"Supported variables are: \n" + "HFtowers HFhits PixelHits PixelTracks Tracks EB EE" + "\n";
      throw cms::Exception("Configuration",errorMessage);
    }
    if(hiPset.exists("nonDefaultGlauberModel")){
       centralityMC_ = hiPset.getParameter<std::string>("nonDefaultGlauberModel");
    }
    centralityLabel_ = centralityVariable_+centralityMC_;
  }else{
  }
  newRun(iSetup);
}

void CentralityProvider::newEvent(const edm::Event& ev,const edm::EventSetup& iSetup){
   ev.getByLabel(tag_,chandle_);
  if(ev.id().run() == prevRun_) return;
  prevRun_ = ev.id().run();
  newRun(iSetup);
}

void CentralityProvider::newRun(const edm::EventSetup& iSetup){
  edm::ESHandle<CentralityTable> inputDB_;
  iSetup.get<HeavyIonRcd>().get(centralityLabel_,inputDB_);
  int nbinsMax = inputDB_->m_table.size();
  table_.clear();
  table_.reserve(nbinsMax);
  for(int j=0; j<nbinsMax; j++){
     const CentralityTable::CBin* thisBin;
     thisBin = &(inputDB_->m_table[j]);
     CBin newBin;
     newBin.bin_edge = thisBin->bin_edge;
     newBin.n_part_mean = thisBin->n_part.mean;
     newBin.n_part_var  = thisBin->n_part.var;
     newBin.n_coll_mean = thisBin->n_coll.mean;
     newBin.n_coll_var  = thisBin->n_coll.var;
     newBin.n_hard_mean = thisBin->n_hard.mean;
     newBin.n_hard_var  = thisBin->n_hard.var;
     newBin.b_mean = thisBin->b.mean;
     newBin.b_var = thisBin->b.var;
     table_.push_back(newBin);
  }
}

void CentralityProvider::print(){
   std::cout<<"Number of bins : "<<table_.size()<<std::endl;
   for(unsigned int j = 0; j < table_.size(); ++j){
      std::cout<<"Bin : "<<j<<std::endl;
      std::cout<<"Bin Low Edge : "<<table_[j].bin_edge <<std::endl;
      std::cout<<"Npart Mean : "<<table_[j].n_part_mean <<std::endl;
      std::cout<<"Npart RMS  : "<<table_[j].n_part_var <<std::endl;
      std::cout<<"Ncoll Mean : "<<table_[j].n_coll_mean <<std::endl;
      std::cout<<"Ncoll RMS  : "<<table_[j].n_coll_var <<std::endl;
      std::cout<<"Nhard Mean : "<<table_[j].n_hard_mean <<std::endl;
      std::cout<<"Nhard RMS  : "<<table_[j].n_hard_var <<std::endl;
      std::cout<<"b Mean     : "<<table_[j].b_mean <<std::endl;
      std::cout<<"b RMS      : "<<table_[j].b_var <<std::endl;
   }
}

double CentralityProvider::centralityValue(const edm::Event& ev) const {
  double var = -99;
  if(varType_ == HFhits) var = chandle_->EtHFhitSum();
  if(varType_ == HFtowers) var = chandle_->EtHFtowerSum();
  if(varType_ == PixelHits) var = chandle_->multiplicityPixel();
  if(varType_ == PixelTracks) var = chandle_->NpixelTracks();
  if(varType_ == Tracks) var = chandle_->Ntracks();
  if(varType_ == EB) var = chandle_->EtEBSum();
  if(varType_ == EE) var = chandle_->EtEESum();

  return var;
}











