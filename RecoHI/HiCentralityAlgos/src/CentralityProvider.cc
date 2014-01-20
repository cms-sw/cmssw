#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"

CentralityProvider::CentralityProvider(const edm::EventSetup& iSetup) :
   prevRun_(0),
   varType_(Missing)
{
   const edm::ParameterSet &thepset = edm::getProcessParameterSet();
   if(thepset.exists("HeavyIonGlobalParameters")){
      edm::ParameterSet hiPset = thepset.getParameter<edm::ParameterSet>("HeavyIonGlobalParameters");
      tag_ = hiPset.getParameter<edm::InputTag>("centralitySrc");
      centralityVariable_ = hiPset.getParameter<std::string>("centralityVariable");
      pPbRunFlip_ = hiPset.getUntrackedParameter<unsigned int>("pPbRunFlip",99999999);
      if(centralityVariable_.compare("HFtowers") == 0) varType_ = HFtowers;
      if(centralityVariable_.compare("HFtowersPlus") == 0) varType_ = HFtowersPlus;
      if(centralityVariable_.compare("HFtowersMinus") == 0) varType_ = HFtowersMinus;
      if(centralityVariable_.compare("HFtowersTrunc") == 0) varType_ = HFtowersTrunc;
      if(centralityVariable_.compare("HFtowersPlusTrunc") == 0) varType_ = HFtowersPlusTrunc;
      if(centralityVariable_.compare("HFtowersMinusTrunc") == 0) varType_ = HFtowersMinusTrunc;
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
      SetName(centralityLabel_.data());
   }else{
   }
   newRun(iSetup);
}

void CentralityProvider::newEvent(const edm::Event& ev,const edm::EventSetup& iSetup){
   ev.getByLabel(tag_,chandle_);
   if(ev.id().run() == prevRun_) return;
   if(prevRun_ < pPbRunFlip_ && ev.id().run() >= pPbRunFlip_){
     std::cout<<"Attention, the sides are flipped from this run on!"<<std::endl;
     if(centralityVariable_.compare("HFtowersPlus") == 0) varType_ = HFtowersMinus;
     if(centralityVariable_.compare("HFtowersMinus") == 0) varType_ = HFtowersPlus;
     if(centralityVariable_.compare("HFtowersPlusTrunc") == 0) varType_ = HFtowersMinusTrunc;
     if(centralityVariable_.compare("HFtowersMinusTrunc") == 0) varType_ = HFtowersPlusTrunc;
   }
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

      newBin.eccRP_mean = thisBin->eccRP.mean;
      newBin.eccRP_var = thisBin->eccRP.var;
      newBin.ecc2_mean = thisBin->ecc2.mean;
      newBin.ecc2_var = thisBin->ecc2.var;
      newBin.ecc3_mean = thisBin->ecc3.mean;
      newBin.ecc3_var = thisBin->ecc3.var;
      newBin.s_mean = thisBin->S.mean;
      newBin.s_var = thisBin->S.var;

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

double CentralityProvider::centralityValue() const {
   double var = -99;
   if(varType_ == HFhits) var = chandle_->EtHFhitSum();
   if(varType_ == HFtowers) var = chandle_->EtHFtowerSum();
   if(varType_ == HFtowersPlus) var = chandle_->EtHFtowerSumPlus();
   if(varType_ == HFtowersMinus) var = chandle_->EtHFtowerSumMinus();
   if(varType_ == HFtowersTrunc) var = chandle_->EtHFtruncated();
   if(varType_ == HFtowersPlusTrunc) var = chandle_->EtHFtruncatedPlus();
   if(varType_ == HFtowersMinusTrunc) var = chandle_->EtHFtruncatedMinus();
   if(varType_ == PixelHits) var = chandle_->multiplicityPixel();
   if(varType_ == PixelTracks) var = chandle_->NpixelTracks();
   if(varType_ == Tracks) var = chandle_->Ntracks();
   if(varType_ == EB) var = chandle_->EtEBSum();
   if(varType_ == EE) var = chandle_->EtEESum();

   return var;
}





