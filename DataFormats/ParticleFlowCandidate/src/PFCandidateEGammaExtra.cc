#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include <ostream>
#include <iomanip>

using namespace reco;

PFCandidateEGammaExtra::PFCandidateEGammaExtra() {
  status_ = 0;  
  mvaStatus_ = 0;
  pout_ = math::XYZTLorentzVector(0.,0.,0.,0.);
  hadEnergy_ = -9999. ;
  sigmaEtaEta_ = -9999.;

  for(MvaVariable m=MVA_FIRST; m<MVA_LAST; m=MvaVariable(m+1))
    mvaVariables_.push_back(-9999.);
      
  gsfTrackRef_ = GsfTrackRef();
  kfTrackRef_ = TrackRef();
}

PFCandidateEGammaExtra::PFCandidateEGammaExtra(const reco::GsfTrackRef& gsfTrack) {
  status_ = 0;  
  mvaStatus_ = 0;
  pout_ = math::XYZTLorentzVector(0.,0.,0.,0.);
  hadEnergy_ = -9999. ;
  sigmaEtaEta_ = -9999.;

  for(MvaVariable m=MVA_FIRST; m<MVA_LAST; m=MvaVariable(m+1))
    mvaVariables_.push_back(-9999.);
      
  gsfTrackRef_ = gsfTrack;
  kfTrackRef_ = TrackRef();
  
  setVariable(MVA_LnPtGsf,log(gsfTrackRef_->ptMode()));
  setVariable(MVA_EtaGsf,gsfTrackRef_->etaMode());
  setVariable(MVA_Chi2Gsf,gsfTrackRef_->normalizedChi2());  
  float ptmodeerror=gsfTrackRef_->ptModeError() ;
  if(ptmodeerror>0.)
    setVariable(MVA_SigmaPtOverPt,ptmodeerror/gsfTrackRef_->ptMode());
  else
    setVariable(MVA_SigmaPtOverPt,-999.);
  
  setVariable(MVA_Fbrem,(gsfTrackRef_->ptMode() - pout_.pt())/gsfTrackRef_->ptMode());
}

void PFCandidateEGammaExtra::setGsfTrackRef(const reco::GsfTrackRef& ref) {
  gsfTrackRef_= ref;
}   

void PFCandidateEGammaExtra::setGsfTrackPout(const math::XYZTLorentzVector& pout) {
  pout_ = pout;
  
}

void PFCandidateEGammaExtra::setKfTrackRef(const reco::TrackRef & ref){
  kfTrackRef_ = ref;
  float nhit_kf=0;
  float chi2_kf=-0.01;
  // if the reference is null, it does not mean that the variables have not been set
  if(kfTrackRef_.isNonnull()) {
    nhit_kf=(float)kfTrackRef_->hitPattern().trackerLayersWithMeasurement();
    chi2_kf=kfTrackRef_->normalizedChi2();
  }
  setVariable(MVA_NhitsKf,nhit_kf);
  setVariable(MVA_Chi2Kf,chi2_kf);
}

void PFCandidateEGammaExtra::setLateBrem(float val) {
  lateBrem_ = val;
  setVariable(MVA_LateBrem,val);
}

void PFCandidateEGammaExtra::setEarlyBrem(float val) {
  earlyBrem_ = val;
  setVariable(MVA_FirstBrem,val);
}

void PFCandidateEGammaExtra::setHadEnergy(float val) {
  hadEnergy_ = val;
  if(clusterEnergies_.size()>0)
    setVariable(MVA_HOverHE,hadEnergy_/(hadEnergy_+clusterEnergies_[0]));
}

void PFCandidateEGammaExtra::setSigmaEtaEta(float val) {
  sigmaEtaEta_ = val;
  setVariable(MVA_LogSigmaEtaEta,val);
}

void PFCandidateEGammaExtra::setDeltaEta(float val) {
  deltaEta_ = val;
  setVariable(MVA_DeltaEtaTrackCluster,val);
}


void PFCandidateEGammaExtra::setClusterEnergies(const std::vector<float>& energies){
  clusterEnergies_=energies;

  if(pout_.t()!=0.)
    setVariable(MVA_EseedOverPout,clusterEnergies_[0]/pout_.t());
 
  static float m_el2=0.00051*0.00051;
  float Ein_gsf =sqrt(gsfTrackRef_->pMode()*
		      gsfTrackRef_->pMode()+m_el2);
  
  float etot=0;
  unsigned size=clusterEnergies_.size();
  //  std::cout << " N clusters "  << size << std::endl;
  float ebrem=0.;
  for(unsigned ic=0;ic<size;++ic) {    
    etot+=clusterEnergies_[ic];
    if(ic>0)
      ebrem+=clusterEnergies_[ic];
  }
  setVariable(MVA_EtotOverPin,etot/Ein_gsf);
  setVariable(MVA_EbremOverDeltaP,ebrem/(Ein_gsf-pout_.t()));  

  // recompute - as in PFElectronAglo, the had energy is filled before the cluster energies
  if(hadEnergy_!=-9999.)
    setHadEnergy(hadEnergy_);

}

void PFCandidateEGammaExtra::setMVA(float val) {
  setVariable(MVA_MVA,val);
}

void PFCandidateEGammaExtra::setVariable(MvaVariable type,float val) {
  
  mvaVariables_[type]=val;
  mvaStatus_ |= (1 << (type)) ;    
}


void PFCandidateEGammaExtra::setStatus(StatusFlag type,bool status){
  if(status)     {
    status_ |= (1 << type);
  }
  else     {
    status_ &= ~(1 <<type);
  }
}

bool PFCandidateEGammaExtra::electronStatus(StatusFlag flag) const {
  return status_ & (1<<flag) ;
}

bool PFCandidateEGammaExtra::mvaStatus(MvaVariable flag) const {
  return mvaStatus_ & (1<< (flag)) ;
}

float PFCandidateEGammaExtra::mvaVariable(MvaVariable var) const {
  return (mvaStatus(var) ? mvaVariables_[var] : -9999. );
}

void PFCandidateEGammaExtra::addSingleLegConvTrackRef(const reco::TrackRef& trackref){
  assoSingleLegRefTrack_.push_back(trackref);
}

void PFCandidateEGammaExtra::addSingleLegConvMva(float& mvasingleleg){
  assoSingleLegMva_.push_back(mvasingleleg);
}

void PFCandidateEGammaExtra::addConversionRef(const reco::ConversionRef& convref){
  assoConversionsRef_.push_back(convref);
}

#include <string>

std::ostream& reco::operator<<(std::ostream& out, 
			       const PFCandidateEGammaExtra& extra ) { 
  if(!out) return out;

  static std::vector<std::string> listVar;
  if(listVar.size()==0) {
    listVar.push_back("LogPt");
    listVar.push_back("Eta");
    listVar.push_back("SigmaPtOverPt");
    listVar.push_back("fbrem");
    listVar.push_back("Chi2Gsf");
    listVar.push_back("NhitsKf");
    listVar.push_back("Chi2Kf");
    listVar.push_back("EtotOverPin");
    listVar.push_back("EseedOverPout");
    listVar.push_back("EbremOverDeltaP");
    listVar.push_back("DeltaEtaTrackCluster");
    listVar.push_back("LogSigmaEtaEta");
    listVar.push_back("H/(H+E)");
    listVar.push_back("LateBrem");
    listVar.push_back("FirstBrem");
    listVar.push_back("MVA");
  }
  out <<  std::setiosflags(std::ios::left)<<  std::setw(20) << "Variable index" << std::setw(20) << "Name"  << std::setw(10) << "Set(0/1)" << std::setw(8) << "value" << std::endl;  
  for(  PFCandidateEGammaExtra::MvaVariable i=PFCandidateEGammaExtra::MVA_FIRST;
	i<PFCandidateEGammaExtra::MVA_LAST;i=PFCandidateEGammaExtra::MvaVariable(i+1)) {
    out << std::setw(20)<< i <<  std::setw(20) << listVar[i] << std::setw(10) << extra.mvaStatus(i) << std::setw(8) << extra.mvaVariable(i) << std::endl;
  }
  
  return out;
}
