#include "RecoParticleFlow/PFRootEvent/interface/METManager.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

METManager::METManager(std::string Filename)
{
  outmetfilename_=Filename;
  std::cout << "Info: DQM is not yet used in METManager."<<std::endl;
  std::cout << "pfMET validation histograms will be saved to '" << outmetfilename_ << "'" << std::endl;
  outfile_ = new TFile(outmetfilename_.c_str(), "RECREATE");
  //void setup(DQMStore *, bool PlotAgainstReco, minDeltaEt,
  //           float maxDeltaEt, float minDeltaPhi, float maxDeltaPhi);
  //GenBenchmark_.setup(NULL, true, -200., 200., -3.2, 3.2);
}

void METManager::FillHisto(std::string Name)
{
  //std::cout << "Name = " << Name << std::endl;
  const std::string fullname=outmetfilename_+":/PFTask/Benchmarks/"+Name;

  std::map<std::string, GenericBenchmark>::const_iterator i = GenBenchmarkMap_.find( Name );
  if ( i == GenBenchmarkMap_.end() )
  {
    std::cout << "Error in METManager::FillHisto(string): " << Name << " is not in GenBenchmarkMap_" << std::endl;
  }
  else
  {
    GenBenchmarkMap_[Name].setfile(outfile_);
    std::vector<reco::MET> vmet1;
    std::vector<reco::MET> vmet2;
    vmet1.push_back(MET1_);
    vmet2.push_back(MET2_);
    GenBenchmarkMap_[Name].fill(&vmet2,&vmet1,true,false,false,-1.,-1.,-1.,999.);
    //std::cout << "MET1_.met = " << MET1_.pt() << std::endl;
    //std::cout << "MET1_.phi = " << MET1_.phi() << std::endl;
    //std::cout << "MET1_.sumEt = " << MET1_.sumEt() << std::endl;
  }
  return;
}

void METManager::write()
{
  if (GenBenchmarkMap_.size()>0)
  {
    std::map<std::string, GenericBenchmark>::iterator i = GenBenchmarkMap_.begin();
    ((*i).second).write(outmetfilename_);
  }
}

void METManager::setMET1(const reco::MET *met)
{
  MET1_=*met;
}

void METManager::setMET2(const reco::MET *met)
{
  MET2_=*met;
}

void METManager::setMET1(const reco::GenParticleCollection *genParticleList)
{
  MET1_=computeGenMET(genParticleList);
}

void METManager::setMET2(const reco::GenParticleCollection *genParticleList)
{
  MET2_=computeGenMET(genParticleList);
}

void METManager::setMET1(const reco::PFCandidateCollection& pfCandidates)
{
  MET1_=recomputePFMET(pfCandidates);
}

void METManager::setMET2(const reco::PFCandidateCollection& pfCandidates)
{
  MET2_=recomputePFMET(pfCandidates);
}

void METManager::SetIgnoreParticlesIDs(const std::vector<unsigned int>* vIgnoreParticlesIDs)
{
  vIgnoreParticlesIDs_=(*vIgnoreParticlesIDs);
}

void METManager::SetSpecificIdCut(const std::vector<unsigned int>* Id, const std::vector<double>* Eta)
{
  trueMetSpecificIdCut_=(*Id);
  trueMetSpecificEtaCut_=(*Eta);
}

reco::MET METManager::computeGenMET(const reco::GenParticleCollection *genParticleList) const
{

  double trueMEY  = 0.0;
  double trueMEX  = 0.0;;
  double true_met = 0.0;;
  double true_set  = 0.0;;

  //std::cout << "(*genParticleList).size() = " << (*genParticleList).size() << std::endl;
  for( unsigned i = 0; i < (*genParticleList).size(); i++ ) {

    //std::cout << "(*genParticleList)[i].eta() = " << (*genParticleList)[i].eta() << std::endl;

    if( (*genParticleList)[i].status() == 1 && fabs((*genParticleList)[i].eta()) < 5.0 ) { 

      bool ignoreThisPart=false;
      if (vIgnoreParticlesIDs_.size()==0) std::cout << "Warning : METManager: vIgnoreParticlesIDs_.size()==0" << std::endl;
      for (unsigned int idc=0;idc<vIgnoreParticlesIDs_.size();++idc)
      {
	if(abs((*genParticleList)[i].pdgId()) == (int)vIgnoreParticlesIDs_[idc]) 
	  ignoreThisPart=true;
      }
      for (unsigned int specificIdc=0;specificIdc<trueMetSpecificIdCut_.size();++specificIdc)
      {
        if (abs((*genParticleList)[i].pdgId())== (int)trueMetSpecificIdCut_[specificIdc] && 
	    fabs((*genParticleList)[i].eta()) > trueMetSpecificEtaCut_[specificIdc]) 
	  ignoreThisPart=true;
      }

      if (!ignoreThisPart) {
	//trueMEX -= (*genParticleList)[i].px();
	//trueMEY -= (*genParticleList)[i].py();
	trueMEX -= (*genParticleList)[i].et()*cos((*genParticleList)[i].phi());
	trueMEY -= (*genParticleList)[i].et()*sin((*genParticleList)[i].phi());
	true_set += (*genParticleList)[i].pt();
      }
    }
  }
  true_met = sqrt( trueMEX*trueMEX + trueMEY*trueMEY );
  //const double true_phi = atan2(trueMEY,trueMEX);
  //std::cout << "true_met = " << true_met << std::endl;
  //std::cout << "true_phi = " << true_phi << std::endl;
  math::XYZTLorentzVector p4(trueMEX, trueMEY, 0., true_met);
  math::XYZPoint vtx(0.,0.,0.); 
  return reco::MET(true_set,p4,vtx);
}

void METManager::addGenBenchmark(std::string GenBenchmarkName)
{
  GenericBenchmark GenBenchmark;
  std::string path = outmetfilename_+":/PFTask";
  if (GenBenchmarkMap_.size()==0)
  {
    //gDirectory->pwd();
    outfile_->mkdir("PFTask");
    //std::cout << "FL : path.c_str() = " << path.c_str() << std::endl;
    //outfile_->pwd();
    //outfile_->cd(path.c_str());
    gDirectory->cd(path.c_str());
    //gDirectory->pwd();
    //outfile_->pwd();
    gDirectory->mkdir("Benchmarks");
    //outfile_->mkdir("Benchmarks");
    //outfile_->pwd();
  }
  path = outmetfilename_+":/PFTask/Benchmarks";
  gDirectory->cd(path.c_str());
  gDirectory->mkdir(GenBenchmarkName.c_str());
  path = outmetfilename_+":/PFTask/Benchmarks/"+GenBenchmarkName;
  //std::cout << "FL : path.c_str() = " << path.c_str() << std::endl;
  gDirectory->cd(path.c_str());
  //gDirectory->pwd();
  //const std::string path = outmetfilename_+":/PFTask/Benchmarks/"+GenBenchmarkName; //+ "/";
  //std::cout << "path.c_str() = " << path.c_str() << std::endl;
  //void setup(DQMStore *, bool PlotAgainstReco, minDeltaEt,
  //           float maxDeltaEt, float minDeltaPhi, float maxDeltaPhi);
  GenBenchmark.setfile(outfile_);
  GenBenchmark.setup(NULL, true, -200., 200., -3.2, 3.2, true);

  GenBenchmarkMap_[GenBenchmarkName]=GenBenchmark;
  //GenBenchmarkV_.push_back(GenBenchmark);
}

reco::MET METManager::recomputePFMET(const reco::PFCandidateCollection& pfCandidates) const
{

  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;

  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;

  double NeutralEMEt = 0.0;
  double NeutralHadEt = 0.0;
  double ChargedEMEt = 0.0;
  double ChargedHadEt = 0.0;
  double MuonEt = 0.0;
  double type6Et = 0.0;
  double type7Et = 0.0;

  for (unsigned int pfc=0;pfc<pfCandidates.size();++pfc) {
    double phi   = pfCandidates[pfc].phi();
    double theta = pfCandidates[pfc].theta();
    double e     = pfCandidates[pfc].energy();
    double et    = e*sin(theta);
    sum_ez += e*cos(theta);
    sum_et += et;
    sum_ex += et*cos(phi);
    sum_ey += et*sin(phi);

    // compute met specific data:
    if (pfCandidates[pfc].particleId() == 1) ChargedHadEt += et;
    if (pfCandidates[pfc].particleId() == 2) ChargedEMEt += et;
    if (pfCandidates[pfc].particleId() == 3) MuonEt += et;
    if (pfCandidates[pfc].particleId() == 4) NeutralEMEt += et;
    if (pfCandidates[pfc].particleId() == 5) NeutralHadEt += et;
    if (pfCandidates[pfc].particleId() == 6) type6Et += et;
    if (pfCandidates[pfc].particleId() == 7) type7Et += et;

  }

  const double Et_total=NeutralEMEt+NeutralHadEt+ChargedEMEt+ChargedHadEt+MuonEt+type6Et+type7Et;

  double met = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  const LorentzVector p4( -sum_ex, -sum_ey, 0.0, met);
  const Point vtx(0.0,0.0,0.0);
 
  SpecificPFMETData specific;
  specific.NeutralEMFraction = NeutralEMEt/Et_total;
  specific.NeutralHadFraction = NeutralHadEt/Et_total;
  specific.ChargedEMFraction = ChargedEMEt/Et_total;
  specific.ChargedHadFraction = ChargedHadEt/Et_total;
  specific.MuonFraction = MuonEt/Et_total;
  specific.Type6Fraction = type6Et/Et_total;
  specific.Type7Fraction = type7Et/Et_total;

  reco::PFMET specificPFMET( specific, sum_et, p4, vtx );
  return specificPFMET;
}

void METManager::coutTailEvents(const int entry, const double DeltaMETcut, const double DeltaPhicut, const double MET1cut) const
{
  const double deltaMET=MET2_.pt()-MET1_.pt();
  const double deltaPhi=PFBenchmarkAlgo::deltaPhi(&MET1_,&MET2_);
  //std::cout << "Delta Phi = " << deltaPhi << std::endl;
  //std::cout << "fabs(Delta Phi) = " << fabs(deltaPhi) << std::endl;

  if (MET1_.pt()>MET1cut && (fabs(deltaMET)>DeltaMETcut || fabs(deltaPhi)>DeltaPhicut))
  {
    std::cout << " =====================PFMETBenchmark =================" << std::endl;
    std::cout << "process entry "<< entry << std::endl;
    std::cout << "Delta MET = " << deltaMET << std::endl;
    std::cout << "MET1 = " << MET1_.pt() << std::endl;
    std::cout << "MET2 = " << MET2_.pt() << std::endl;

    std::cout << "Delta Phi = " << deltaPhi << std::endl;
    std::cout << "Phi1 = " << MET1_.phi() << std::endl;
    std::cout << "Phi2 = " << MET2_.phi() << std::endl;

  }
}

void METManager::propagateJECtoMET1(const std::vector<reco::CaloJet> caloJets,
		    const std::vector<reco::CaloJet> corr_caloJets)
{
  MET1_=propagateJEC(MET1_,caloJets,corr_caloJets);
}

void METManager::propagateJECtoMET2(const std::vector<reco::CaloJet> caloJets,
		    const std::vector<reco::CaloJet> corr_caloJets)
{
  MET2_=propagateJEC(MET2_,caloJets,corr_caloJets);
}

reco::MET METManager::propagateJEC(const reco::MET &MET, const std::vector<reco::CaloJet> caloJets,
			 const std::vector<reco::CaloJet> corr_caloJets) const
{

  //std::cout << "FL : MET = " << MET.pt() << std::endl;
  double caloJetCorPX = 0.0;
  double caloJetCorPY = 0.0;
  double caloJetCorSET = 0.0;

  if (caloJets.size()>0 && corr_caloJets.size()==0) std::cout << "No corrected calo jets found !" << std::endl;
  //std::cout << "caloJets.size() = " << caloJets.size() << std::endl;
  //std::cout << "corr_caloJets.size() = " << corr_caloJets.size() << std::endl;

  for(unsigned int caloJetc=0;caloJetc<caloJets.size();++caloJetc)
  {
    //std::cout << "caloJets[" << caloJetc << "].pt() = " << caloJets[caloJetc].pt() << std::endl;
    //std::cout << "caloJets[" << caloJetc << "].phi() = " << caloJets[caloJetc].phi() << std::endl;
    //std::cout << "caloJets[" << caloJetc << "].eta() = " << caloJets[caloJetc].eta() << std::endl;
    //}
    for(unsigned int corr_caloJetc=0;corr_caloJetc<corr_caloJets.size();++corr_caloJetc)
    {
      //std::cout << "corr_caloJets[" << corr_caloJetc << "].pt() = " << corr_caloJets[corr_caloJetc].pt() << std::endl;
      //std::cout << "corr_caloJets[" << corr_caloJetc << "].phi() = " << corr_caloJets[corr_caloJetc].phi() << std::endl;
      //std::cout << "corr_caloJets[" << corr_caloJetc << "].eta() = " << corr_caloJets[corr_caloJetc].eta() << std::endl;
      //}
      Float_t DeltaPhi = corr_caloJets[corr_caloJetc].phi() - caloJets[caloJetc].phi();
      Float_t DeltaEta = corr_caloJets[corr_caloJetc].eta() - caloJets[caloJetc].eta();
      Float_t DeltaR2 = DeltaPhi*DeltaPhi + DeltaEta*DeltaEta;
      if( DeltaR2 < 0.0001 && caloJets[caloJetc].pt() > 20.0 ) 
      {
	caloJetCorPX  += (corr_caloJets[corr_caloJetc].px() - caloJets[caloJetc].px());
	caloJetCorPY  += (corr_caloJets[corr_caloJetc].py() - caloJets[caloJetc].py());
	caloJetCorSET += (corr_caloJets[corr_caloJetc].pt() - caloJets[caloJetc].pt());
      }
    }
  }
  const double corr_calomet=sqrt((MET.px()-caloJetCorPX)*(MET.px()-caloJetCorPX)+(MET.py()-caloJetCorPY)*(MET.py()-caloJetCorPY));
  const double corr_set=MET.sumEt()+caloJetCorSET;
  //calo_phi = atan2((cm.py()-caloJetCorPY),(cm.px()-caloJetCorPX));
  math::XYZTLorentzVector p4(MET.px()-caloJetCorPX, MET.py()-caloJetCorPY, 0., corr_calomet);
  math::XYZPoint vtx(0.,0.,0.);

  //std::cout << "FL : corrMET = " << corr_calomet << std::endl;
  return reco::MET(corr_set,p4,vtx);
}
