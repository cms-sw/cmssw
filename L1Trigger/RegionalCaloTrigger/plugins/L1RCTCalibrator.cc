#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCalibrator.h"

// Framework Stuff
#include "FWCore/Framework/interface/ESHandle.h"

// Gen Collections
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// Calo Collections
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// Reco Collections
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include <fstream>
#include <map>

L1RCTCalibrator::L1RCTCalibrator(edm::ParameterSet const& ps):
  gen(ps.getUntrackedParameter<edm::InputTag>("GeneratorInput",edm::InputTag("genParticles"))),
  rphoton(ps.getUntrackedParameter<edm::InputTag>("RecoPhotonInput",edm::InputTag("photons"))),
  rjet(ps.getUntrackedParameter<edm::InputTag>("RecoJetInput",edm::InputTag("midPointCone5PFJets"))),
  ecalTPG(ps.getUntrackedParameter<edm::InputTag>("EcalTPGInput",edm::InputTag("ecalTriggerPrimitiveDigis"))),
  hcalTPG(ps.getUntrackedParameter<edm::InputTag>("HcalTPGInput",edm::InputTag("hcalTriggerPrimitiveDigis"))),
  regions(ps.getUntrackedParameter<edm::InputTag>("RegionsInput",edm::InputTag("rctDigis"))),
  outfile(ps.getUntrackedParameter<std::string>("OutputFile","RCTCalibration")),
  calib_mode(ps.getUntrackedParameter<std::string>("CalibrationMode","gen")),
  debug(ps.getUntrackedParameter<int>("debug",-1)),
  python(ps.getUntrackedParameter<bool>("PythonOut", true)),
  deltaEtaBarrel(ps.getUntrackedParameter<double>("DeltaEtaBarrel",0.0870)),
  maxEtaBarrel(ps.getUntrackedParameter<int>("TowersInBarrel",20)*deltaEtaBarrel),
  deltaPhi(ps.getUntrackedParameter<double>("TowerDeltaPhi",0.0870)),
  endcapEta(ps.getParameter<std::vector<double> >("EndcapEtaBoundaries")),
  fitOpts((debug > 0) ? "ELM" : "QELM")
{
  // Define allowable calibration types.
  // Can't be put into configuration file since I can't dynamically write
  // what you want me to do.
  allowed_calibs.insert(std::pair<std::string,calib_types>("gen",GEN));
  allowed_calibs.insert(std::pair<std::string,calib_types>("reco",RECO));
  // End

  for(int i = 0; i < 28; ++i)
    {
      he_low_smear[i]  = ((debug > 9) ? i : -999);
      he_high_smear[i] = ((debug > 9) ? i : -999);
      for(int j = 0; j < 6; ++j)
	{
	  if(j < 3)
	    {
	      ecal[i][j]      = ((debug > 9) ? i*3 + j : -999);
	      hcal[i][j]      = ((debug > 9) ? i*3 + j : -999);
	      hcal_high[i][j] = ((debug > 9) ? i*3 + j : -999);
	    }
	  cross[i][j] = ((debug > 9) ? i*6 + j : -999);
	}
    }
}

L1RCTCalibrator::~L1RCTCalibrator()
{
}

void L1RCTCalibrator::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  if(debug > 8) return; // don't need to run analyze if we're just making sure parts of the code work or making an empty table

  // RCT parameters (thresholds, etc)
  edm::ESHandle<L1RCTParameters> rctParameters;
  es.get<L1RCTParametersRcd>().get(rctParameters);
  rctParams = const_cast<L1RCTParameters*>(rctParameters.product());

  // list of RCT channels to mask
  edm::ESHandle<L1RCTChannelMask> channelMask;
  es.get<L1RCTChannelMaskRcd>().get(channelMask);
  chMask = const_cast<L1RCTChannelMask*>(channelMask.product());

  // get energy scale to convert input from ECAL
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  es.get<L1CaloEcalScaleRcd>().get(ecalScale);
  eScale = const_cast<L1CaloEcalScale*>(ecalScale.product());
  
  // get energy scale to convert input from HCAL
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  es.get<L1CaloHcalScaleRcd>().get(hcalScale);
  hScale = const_cast<L1CaloHcalScale*>(hcalScale.product());

  switch(calibration)
    {
    case GEN:
      genCalibration(e);
      break;
    case RECO:
      recoCalibration(e);
      break;
    default:
      cms::Exception("Impossible") << "You really shouldn't ever see this.\n";
    };  
}

void L1RCTCalibrator::beginJob(const edm::EventSetup& es)
{
  std::map<std::string,calib_types>::const_iterator i = allowed_calibs.find(calib_mode);
  if(i == allowed_calibs.end())
    {
      std::stringstream ss;      
      std::map<std::string,calib_types>::const_iterator j = allowed_calibs.begin();
      for(; j != allowed_calibs.end(); ++j)
	ss << j->first << " ";
      throw cms::Exception("Invalid Calibration Type") << "CalibrationMode must be one of: { " << ss.str() << "}.\n";;
    }
  else
    {
      if(debug > -1) edm::LogInfo("L1RCTCalibrator") << "Doing " << i->first << " calibration!";
      calibration = i->second;
    }
  if(!sanityCheck()) throw cms::Exception("Failed Sanity Check") << "Coordinate recalculation failed!\n";
}


void L1RCTCalibrator::endJob()
{
  switch(calibration)
    {
    case GEN:
      genCalibration();
      break;
    case RECO:
      recoCalibration();
      break;
    default:
      cms::Exception("Impossible") << "You really shouldn't ever see this!\n";
    };

  std::ofstream out;  
  if(debug > 0)
    {
      printCfFragment(std::cout);
    }
  out.open((outfile + ((python) ? std::string("_cff.py") : std::string(".cff"))).c_str());
  printCfFragment(out);
  out.close();  
}

//This prints out a nicely formatted .cfi file to be included in RCTConfigProducer.cfi
void L1RCTCalibrator::printCfFragment(std::ostream& out) const
{
  double* p = NULL;

  out << ((python) ? "import FWCore.ParameterSet.Config as cms\n\nrct_calibration = cms.PSet(\n" : "block rct_calibration = {\n");
  for(int i = 0; i < 6; ++i)
    {
      switch(i)
	{
	case 0:
	  p = const_cast<double*>(reinterpret_cast<const double*>(ecal));
	  out << ((python) ? "\tecal_calib_Lindsey = cms.vdouble(\n" : "\tvdouble ecal_calib_Lindsey = {\n");
	  break;
	case 1:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal));
	  out << ((python) ? "\thcal_calib_Lindsey = cms.vdouble(\n" : "\tvdouble hcal_calib_Lindsey = {\n");
	  break;
	case 2:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_high));
	  out << ((python) ? "\thcal_high_calib_Lindsey = cms.vdouble(\n" : "\tvdouble hcal_high_calib_Lindsey = {\n");
	  break;
	case 3:
	  p = const_cast<double*>(reinterpret_cast<const double*>(cross));
	  out << ((python) ? "\tcross_terms_Lindsey = cms.vdouble(\n" : "\tvdouble cross_terms_Lindsey = {\n");
	  break;
	case 4:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_low_smear));
	  out << ((python) ? "\tHoverE_low_Lindsey = cms.vdouble(\n" : "\tvdouble HoverE_low_Lindsey = {\n");
	  break;
	case 5:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_high_smear));
	  out << ((python) ? "\tHoverE_high_Lindsey = cms.vdouble(\n" : "\tvdouble HoverE_high_Lindsey = {\n");
	};

      for(int j = 0; j < 28; ++j)
	{
	  if( p == reinterpret_cast<const double*>(ecal) || p == reinterpret_cast<const double*>(hcal) || 
	      p == reinterpret_cast<const double*>(hcal_high) )
	    {	
	      double *q = p + 3*j;
	      if(q[0] != -999 && q[1] != -999 && q[2] != -999)
		{
		  out << "\t\t" << q[0] << ", " << q[1] << ", " << q[2];
		  out << ((j==27) ? "\n" : ",\n");
		}
	      else
		out << ((j==27) ? "\n" : "");
	    }
	  else if( p == reinterpret_cast<const double*>(cross) )
	    {
	      double *q = p + 6*j;
	      if(q[0] != -999 && q[1] != -999 && q[2] != -999 &&
		 q[3] != -999 && q[4] != -999 && q[5] != -999)
		{
		  out << "\t\t" << q[0] << ", " << q[1] << ", " << q[2] << ", "
		      << q[3] << ", " << q[4] << ", " << q[5];
		  out << ((j==27) ? "\n" : ",\n");
		}
	      else
		out << ((j==27) ? "\n" : "");
	    }
	  else
	    {
	      double *q = p;
	      if(q[j] != -999)
		out << "\t\t" << q[j] << ((j==27) ? "\n" : ",\n");
	      else
		out << ((j==27) ?  "\n" : "");
	    }
	}
      if(python)
	{
	  out << ((i != 5) ? "\t),\n" : "\t)\n");
	}
      else 
	out << ((python) ? "\t)\n" : "\t}\n");
    }
  out << ((python) ? ")\n" : "}\n");
}

//calculate Delta R between two (eta,phi) coordinates
void L1RCTCalibrator::deltaR(const double& eta1, const double& phi1, 
			     const double& eta2, const double& phi2,double& dr) const
{
  double deta2 = std::pow(eta1-eta2,2.),
    tphi1 = ((phi1 < 0) ? phi1 + 2*M_PI : phi1),
    tphi2 = ((phi2 < 0) ? phi2 + 2*M_PI : phi2);

  while(tphi1 > 2*M_PI)
    tphi1 -= 2*M_PI;
  while(tphi2 > 2*M_PI)
    tphi2 -= 2*M_PI;
  
  double dphi2 = std::pow(tphi1-tphi2,2.);

  dr = std::sqrt(deta2 + dphi2);
}

void L1RCTCalibrator::etaBin(const double& veta, int& ieta) const
{
  double absEta = fabs(veta);

  if(absEta < maxEtaBarrel)
    {
      ieta = static_cast<int>((absEta+0.000001)/deltaEtaBarrel) + 1;
    }
  else
    {
      double temp = absEta - maxEtaBarrel;
      int i = 0;
      while(temp > -0.0000001 && i < 8)
	{
	  temp -= endcapEta[i++];
	}
      ieta = 20 + i;
    }
  ieta = ((veta < 0) ? -ieta : ieta);
}
 
void L1RCTCalibrator::etaValue(const int& ieta, double& veta) const
{
  int absEta = abs(ieta);

  if(absEta <= 20)
    {
      veta = (absEta-1)*0.0870 + 0.0870/2.;
    }
  else
    {
      int offset = abs(ieta) - 21;
      veta = 20*0.0870;
      for(int i = 0; i < offset; ++i)
	veta += endcapEta[i];
      veta += endcapEta[offset]/2.;
    }
  veta = ((ieta < 0) ? -veta : veta);
}

void L1RCTCalibrator::phiBin(const double& vphi, int& iphi) const
{
  double tempPhi = ((vphi < 0) ? vphi + 2*M_PI : vphi);
  while(tempPhi > 2*M_PI)
    tempPhi -= 2*M_PI;
  iphi = static_cast<int>(tempPhi/deltaPhi);  
}

void L1RCTCalibrator::phiValue(const int& iphi, double& vphi) const
{
  vphi = iphi*deltaPhi + deltaPhi/2.;
}

bool L1RCTCalibrator::sanityCheck() const
{
  for(int i = 1; i <= 28; ++i)
    {
      int j, l;
      double p, q;
      etaValue(i,p);
      etaBin(p,j);
      etaValue(-i,q);
      etaBin(q,l);
      if( i != j || -i != l)
	{
	  if(debug > -1)
	    edm::LogError("Failed Eta Sanity Check") << i <<  "\t" << p << "\t" << j << "\t" 
						     << -i << "\t" << q << "\t" << l << std::endl;
	  return false;
	}
    }
  for(int i = 0; i < 72; ++i)
    {
      int j;
      double p;
      phiValue(i,p);
      phiBin(p,j);
      if(i != j)
	{
	  if(debug > -1)
	    edm::LogError("Failed Phi Sanity Check") << i << "\t" << p << "\t" << j << std::endl;
	  return false;
	}
    }
  return true;
}

void L1RCTCalibrator::recoCalibration(const edm::Event& e)
{
  view_vector v;
  edm::Handle<cand_view> photons, jets;

  // get the Photon collection
  e.getByLabel(rphoton, photons);  
  // get the Jet collection
  e.getByLabel(rjet, jets);

  v.push_back(photons);
  v.push_back(jets);
  saveCalibrationInfo(v,e);
}

void L1RCTCalibrator::recoCalibration()
{
}

void L1RCTCalibrator::genCalibration(const edm::Event& e)
{
  view_vector v;
  edm::Handle<cand_view> genParticles;

  //get the Gen Particle collection
  e.getByLabel(gen, genParticles);  

  v.push_back(genParticles);
  saveCalibrationInfo(v,e);
}

void L1RCTCalibrator::genCalibration()
{
}

// this gets called ONCE per event!!!!!
void L1RCTCalibrator::saveCalibrationInfo(const view_vector& calib_to,
					  const edm::Event& e)
{  
  event_data temp; // event information to save for reprocessing later
  std::vector<generator>* gtemp = &(temp.gen_particles);
  std::vector<photon>* phtemp = &(temp.photons);
  std::vector<oneprong_jet>* opjtemp = &(temp.jets);
  std::vector<region>* regtemp = &(temp.regions);
  std::vector<tpg>* tpgtemp = &(temp.tpgs);
  edm::Handle<ecal_view> ecaltp;
  edm::Handle<hcal_view> hcaltp;
  edm::Handle<reg_view> regs;

  // get ecal / hcal tpgs and regions
  e.getByLabel(ecalTPG,ecaltp);
  e.getByLabel(hcalTPG,hcaltp);
  e.getByLabel(regions, regs);
 
  view_vector::const_iterator view = calib_to.begin();
  for(; view != calib_to.end(); ++view)
    {
      cand_iter c = (*view)->begin();
      for(; c != (*view)->end(); ++c)
	{
	  cand_view::pointer cand_ = &(*c); // get abstract candidate pointer

	  // ------- you can edit from here --------
	  
	  const reco::Photon* photon_ = dynamic_cast<const reco::Photon*>(cand_);
	  const reco::PFJet* pfjet_ = dynamic_cast<const reco::PFJet*>(cand_);
	  const reco::GenParticle* genp_ = dynamic_cast<const reco::GenParticle*>(cand_);
	  const reco::Jet* jet_ = dynamic_cast<const reco::Jet*>(cand_);

	  if(photon_)
	    {
	      savePhotonInfo(photon_,ecaltp,hcaltp,regs,phtemp,regtemp,tpgtemp);
	    }
	  else if(pfjet_)
	    {
	      savePFJetInfo(pfjet_,ecaltp,hcaltp,regs,opjtemp,regtemp,tpgtemp);
	    }
	  else if(jet_) // place after PFJet since a PFJet can cast to a Jet too!
	    {
	      saveJetInfo(pfjet_,ecaltp,hcaltp,regs,opjtemp,regtemp,tpgtemp);
	    }
	  else if(genp_)
	    {
	      saveGenInfo(genp_,ecaltp,hcaltp,regs,gtemp,regtemp,tpgtemp);
	    }
	  
	  // ------- to here -------
	}
    }

  temp.event = e.id().event();
  temp.run = e.id().run();

  data_.push_back(temp);
}

void L1RCTCalibrator::savePhotonInfo(const reco::Photon* p_, const edm::Handle<ecal_view>& e_, const edm::Handle<hcal_view>& h_, 
				     const edm::Handle<reg_view>& r_, std::vector<photon>* p, std::vector<region>* r,
				     std::vector<tpg>* tp)
{

} 

void L1RCTCalibrator::savePFJetInfo(const reco::PFJet* j_, const edm::Handle<ecal_view>& e_, const edm::Handle<hcal_view>& h_,
				    const edm::Handle<reg_view>& r_, std::vector<oneprong_jet>* j, std::vector<region>* r,
				    std::vector<tpg>* tp)
{
  
}
void L1RCTCalibrator::saveJetInfo(const reco::Jet* j_, const edm::Handle<ecal_view>& e_, const edm::Handle<hcal_view>& h_,
				  const edm::Handle<reg_view>& r_, std::vector<oneprong_jet>* j, std::vector<region>* r,
				  std::vector<tpg>* tp)
{

}

void L1RCTCalibrator::saveGenInfo(const reco::GenParticle* g_ , const edm::Handle<ecal_view>& e_, const edm::Handle<hcal_view>& h_,
				  const edm::Handle<reg_view>& r_, std::vector<generator>* g, std::vector<region>* r,
				  std::vector<tpg>* tp)
{

}
