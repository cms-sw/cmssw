#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCalibrator.h"

// Framework Stuff
#include "FWCore/Framework/interface/ESHandle.h"

// Calo Collections
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include <fstream>
#include <map>

L1RCTCalibrator::L1RCTCalibrator(edm::ParameterSet const& ps):
  cand_inputs_(ps.getParameter<std::vector<edm::InputTag> >("CalibrationInputs")),
  ecalTPG_(ps.getParameter<edm::InputTag>("EcalTPGInput")),
  hcalTPG_(ps.getParameter<edm::InputTag>("HcalTPGInput")),
  regions_(ps.getParameter<edm::InputTag>("RegionsInput")),
  outfile_(ps.getParameter<std::string>("OutputFile")),
  debug_(ps.getUntrackedParameter<int>("debug",-1)),
  python_(ps.getUntrackedParameter<bool>("PythonOut")),
  deltaEtaBarrel_(ps.getUntrackedParameter<double>("DeltaEtaBarrel",0.0870)),
  maxEtaBarrel_(ps.getUntrackedParameter<int>("TowersInBarrel",20)*deltaEtaBarrel_),
  deltaPhi_(ps.getUntrackedParameter<double>("TowerDeltaPhi",0.0870)),
  endcapEta_(ps.getParameter<std::vector<double> >("EndcapEtaBoundaries")),
  fitOpts_((debug_ > 0) ? "ELM" : "QELM")
{
  for(int i = 0; i < 28; ++i)
    {
      he_low_smear_[i]  = ((debug_ > 9) ? i : -999);
      he_high_smear_[i] = ((debug_ > 9) ? i : -999);
      for(int j = 0; j < 6; ++j)
	{
	  if(j < 3)
	    {
	      ecal_[i][j]      = ((debug_ > 9) ? i*3 + j : -999);
	      hcal_[i][j]      = ((debug_ > 9) ? i*3 + j : -999);
	      hcal_high_[i][j] = ((debug_ > 9) ? i*3 + j : -999);
	    }
	  cross_[i][j] = ((debug_ > 9) ? i*6 + j : -999);
	}
    }
}

L1RCTCalibrator::~L1RCTCalibrator()
{
}

void L1RCTCalibrator::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  if(debug_ > 8) return; // don't need to run analyze if we're just making sure parts of the code work or making an empty table

  // RCT parameters (thresholds, etc)
  edm::ESHandle<L1RCTParameters> rctParameters;
  es.get<L1RCTParametersRcd>().get(rctParameters);
  rctParams_ = const_cast<L1RCTParameters*>(rctParameters.product());

  // list of RCT channels to mask
  edm::ESHandle<L1RCTChannelMask> channelMask;
  es.get<L1RCTChannelMaskRcd>().get(channelMask);
  chMask_ = const_cast<L1RCTChannelMask*>(channelMask.product());

  // get energy scale to convert input from ECAL
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  es.get<L1CaloEcalScaleRcd>().get(ecalScale);
  eScale_ = const_cast<L1CaloEcalScale*>(ecalScale.product());
  
  // get energy scale to convert input from HCAL
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  es.get<L1CaloHcalScaleRcd>().get(hcalScale);
  hScale_ = const_cast<L1CaloHcalScale*>(hcalScale.product());

  view_vector the_cands(cand_inputs_.size());

  std::vector<edm::InputTag>::const_iterator i = cand_inputs_.begin();
  view_vector::iterator j = the_cands.begin();

  for(; i != cand_inputs_.end(); ++i, ++j)
    {
      e.getByLabel((*i),(*j));
    }

  edm::Handle<ecal_view> ecal;
  edm::Handle<hcal_view> hcal;
  edm::Handle<reg_view> regions;
  
  e.getByLabel(ecalTPG_,ecal);
  e.getByLabel(hcalTPG_,hcal);
  e.getByLabel(regions_,regions);

  saveCalibrationInfo(the_cands, ecal, hcal, regions);
}

void L1RCTCalibrator::beginJob(const edm::EventSetup& es)
{
  if(!sanityCheck()) throw cms::Exception("Failed Sanity Check") << "Coordinate recalculation failed!\n";
  bookHistograms();
}


void L1RCTCalibrator::endJob()
{
  postProcessing();

  std::ofstream out;  
  if(debug_ > 0)
    {
      printCfFragment(std::cout);
    }
  out.open((outfile_ + ((python_) ? std::string("_cff.py") : std::string(".cff"))).c_str());
  printCfFragment(out);
  out.close();  
}

//This prints out a nicely formatted .cfi file to be included in RCTConfigProducer.cfi
void L1RCTCalibrator::printCfFragment(std::ostream& out) const
{
  double* p = NULL;

  out << ((python_) ? "import FWCore.ParameterSet.Config as cms\n\nrct_calibration = cms.PSet(\n" : "block rct_calibration = {\n");
  for(int i = 0; i < 6; ++i)
    {
      switch(i)
	{
	case 0:
	  p = const_cast<double*>(reinterpret_cast<const double*>(ecal_));
	  out << ((python_) ? "\tecal_calib_Lindsey = cms.vdouble(\n" : "\tvdouble ecal_calib_Lindsey = {\n");
	  break;
	case 1:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_));
	  out << ((python_) ? "\thcal_calib_Lindsey = cms.vdouble(\n" : "\tvdouble hcal_calib_Lindsey = {\n");
	  break;
	case 2:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_high_));
	  out << ((python_) ? "\thcal_high_calib_Lindsey = cms.vdouble(\n" : "\tvdouble hcal_high_calib_Lindsey = {\n");
	  break;
	case 3:
	  p = const_cast<double*>(reinterpret_cast<const double*>(cross_));
	  out << ((python_) ? "\tcross_terms_Lindsey = cms.vdouble(\n" : "\tvdouble cross_terms_Lindsey = {\n");
	  break;
	case 4:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_low_smear_));
	  out << ((python_) ? "\tHoverE_low_Lindsey = cms.vdouble(\n" : "\tvdouble HoverE_low_Lindsey = {\n");
	  break;
	case 5:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_high_smear_));
	  out << ((python_) ? "\tHoverE_high_Lindsey = cms.vdouble(\n" : "\tvdouble HoverE_high_Lindsey = {\n");
	};

      for(int j = 0; j < 28; ++j)
	{
	  if( p == reinterpret_cast<const double*>(ecal_) || p == reinterpret_cast<const double*>(hcal_) || 
	      p == reinterpret_cast<const double*>(hcal_high_) )
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
	  else if( p == reinterpret_cast<const double*>(cross_) )
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
      if(python_)
	{
	  out << ((i != 5) ? "\t),\n" : "\t)\n");
	}
      else 
	out << ((python_) ? "\t)\n" : "\t}\n");
    }
  out << ((python_) ? ")\n" : "}\n");
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

  if(absEta < maxEtaBarrel_)
    {
      ieta = static_cast<int>((absEta+0.000001)/deltaEtaBarrel_) + 1;
    }
  else
    {
      double temp = absEta - maxEtaBarrel_;
      int i = 0;
      while(temp > -0.0000001 && i < 8)
	{
	  temp -= endcapEta_[i++];
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
	veta += endcapEta_[i];
      veta += endcapEta_[offset]/2.;
    }
  veta = ((ieta < 0) ? -veta : veta);
}

void L1RCTCalibrator::phiBin(const double& vphi, int& iphi) const
{
  double tempPhi = ((vphi < 0) ? vphi + 2*M_PI : vphi);
  while(tempPhi > 2*M_PI)
    tempPhi -= 2*M_PI;
  iphi = static_cast<int>(tempPhi/deltaPhi_);  
}

void L1RCTCalibrator::phiValue(const int& iphi, double& vphi) const
{
  vphi = iphi*deltaPhi_ + deltaPhi_/2.;
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
	  if(debug_ > -1)
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
	  if(debug_ > -1)
	    edm::LogError("Failed Phi Sanity Check") << i << "\t" << p << "\t" << j << std::endl;
	  return false;
	}
    }
  return true;
}

