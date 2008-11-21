#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCalibrator.h"

// Framework Stuff
#include "FWCore/Framework/interface/ESHandle.h"

// Calo Collections
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTNeighborMap.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTReceiverCard.h"

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
  farmout_(ps.getUntrackedParameter<bool>("FarmoutMode")),
  deltaEtaBarrel_(ps.getParameter<double>("DeltaEtaBarrel")),
  maxEtaBarrel_(ps.getParameter<int>("TowersInBarrel")*deltaEtaBarrel_),
  deltaPhi_(ps.getParameter<double>("TowerDeltaPhi")),
  endcapEta_(ps.getParameter<std::vector<double> >("EndcapEtaBoundaries")),
  fitOpts_((debug_ > -1) ? "ELMRNF" : "QELMRNF"),
  total_(0)
{
  for(int i = 0; i < 28; ++i)
    {
      he_low_smear_[i]  = ((debug_ > 9) ? ( (debug_ > 10) ? 1  : i ) : -999);
      he_high_smear_[i] = ((debug_ > 9) ? ( (debug_ > 10) ? 1  : i ) : -999);
      for(int j = 0; j < 6; ++j)
	{
	  if(j < 3)
	    {
	      ecal_[i][j]      = ((debug_ > 9) ? ( (debug_ > 10) ? ( (j == 2) ? 1 : 0 ) : i*3 + j ) : -999);
	      hcal_[i][j]      = ((debug_ > 9) ? ( (debug_ > 10) ? ( (j == 2) ? 1 : 0 ) : i*3 + j ) : -999);
	      hcal_high_[i][j] = ((debug_ > 9) ? ( (debug_ > 10) ? ( (j == 2) ? 1 : 0 ) : i*3 + j ) : -999);
	    }
	  cross_[i][j] = ((debug_ > 9) ? ( (debug_ > 10) ? 0  : i*6 + j ) : -999);
	}
    }
}

L1RCTCalibrator::~L1RCTCalibrator()
{
  delete rootOut_;
}

void L1RCTCalibrator::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  if(debug_ > 8) return; // don't need to run analyze if we're just making sure parts of the code work or making an empty table

  total_++;

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

  // get L1 Calo Trigger Geometry
  edm::ESHandle<L1CaloGeometry> level1Geometry;
  es.get<L1CaloGeometryRecord>().get(level1Geometry);
  l1Geometry_ = const_cast<L1CaloGeometry*>(level1Geometry.product());

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

  event_ = e.id().event();
  run_ = e.id().run();

  saveCalibrationInfo(the_cands, ecal, hcal, regions);
}

void L1RCTCalibrator::writeHistograms()
{
  for(std::vector<TObject*>::const_iterator i = hists_.begin(); i != hists_.end(); ++i)
    (*i)->Write();
}

void L1RCTCalibrator::beginJob(const edm::EventSetup& es)
{
  if(!sanityCheck()) throw cms::Exception("Failed Sanity Check") << "Coordinate recalculation failed!\n";
  
  rootOut_ = new TFile(outfile_.c_str(),"RECREATE");
  rootOut_->cd();
  
  if(farmout_) putHist(theTree_ = new TTree("L1RCTCalibrator","RCT Calibration Tree"));
  else theTree_ = NULL;		 

  bookHistograms();
}


void L1RCTCalibrator::endJob()
{
  if(debug_ < 9 && !farmout_) postProcessing();
 
  rootOut_->cd();
  writeHistograms();

  rootOut_->Write();
  rootOut_->Close();

  if(!farmout_)
    {
      std::ofstream out;  
      if(debug_ > 0)
	{
	  printCfFragment(std::cout);
	}
      out.open((outfile_ + ((python_) ? std::string("_cff.py") : std::string(".cff"))).c_str());
      printCfFragment(out);
      out.close(); 
    }
}

//This prints out a nicely formatted .cfi file to be included in RCTConfigProducer.cfi
void L1RCTCalibrator::printCfFragment(std::ostream& out) const
{
  double* p = NULL;

  out.flush();
  
  out << ((python_) ? "import FWCore.ParameterSet.Config as cms\n\nrct_calibration = cms.PSet(" : "block rct_calibration = {") << std::endl;
  for(int i = 0; i < 6; ++i)
    {
      switch(i)
	{
	case 0:
	  p = const_cast<double*>(reinterpret_cast<const double*>(ecal_));
	  out << ((python_) ? "\tecal_calib_Lindsey = cms.vdouble(" : "\tvdouble ecal_calib_Lindsey = {") << std::endl;
	  break;
	case 1:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_));
	  out << ((python_) ? "\thcal_calib_Lindsey = cms.vdouble(" : "\tvdouble hcal_calib_Lindsey = {") << std::endl;
	  break;
	case 2:
	  p = const_cast<double*>(reinterpret_cast<const double*>(hcal_high_));
	  out << ((python_) ? "\thcal_high_calib_Lindsey = cms.vdouble(" : "\tvdouble hcal_high_calib_Lindsey = {") << std::endl;
	  break;
	case 3:
	  p = const_cast<double*>(reinterpret_cast<const double*>(cross_));
	  out << ((python_) ? "\tcross_terms_Lindsey = cms.vdouble(" : "\tvdouble cross_terms_Lindsey = {") << std::endl;
	  break;
	case 4:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_low_smear_));
	  out << ((python_) ? "\tHoverE_low_Lindsey = cms.vdouble(" : "\tvdouble HoverE_low_Lindsey = {") << std::endl;
	  break;
	case 5:
	  p = const_cast<double*>(reinterpret_cast<const double*>(he_high_smear_));
	  out << ((python_) ? "\tHoverE_high_Lindsey = cms.vdouble(" : "\tvdouble HoverE_high_Lindsey = {") << std::endl;
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
		  out << ((j==27) ? "" : ",") << std::endl;
		}
	    }
	  else if( p == reinterpret_cast<const double*>(cross_) )
	    {
	      double *q = p + 6*j;
	      if(q[0] != -999 && q[1] != -999 && q[2] != -999 &&
		 q[3] != -999 && q[4] != -999 && q[5] != -999)
		{
		  out << "\t\t" << q[0] << ", " << q[1] << ", " << q[2] << ", "
		      << q[3] << ", " << q[4] << ", " << q[5];
		  out << ((j==27) ? "" : ",") << std::endl;
		}
	    }
	  else
	    {
	      double *q = p;
	      if(q[j] != -999)
		out << "\t\t" << q[j] << ((j==27) ? "" : ",") << std::endl;
	    }
	}
      if(python_)
	{
	  out << ((i != 5) ? "\t)," : "\t)") << std::endl;
	}
      else 
	out << "\t}" << std::endl;
    }
  out << ((python_) ? ")" : "}") << std::endl;
}

double L1RCTCalibrator::uniPhi(const double& phi) const
{
  double result = ((phi < 0) ? phi + 2*M_PI : phi);
  while(result > 2*M_PI) result -= 2*M_PI;
  return result;
}

//calculate Delta R between two (eta,phi) coordinates
void L1RCTCalibrator::deltaR(const double& eta1, double phi1, 
			     const double& eta2, double phi2,double& dr) const
{
  double deta2 = std::pow(eta1-eta2,2.);  
  double dphi2 = std::pow(uniPhi(phi1)-uniPhi(phi2),2.);

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
      veta = maxEtaBarrel_;;
      for(int i = 0; i < offset; ++i)
	veta += endcapEta_[i];
      veta += endcapEta_[offset]/2.;
    }
  veta = ((ieta < 0) ? -veta : veta);
}

void L1RCTCalibrator::phiBin(double vphi, int& iphi) const
{  
  iphi = static_cast<int>(uniPhi(vphi)/deltaPhi_);  
}

void L1RCTCalibrator::phiValue(const int& iphi, double& vphi) const
{
  vphi = iphi*deltaPhi_ + deltaPhi_/2.;
}

L1RCTCalibrator::rct_location L1RCTCalibrator::makeRctLocation(const double& eta, const double& phi) const
{  
  int etab;
  etaBin(eta, etab);
  int phib;
  phiBin(phi,phib);

  return makeRctLocation(etab, phib);
}

L1RCTCalibrator::rct_location L1RCTCalibrator::makeRctLocation(const int& ieta, const int& iphi) const
{
  unsigned ietaAbs = abs(ieta);
  unsigned rctiphi = (72 + 18 - iphi)%72;

  rct_location r;

  r.crate = rctParams()->calcCrate(rctiphi, ieta);
  r.card = rctParams()->calcCard(rctiphi, ietaAbs);

  L1RCTReceiverCard* rC = new L1RCTReceiverCard(r.crate, r.card, NULL);

  r.region = rC->towerToRegionMap(rctParams()->calcTower(rctiphi,ietaAbs)).at(0);

  delete rC;

  return r;
}

bool L1RCTCalibrator::isSelfOrNeighbor(const rct_location& one, const rct_location& two) const
{
  return (one == two || find<rct_location>(two, neighbors(one)) != -1);
}

std::vector<L1RCTCalibrator::rct_location> L1RCTCalibrator::neighbors(const rct_location& loc) const
{
  std::vector<rct_location> result;
  rct_location temp;
  std::vector<int> raw_loc;
  L1RCTNeighborMap rct_map;
  
  raw_loc = rct_map.north(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.south(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.east(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.west(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.se(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.sw(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.ne(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  raw_loc = rct_map.nw(loc.crate,loc.card,loc.region);
  temp.crate = raw_loc.at(0);
  temp.card = raw_loc.at(1);
  temp.region = raw_loc.at(2);
  result.push_back(temp);

  return result;
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

double L1RCTCalibrator::ecalEt(const EcalTriggerPrimitiveDigi& e) const
{
  return (rctParams()->eGammaECalScaleFactors().at(e.id().ietaAbs() - 1)*
	  eScale()->et(e.compressedEt(), e.id().ietaAbs(), e.id().zside()));
}

double L1RCTCalibrator::hcalEt(const HcalTriggerPrimitiveDigi& h) const
{
  return hScale()->et(h.SOI_compressedEt(), h.id().ietaAbs(), h.id().zside());
}

double L1RCTCalibrator::ecalE(const EcalTriggerPrimitiveDigi& e) const
{
  double eta;  
  etaValue(e.id().ietaAbs(), eta);  
  
  double theta = 2*std::atan(std::exp(-eta));
  
  return ecalEt(e)/std::sin(theta);
}

double L1RCTCalibrator::hcalE(const HcalTriggerPrimitiveDigi& h) const
{
  double eta;
  etaValue(h.id().ietaAbs(), eta);

  double theta = 2*std::atan(std::exp(-eta));

  return hcalEt(h)/std::sin(theta);
}

// energy, deltaR
std::pair<double,double> L1RCTCalibrator::showerSize(const std::vector<tpg>& tp, const double frac, const double& max_dr,
						     const bool& ecal, const bool& hcal) const
{
  double c_eta = avgEta(tp), c_phi = avgPhi(tp);
  double result = 0.0, 
    e_max = sumEt(c_eta, c_phi, tp, max_dr, ecal, hcal);
  
  double dr_iter = 0.0;
  
  do{
    result = sumEt(c_eta, c_phi, tp, dr_iter, ecal, hcal);
    dr_iter += 0.01;
  }while(result/e_max < frac);
  
  return std::make_pair(result,dr_iter);
}

double L1RCTCalibrator::sumEt(const double& eta, const double& phi, const std::vector<tpg>& tp, const double& dr, 
			      const bool& ecal, const bool& hcal, const bool& c, const double& crossover) const
{
  double delta_r, tp_phi, tp_eta, sum = 0.0;  

  for(std::vector<tpg>::const_iterator i = tp.begin(); i != tp.end(); ++i)
    {
      etaValue(i->ieta, tp_eta);
      phiValue(i->iphi, tp_phi);

      deltaR(eta,phi,tp_eta,tp_phi,delta_r);

      if(delta_r < dr)
	{
	  if(c)
	    {
	      int etabin = abs(i->ieta) - 1;
	      if(i->ecalE > .5 && ecal)
		{
		  if(ecal_[etabin][0] != -999 && ecal_[etabin][1] != -999 && ecal_[etabin][2] != -999) 		
		    sum += (ecal_[etabin][0]*std::pow(i->ecalEt,3.) +
			    ecal_[etabin][1]*std::pow(i->ecalEt,2.) +
			    ecal_[etabin][2]*i->ecalEt);
		  else
		    sum += i->ecalEt;
		}
	      if(i->hcalE > .5 && hcal)
		{
		  double crossterm = 0.0, hcal_c = 0.0;
		  if(i->ecalEt + i->hcalEt < crossover)
		    {
		      if(cross_[etabin][0] != -999 && cross_[etabin][1] != -999 && cross_[etabin][2] != -999 &&
			 cross_[etabin][3] != -999 && cross_[etabin][4] != -999 && cross_[etabin][5] != -999 &&
			 hcal_[etabin][0] != -999 && hcal_[etabin][1] != -999 && hcal_[etabin][2] != -999)
			{
			  crossterm = (cross_[etabin][0]*std::pow(i->ecalEt,2)*i->hcalEt +
				       cross_[etabin][1]*std::pow(i->hcalEt,2)*i->ecalEt +
				       cross_[etabin][2]*i->ecalEt*i->hcalEt +
				       cross_[etabin][3]*std::pow(i->ecalEt,3)*i->hcalEt +
				       cross_[etabin][4]*std::pow(i->hcalEt,3)*i->ecalEt +
				       cross_[etabin][5]*std::pow(i->ecalEt,2)*std::pow(i->hcalEt,2));
			  hcal_c = (hcal_[etabin][0]*std::pow(i->hcalEt,3.) +
				    hcal_[etabin][1]*std::pow(i->hcalEt,2.) +
				    hcal_[etabin][2]*i->hcalEt);
			}
		      else
			hcal_c = i->hcalEt;
		    }
		  else
		    {
		      if(hcal_high_[etabin][0] != -999 && hcal_high_[etabin][1] != -999 && hcal_high_[etabin][2] != -999)
			{
			  hcal_c = (hcal_high_[etabin][0]*std::pow(i->hcalEt,3.) +
				    hcal_high_[etabin][1]*std::pow(i->hcalEt,2.) +
				    hcal_high_[etabin][2]*i->hcalEt);
			}
		      else
			hcal_c = i->hcalEt;
		    }
		  sum += hcal_c + crossterm;
		}
	    }
	  else
	    {
	      if(i->ecalE > .5 && ecal) sum += i->ecalEt;
	      if(i->hcalE > .5 && hcal) sum += i->hcalEt;
	    }
	}
    }
  return sum;
}

double L1RCTCalibrator::sumEt(const double& eta, const double& phi, const std::vector<region>& regs, const double& dr) const
{
  double sum = 0.0, delta_r, reta, rphi;

  for(std::vector<region>::const_iterator i = regs.begin(); i != regs.end(); ++i)
    {
      rphi = l1Geometry()->emJetPhiBinCenter(i->iphi);
      reta = l1Geometry()->globalEtaBinCenter(i->ieta);

      deltaR(eta,phi,reta,rphi,delta_r);

      if(delta_r < dr) sum += i->linear_et*rctParams()->jetMETLSB();
    }
  return sum;
}

double L1RCTCalibrator::avgPhi(const std::vector<tpg>& t) const
{
  double n = 0.0, d = 0.0;

  for(std::vector<tpg>::const_iterator i = t.begin(); i != t.end(); ++i)
    {
      double temp_phi;
      phiValue(i->iphi, temp_phi);
      n = (i->ecalEt + i->hcalEt)*temp_phi;
      d = i->ecalEt + i->hcalEt;
    }

  return n/d;
}

double L1RCTCalibrator::avgEta(const std::vector<tpg>& t) const
{
  double n = 0.0, d = 0.0;

  for(std::vector<tpg>::const_iterator i = t.begin(); i != t.end(); ++i)
    {
      double temp_eta;
      etaValue(i->ieta, temp_eta);
      n = (i->ecalEt + i->hcalEt)*temp_eta;
      d = i->ecalEt + i->hcalEt;
    }
  return n/d;
}

std::vector<L1RCTCalibrator::tpg> L1RCTCalibrator::tpgsNear(const double& eta, const double& phi, const std::vector<tpg>& tpgs, 
							    const double& dr) const
{
  std::vector<tpg> result;

  for(std::vector<tpg>::const_iterator i = tpgs.begin(); i != tpgs.end(); ++i)
    {
      double delta_r;
      double teta, tphi;
      phiValue(i->iphi, tphi);
      etaValue(i->ieta, teta);
      deltaR(eta,phi,teta,tphi,delta_r);
      
      if(delta_r < dr) result.push_back(*i);
    }
  return result;
}

