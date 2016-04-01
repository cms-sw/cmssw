// -*- C++ -*-
//
// Package:    HiJetBackground/HiFJGridEmptyAreaCalculator
// Class:      HiFJGridEmptyAreaCalculator
// Based on:   fastjet/tools/GridMedianBackgroundEstimator 
//
// Original Author:  Doga Gulhan
//         Created:  Wed Mar 16 14:00:04 CET 2016
//
//

#include "RecoHI/HiJetAlgos/plugins/HiFJGridEmptyAreaCalculator.h"
using namespace std;
using namespace edm;


HiFJGridEmptyAreaCalculator::HiFJGridEmptyAreaCalculator(const edm::ParameterSet& iConfig):
  gridWidth_(iConfig.getUntrackedParameter<double>("gridWidth",0.005)),
  band_(iConfig.getUntrackedParameter<double>("bandWidth",0.2)),
  hiBinCut_(iConfig.getUntrackedParameter<int>("hiBinCut",60)),
  doCentrality_(iConfig.getUntrackedParameter<bool>("doCentrality",true))
{
  pfCandsToken_ = consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>( "pfCandSource" ));
  mapEtaToken_ = consumes<std::vector<double> >(iConfig.getParameter<edm::InputTag>( "mapEtaEdges" ));
  mapRhoToken_ = consumes<std::vector<double> >(iConfig.getParameter<edm::InputTag>( "mapToRho" ));
  mapRhoMToken_ = consumes<std::vector<double> >(iConfig.getParameter<edm::InputTag>( "mapToRhoM" ));
  jetsToken_ = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>( "jetSource" ));
  centralityBinToken_ = consumes<int>(iConfig.getParameter<edm::InputTag>("CentralityBinSrc"));

  //register your products
  produces<std::vector<double > >("mapEmptyCorrFac"); 
  produces<std::vector<double > >("mapToRhoCorr");
  produces<std::vector<double > >("mapToRhoMCorr");
  produces<std::vector<double > >("mapToRhoCorr1Bin");
  produces<std::vector<double > >("mapToRhoMCorr1Bin");
  //rho calculation on a grid using median
  produces<std::vector<double > >("mapRhoVsEtaGrid");
  produces<std::vector<double > >("mapMeanRhoVsEtaGrid");
  produces<std::vector<double > >("mapEtaMaxGrid");
  produces<std::vector<double > >("mapEtaMinGrid");
  
}


HiFJGridEmptyAreaCalculator::~HiFJGridEmptyAreaCalculator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to produce the data  ------------
void
HiFJGridEmptyAreaCalculator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  //Values from rho calculator  
  edm::Handle<std::vector<double>> mapEtaRanges;
  iEvent.getByToken(mapEtaToken_, mapEtaRanges);
  
  edm::Handle<std::vector<double>> mapRho;
  iEvent.getByToken(mapRhoToken_, mapRho);

  edm::Handle<std::vector<double>> mapRhoM;  
  iEvent.getByToken(mapRhoMToken_, mapRhoM);

  // if doCentrality is set to true calculate empty area correction
  // for only events with hiBin > hiBinCut  
  int hiBin = -1;
  bool doEmptyArea = true; 
  if(doCentrality_){
   edm::Handle<int> cbin_;
   iEvent.getByToken(centralityBinToken_,cbin_);
   hiBin = *cbin_;
   
   if(hiBin < hiBinCut_) doEmptyArea = false;
  }
  
  //Define output vectors
  int neta = (int)mapEtaRanges->size();
   
  std::auto_ptr<std::vector<double>> mapToRhoCorrOut ( new std::vector<double>(neta-1,1e-6));
  std::auto_ptr<std::vector<double>> mapToRhoMCorrOut ( new std::vector<double>(neta-1,1e-6));
  std::auto_ptr<std::vector<double>> mapToRhoCorr1BinOut ( new std::vector<double>(neta-1,1e-6));
  std::auto_ptr<std::vector<double>> mapToRhoMCorr1BinOut ( new std::vector<double>(neta-1,1e-6));

  setup_grid(mapEtaRanges->at(0), mapEtaRanges->at(neta-1));
  
  //calculate empty area correction over full acceptance leaving eta bands on the sides
  double all_acceptance_corr = 1;
  if(doEmptyArea){
   _eta_min_jet = mapEtaRanges->at(0) - band_;
   _eta_max_jet = mapEtaRanges->at(neta-1) + band_;
  
   calculate_area_fraction_of_jets(iEvent, iSetup);
  
   all_acceptance_corr =  _total_inbound_area;
  }
  
  //calculate empty area correction in each eta range
  for(int ieta = 0; ieta<(neta-1); ieta++) {
   
   double correction_kt = 1;   
   double rho = mapRho->at(ieta);
   double rhoM = mapRhoM->at(ieta);
    
   if(doEmptyArea){
    double eta_min = mapEtaRanges->at(ieta);
    double eta_max = mapEtaRanges->at(ieta+1);
      
    _eta_min_jet = eta_min + band_;  
    _eta_max_jet = eta_max - band_;  
   
    calculate_area_fraction_of_jets(iEvent, iSetup);
    correction_kt = _total_inbound_area;
   }
   
   mapToRhoCorrOut->at(ieta) = correction_kt*rho;
   mapToRhoMCorrOut->at(ieta) = correction_kt*rhoM;
   
   mapToRhoCorr1BinOut->at(ieta) = all_acceptance_corr*rho;
   mapToRhoMCorr1BinOut->at(ieta) = all_acceptance_corr*rhoM;
  }

  iEvent.put(mapToRhoCorrOut,"mapToRhoCorr");
  iEvent.put(mapToRhoMCorrOut,"mapToRhoMCorr");
  iEvent.put(mapToRhoCorr1BinOut,"mapToRhoCorr1Bin");
  iEvent.put(mapToRhoMCorr1BinOut,"mapToRhoMCorr1Bin");
  
  //calculate rho from grid as a function of eta over full range using PF candidates
  calculate_grid_rho(iEvent, iSetup);

  std::auto_ptr<std::vector<double>> mapRhoVsEtaGridOut ( new std::vector<double>(_ny,0.));
  std::auto_ptr<std::vector<double>> mapMeanRhoVsEtaGridOut ( new std::vector<double>(_ny,0.));
  std::auto_ptr<std::vector<double>> mapEtaMaxGridOut ( new std::vector<double>(_ny,0.));
  std::auto_ptr<std::vector<double>> mapEtaMinGridOut ( new std::vector<double>(_ny,0.));
  for(int ieta = 0; ieta < _ny; ieta++) {
   mapRhoVsEtaGridOut->at(ieta) = _rho_vs_eta[ieta];
   mapMeanRhoVsEtaGridOut->at(ieta) = _mean_rho_vs_eta[ieta];
   mapEtaMaxGridOut->at(ieta) = _eta_max_grid[ieta];
   mapEtaMinGridOut->at(ieta) = _eta_min_grid[ieta];
  }

  iEvent.put(mapRhoVsEtaGridOut,"mapRhoVsEtaGrid");
  iEvent.put(mapMeanRhoVsEtaGridOut,"mapMeanRhoVsEtaGrid");
  iEvent.put(mapEtaMaxGridOut,"mapEtaMaxGrid");
  iEvent.put(mapEtaMinGridOut,"mapEtaMinGrid");
}

//----------------------------------------------------------------------
// setting a new event
//----------------------------------------------------------------------
// tell the background estimator that it has a new event, composed
// of the specified particles.
void
HiFJGridEmptyAreaCalculator::calculate_grid_rho(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  vector<vector<double>> scalar_pt(_ny, vector<double>(_nphi, 0.0));
  
  edm::Handle<reco::PFCandidateCollection> pfCands;
  iEvent.getByToken(pfCandsToken_, pfCands);
  const reco::PFCandidateCollection *pfCandidateColl = pfCands.product();
  for(unsigned icand = 0; icand < pfCandidateColl->size(); icand++) {
   const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
   //use ony the particles within the eta range
   if (pfCandidate.eta() < _ymin || pfCandidate.eta() > _ymax ) continue;
   int jeta = tile_index_eta(&pfCandidate);
   int jphi = tile_index_phi(&pfCandidate);
   scalar_pt[jeta][jphi] += pfCandidate.pt();
  }
  
 _rho_vs_eta.resize(_ny);
 _mean_rho_vs_eta.resize(_ny);
  for(int jeta = 0; jeta < _ny; jeta++){
  
 	 _rho_vs_eta[jeta] = 0;
 	 _mean_rho_vs_eta[jeta] = 0;
	 vector<double> rho_vs_phi;
	 int n_empty = 0;
	
	for(int jphi = 0; jphi < _nphi; jphi++){
	  double binpt = scalar_pt[jeta][jphi];
	  _mean_rho_vs_eta[jeta] += binpt;
      //fill in the vector for median calculation
	  if(binpt > 0) rho_vs_phi.push_back(binpt);
	  else n_empty++;
	 } 
	 _mean_rho_vs_eta[jeta] /= ((double)_nphi);
     _mean_rho_vs_eta[jeta] /= _tile_area;

	 //median calculation
	 sort(rho_vs_phi.begin(), rho_vs_phi.end());
	 //use only the nonzero grid cells for median calculation;
	 int n_full = _nphi - n_empty;
	 if(n_full == 0){
 	  _rho_vs_eta[jeta] = 0;
 	  continue;
	 }
	 if (n_full  % 2 == 0)
     {
       _rho_vs_eta[jeta] = (rho_vs_phi[(int)(n_full / 2 - 1)] + rho_vs_phi[(int)(n_full / 2)]) / 2;
     }
     else 
     {
       _rho_vs_eta[jeta] = rho_vs_phi[(int)(n_full / 2)];
     }
     //correct for empty cells 
	 _rho_vs_eta[jeta] *= (((double) n_full)/((double) _nphi));
	 //normalize to area
     _rho_vs_eta[jeta] /= _tile_area;
  }
}

void
HiFJGridEmptyAreaCalculator::calculate_area_fraction_of_jets(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(jetsToken_, jets);
  
  //calculate jet kt area fraction inside boundary by grid
  _total_inbound_area = 0;
  
  for(auto jet = jets->begin(); jet != jets->end(); ++jet) {
   if (jet->eta() < _eta_min_jet || jet->eta() > _eta_max_jet) continue;
   
   double area_kt = jet->jetArea(); 
   setup_grid_jet(&*jet);
   std::vector<std::pair<int, int> > pf_indices_jet;
   std::vector<std::pair<int, int> > pf_indices_jet_inbound;
   int n_constit_jet = 0;
   int n_constit_jet_inbound = 0;
   for(auto daughter : jet->getJetConstituentsQuick()){
   	auto pfCandidate = static_cast<const reco::PFCandidate*>(daughter);

	int jeta = tile_index_eta_jet(&*pfCandidate);
	int jphi = tile_index_phi(&*pfCandidate);
	pf_indices_jet.push_back(std::make_pair(jphi, jeta));
	n_constit_jet++;
	if (pfCandidate->eta() < _eta_min_jet && pfCandidate->eta() > _eta_max_jet) continue;
	pf_indices_jet_inbound.push_back(std::make_pair(jphi, jeta));
	n_constit_jet_inbound++;
   }   
   
   //if the jet is well within the eta range just add the area
   if(n_constit_jet == n_constit_jet_inbound){
	_total_inbound_area += area_kt;
    continue;
   }
   
   //for jets that fall outside of eta range calculate fraction of area
   //inside the range with a grid
   int nthis = 0;
   if (n_constit_jet > 0) nthis = num_jet_grid_cells(pf_indices_jet);

   int nthis_inbound = 0;
   if (n_constit_jet_inbound > 0) nthis_inbound = num_jet_grid_cells(pf_indices_jet_inbound);
   
  
   double fraction_area = ((double)nthis_inbound)/((double)nthis);
   _total_inbound_area += area_kt*fraction_area;    
  } 
  
  //divide by the total area in that range
  _total_inbound_area /= ((_eta_max_jet - _eta_min_jet)*twopi);
  
  //the fraction can still be greater than 1 because kt area fraction inside 
  //the range can differ from what we calculated with the grid
  if (_total_inbound_area > 1) _total_inbound_area = 1;
}


// #ifndef FASTJET_GMBGE_USEFJGRID
//----------------------------------------------------------------------
// protected material
//----------------------------------------------------------------------
// configure the grid
void
HiFJGridEmptyAreaCalculator::setup_grid(double eta_min, double eta_max) {

  // since we've exchanged the arguments of the grid constructor,
  // there's a danger of calls with exchanged ymax,spacing arguments -- 
  // the following check should catch most such situations.
  _ymin = eta_min;
  _ymax = eta_max;
  
  assert(_ymax - _ymin >= gridWidth_);

  // this grid-definition code is becoming repetitive -- it should
  // probably be moved somewhere central...
  double ny_double = (_ymax-_ymin) / gridWidth_;
  _ny = int(ny_double+0.5);
  _dy = (_ymax-_ymin) / _ny;
  
  _nphi = int (twopi / gridWidth_ + 0.5);
  _dphi = twopi / _nphi;

  // some sanity checking (could throw a fastjet::Error)
  assert(_ny >= 1 && _nphi >= 1);

  _ntotal = _nphi * _ny;
  //_scalar_pt.resize(_ntotal);
  _tile_area = _dy * _dphi;
  
  
  _eta_max_grid.resize(_ny);
  _eta_min_grid.resize(_ny);
  for(int jeta = 0; jeta < _ny; jeta++){
   _eta_min_grid[jeta] = eta_min + _dy*((double)jeta);
   _eta_max_grid[jeta] = eta_min + _dy*((double)jeta + 1.);
  }
}

//----------------------------------------------------------------------
// retrieve the grid tile index for a given PseudoJet
int
HiFJGridEmptyAreaCalculator::tile_index_phi(const reco::PFCandidate *pfCand)  {
  // directly taking int does not work for values between -1 and 0
  // so use floor instead
  // double iy_double = (p.rap() - _ymin) / _dy;
  // if (iy_double < 0.0) return -1;
  // int iy = int(iy_double);
  // if (iy >= _ny) return -1;

  // writing it as below gives a huge speed gain (factor two!). Even
  // though answers are identical and the routine here is not the
  // speed-critical step. It's not at all clear why.
 
  int iphi = int( (pfCand->phi() + (twopi/2.))/_dphi );
  assert(iphi >= 0 && iphi <= _nphi);
  if (iphi == _nphi) iphi = 0; // just in case of rounding errors

  return iphi;
}

//----------------------------------------------------------------------
// retrieve the grid tile index for a given PseudoJet
int
HiFJGridEmptyAreaCalculator::tile_index_eta(const reco::PFCandidate *pfCand)  {
  // directly taking int does not work for values between -1 and 0
  // so use floor instead
  // double iy_double = (p.rap() - _ymin) / _dy;
  // if (iy_double < 0.0) return -1;
  // int iy = int(iy_double);
  // if (iy >= _ny) return -1;

  // writing it as below gives a huge speed gain (factor two!). Even
  // though answers are identical and the routine here is not the
  // speed-critical step. It's not at all clear why.
  int iy = int(floor( (pfCand->eta() - _ymin) / _dy ));
  if (iy < 0 || iy >= _ny) return -1;
  
  assert (iy < _ny && iy >= 0);

  return iy;
}

// #ifndef FASTJET_GMBGE_USEFJGRID
//----------------------------------------------------------------------
// protected material
//----------------------------------------------------------------------
// configure the grid
void
HiFJGridEmptyAreaCalculator::setup_grid_jet(const reco::Jet *jet) {

  // since we've exchanged the arguments of the grid constructor,
  // there's a danger of calls with exchanged ymax,spacing arguments -- 
  // the following check should catch most such situations.
  _yminjet = jet->eta()-0.6;
  _ymaxjet = jet->eta()+0.6;
  
  assert(_ymaxjet - _yminjet >= gridWidth_);

  // this grid-definition code is becoming repetitive -- it should
  // probably be moved somewhere central...
  double ny_double = (_ymaxjet-_yminjet) / gridWidth_;
  _nyjet = int(ny_double+0.5);
  _dyjet = (_ymaxjet-_yminjet) / _nyjet;
  
  assert(_nyjet >= 1);

  _ntotaljet = _nphi * _nyjet;
  //_scalar_pt.resize(_ntotal);
}


//----------------------------------------------------------------------
// retrieve the grid tile index for a given PseudoJet
int
HiFJGridEmptyAreaCalculator::tile_index_eta_jet(const reco::PFCandidate *pfCand) {
  // directly taking int does not work for values between -1 and 0
  // so use floor instead
  // double iy_double = (p.rap() - _ymin) / _dy;
  // if (iy_double < 0.0) return -1;
  // int iy = int(iy_double);
  // if (iy >= _ny) return -1;

  // writing it as below gives a huge speed gain (factor two!). Even
  // though answers are identical and the routine here is not the
  // speed-critical step. It's not at all clear why.
  int iyjet = int(floor( (pfCand->eta() - _yminjet) / _dy ));
  if (iyjet < 0 || iyjet >= _nyjet) return -1;
  
  assert (iyjet < _nyjet && iyjet >= 0);

  return iyjet;
}

int 
HiFJGridEmptyAreaCalculator::num_jet_grid_cells( std::vector<std::pair<int, int> >& indices )
{
  int ngrid = 0;
  //sort phi eta grid indices in phi
  std::sort(indices.begin(),indices.end());
  int lowest_jphi = indices[0].first;
  int lowest_jeta = indices[0].second;
  int highest_jeta = lowest_jeta;
    
  //for each fixed phi value calculate the number of grids in eta
  for(unsigned int iconst = 1; iconst < indices.size(); iconst++){
   int jphi = indices[iconst].first;
   int jeta = indices[iconst].second;
   if (jphi == lowest_jphi){
	if (jeta < lowest_jeta) lowest_jeta = jeta;
	if (jeta > highest_jeta) highest_jeta = jeta;
   }else{
	lowest_jphi = jphi;
	ngrid += highest_jeta - lowest_jeta + 1;
    lowest_jeta = jeta;
    highest_jeta = jeta;
   }
  }
  ngrid += highest_jeta - lowest_jeta + 1;
  return ngrid;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HiFJGridEmptyAreaCalculator::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiFJGridEmptyAreaCalculator::endJob() {
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiFJGridEmptyAreaCalculator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HiFJGridEmptyAreaCalculator);

