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
  gridWidth_(iConfig.getParameter<double>("gridWidth")),
  band_(iConfig.getParameter<double>("bandWidth")),
  hiBinCut_(iConfig.getParameter<int>("hiBinCut")),
  doCentrality_(iConfig.getParameter<bool>("doCentrality")),
  keepGridInfo_(iConfig.getParameter<bool>("keepGridInfo"))
{
  ymin_ = -99;
  ymax_ = -99;
  dy_ = -99;
  dphi_ = -99;
  tileArea_ = - 99;
  
  dyJet_ = 99;
  yminJet_ = - 99;
  ymaxJet_ = -99;
  totalInboundArea_ = -99;
  etaminJet_= -99;
  etamaxJet_ = -99;
  
  ny_ = 0;
  nphi_ = 0;
  ntotal_ = 0;
 
  ntotalJet_ = 0;
  nyJet_ = 0;
    
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
  if(keepGridInfo_){
    produces<std::vector<double > >("mapRhoVsEtaGrid");
    produces<std::vector<double > >("mapMeanRhoVsEtaGrid");
    produces<std::vector<double > >("mapEtaMaxGrid");
    produces<std::vector<double > >("mapEtaMinGrid");
  }
}


HiFJGridEmptyAreaCalculator::~HiFJGridEmptyAreaCalculator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

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
    edm::Handle<int> cbin;
    iEvent.getByToken(centralityBinToken_,cbin);
    hiBin = *cbin;
   
    if(hiBin < hiBinCut_) doEmptyArea = false;
  }
  
  //Define output vectors
  int neta = (int)mapEtaRanges->size();
   
  std::unique_ptr<std::vector<double>> mapToRhoCorrOut ( new std::vector<double>(neta-1,1e-6));
  std::unique_ptr<std::vector<double>> mapToRhoMCorrOut ( new std::vector<double>(neta-1,1e-6));
  std::unique_ptr<std::vector<double>> mapToRhoCorr1BinOut ( new std::vector<double>(neta-1,1e-6));
  std::unique_ptr<std::vector<double>> mapToRhoMCorr1BinOut ( new std::vector<double>(neta-1,1e-6));

  setupGrid(mapEtaRanges->at(0), mapEtaRanges->at(neta-1));
  
  //calculate empty area correction over full acceptance leaving eta bands on the sides
  double allAcceptanceCorr = 1;
  if(doEmptyArea){
    etaminJet_ = mapEtaRanges->at(0) - band_;
    etamaxJet_ = mapEtaRanges->at(neta-1) + band_;
  
    calculateAreaFractionOfJets(iEvent, iSetup);
  
    allAcceptanceCorr =  totalInboundArea_;
  }
  
  //calculate empty area correction in each eta range
  for(int ieta = 0; ieta<(neta-1); ieta++) {
   
    double correctionKt = 1;   
    double rho = mapRho->at(ieta);
    double rhoM = mapRhoM->at(ieta);
    
    if(doEmptyArea){
      double etamin = mapEtaRanges->at(ieta);
      double etamax = mapEtaRanges->at(ieta+1);
      
      etaminJet_ = etamin + band_;  
      etamaxJet_ = etamax - band_;  
   
      calculateAreaFractionOfJets(iEvent, iSetup);
      correctionKt = totalInboundArea_;
    }
   
    mapToRhoCorrOut->at(ieta) = correctionKt*rho;
    mapToRhoMCorrOut->at(ieta) = correctionKt*rhoM;
   
    mapToRhoCorr1BinOut->at(ieta) = allAcceptanceCorr*rho;
    mapToRhoMCorr1BinOut->at(ieta) = allAcceptanceCorr*rhoM;
  }

  iEvent.put(std::move(mapToRhoCorrOut),"mapToRhoCorr");
  iEvent.put(std::move(mapToRhoMCorrOut),"mapToRhoMCorr");
  iEvent.put(std::move(mapToRhoCorr1BinOut),"mapToRhoCorr1Bin");
  iEvent.put(std::move(mapToRhoMCorr1BinOut),"mapToRhoMCorr1Bin");
  
  //calculate rho from grid as a function of eta over full range using PF candidates
  
  std::unique_ptr<std::vector<double>> mapRhoVsEtaGridOut ( new std::vector<double>(ny_,0.));
  std::unique_ptr<std::vector<double>> mapMeanRhoVsEtaGridOut ( new std::vector<double>(ny_,0.));
  std::unique_ptr<std::vector<double>> mapEtaMaxGridOut ( new std::vector<double>(ny_,0.));
  std::unique_ptr<std::vector<double>> mapEtaMinGridOut ( new std::vector<double>(ny_,0.));
  calculateGridRho(iEvent, iSetup);
  if(keepGridInfo_){
    for(int ieta = 0; ieta < ny_; ieta++) {
      mapRhoVsEtaGridOut->at(ieta) = rhoVsEta_[ieta];
      mapMeanRhoVsEtaGridOut->at(ieta) = meanRhoVsEta_[ieta];
      mapEtaMaxGridOut->at(ieta) = etaMaxGrid_[ieta];
      mapEtaMinGridOut->at(ieta) = etaMinGrid_[ieta];
    }

    iEvent.put(std::move(mapRhoVsEtaGridOut),"mapRhoVsEtaGrid");
    iEvent.put(std::move(mapMeanRhoVsEtaGridOut),"mapMeanRhoVsEtaGrid");
    iEvent.put(std::move(mapEtaMaxGridOut),"mapEtaMaxGrid");
    iEvent.put(std::move(mapEtaMinGridOut),"mapEtaMinGrid");
  }
}

//----------------------------------------------------------------------
// setting a new event
//----------------------------------------------------------------------
// tell the background estimator that it has a new event, composed
// of the specified particles.
void
HiFJGridEmptyAreaCalculator::calculateGridRho(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  vector<vector<double>> scalarPt(ny_, vector<double>(nphi_, 0.0));
  
  edm::Handle<reco::PFCandidateCollection> pfCands;
  iEvent.getByToken(pfCandsToken_, pfCands);
  const reco::PFCandidateCollection *pfCandidateColl = pfCands.product();
  for(unsigned icand = 0; icand < pfCandidateColl->size(); icand++) {
   const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
   //use ony the particles within the eta range
   if (pfCandidate.eta() < ymin_ || pfCandidate.eta() > ymax_ ) continue;
   int jeta = tileIndexEta(&pfCandidate);
   int jphi = tileIndexPhi(&pfCandidate);
   scalarPt[jeta][jphi] += pfCandidate.pt();
  }
  
  rhoVsEta_.resize(ny_);
  meanRhoVsEta_.resize(ny_);
  for(int jeta = 0; jeta < ny_; jeta++){
  
    rhoVsEta_[jeta] = 0;
    meanRhoVsEta_[jeta] = 0;
    vector<double> rhoVsPhi;
    int nEmpty = 0;
    
    for(int jphi = 0; jphi < nphi_; jphi++){
      double binpt = scalarPt[jeta][jphi];
      meanRhoVsEta_[jeta] += binpt;
      //fill in the vector for median calculation
      if(binpt > 0) rhoVsPhi.push_back(binpt);
      else nEmpty++;
    }
    meanRhoVsEta_[jeta] /= ((double)nphi_);
    meanRhoVsEta_[jeta] /= tileArea_;

    //median calculation
    sort(rhoVsPhi.begin(), rhoVsPhi.end());
    //use only the nonzero grid cells for median calculation;
    int nFull = nphi_ - nEmpty;
    if(nFull == 0){
      rhoVsEta_[jeta] = 0;
      continue;
    }
    if (nFull  % 2 == 0)
      {
        rhoVsEta_[jeta] = (rhoVsPhi[(int)(nFull / 2 - 1)] + rhoVsPhi[(int)(nFull / 2)]) / 2;
      }
    else 
      {
        rhoVsEta_[jeta] = rhoVsPhi[(int)(nFull / 2)];
      }
    //correct for empty cells 
    rhoVsEta_[jeta] *= (((double) nFull)/((double) nphi_));
    //normalize to area
    rhoVsEta_[jeta] /= tileArea_;
  }
}

void
HiFJGridEmptyAreaCalculator::calculateAreaFractionOfJets(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(jetsToken_, jets);
  
  //calculate jet kt area fraction inside boundary by grid
  totalInboundArea_ = 0;
  
  for(auto jet = jets->begin(); jet != jets->end(); ++jet) {
    if (jet->eta() < etaminJet_ || jet->eta() > etamaxJet_) continue;
   
    double areaKt = jet->jetArea(); 
    setupGridJet(&*jet);
    std::vector<std::pair<int, int> > pfIndicesJet;
    std::vector<std::pair<int, int> > pfIndicesJetInbound;
    int nConstitJet = 0;
    int nConstitJetInbound = 0;
    for(auto daughter : jet->getJetConstituentsQuick()){
      auto pfCandidate = static_cast<const reco::PFCandidate*>(daughter);

	  int jeta = tileIndexEtaJet(&*pfCandidate);
	  int jphi = tileIndexPhi(&*pfCandidate);
	  pfIndicesJet.push_back(std::make_pair(jphi, jeta));
	  nConstitJet++;
	  if (pfCandidate->eta() < etaminJet_ && pfCandidate->eta() > etamaxJet_) continue;
	  pfIndicesJetInbound.push_back(std::make_pair(jphi, jeta));
	  nConstitJetInbound++;
   }   
   
   //if the jet is well within the eta range just add the area
   if(nConstitJet == nConstitJetInbound){
	totalInboundArea_ += areaKt;
    continue;
   }
   
   //for jets that fall outside of eta range calculate fraction of area
   //inside the range with a grid
   int nthis = 0;
   if (nConstitJet > 0) nthis = numJetGridCells(pfIndicesJet);

   int nthisInbound = 0;
   if (nConstitJetInbound > 0) nthisInbound = numJetGridCells(pfIndicesJetInbound);
   
  
   double fractionArea = ((double)nthisInbound)/((double)nthis);
   totalInboundArea_ += areaKt*fractionArea;    
  } 
  
  //divide by the total area in that range
  totalInboundArea_ /= ((etamaxJet_ - etaminJet_)*twopi_);
  
  //the fraction can still be greater than 1 because kt area fraction inside 
  //the range can differ from what we calculated with the grid
  if (totalInboundArea_ > 1) totalInboundArea_ = 1;
}


// #ifndef FASTJET_GMBGE_USEFJGRID
//----------------------------------------------------------------------
// protected material
//----------------------------------------------------------------------
// configure the grid
void
HiFJGridEmptyAreaCalculator::setupGrid(double etamin, double etamax) {

  // since we've exchanged the arguments of the grid constructor,
  // there's a danger of calls with exchanged ymax,spacing arguments -- 
  // the following check should catch most such situations.
  ymin_ = etamin;
  ymax_ = etamax;
  
  assert(ymax_ - ymin_ >= gridWidth_);

  // this grid-definition code is becoming repetitive -- it should
  // probably be moved somewhere central...
  double nyDouble = (ymax_ - ymin_) / gridWidth_;
  ny_ = int(nyDouble+0.5);
  dy_ = (ymax_ - ymin_) / ny_;
  
  nphi_ = int (twopi_ / gridWidth_ + 0.5);
  dphi_ = twopi_ / nphi_;

  // some sanity checking (could throw a fastjet::Error)
  assert(ny_ >= 1 && nphi_ >= 1);

  ntotal_ = nphi_ * ny_;
  //_scalar_pt.resize(_ntotal);
  tileArea_ = dy_ * dphi_;
  
  
  etaMaxGrid_.resize(ny_);
  etaMinGrid_.resize(ny_);
  for(int jeta = 0; jeta < ny_; jeta++){
   etaMinGrid_[jeta] = etamin + dy_*((double)jeta);
   etaMaxGrid_[jeta] = etamin + dy_*((double)jeta + 1.);
  }
}

//----------------------------------------------------------------------
// retrieve the grid tile index for a given PseudoJet
int
HiFJGridEmptyAreaCalculator::tileIndexPhi(const reco::PFCandidate *pfCand)  {
  // directly taking int does not work for values between -1 and 0
  // so use floor instead
  // double iy_double = (p.rap() - _ymin) / _dy;
  // if (iy_double < 0.0) return -1;
  // int iy = int(iy_double);
  // if (iy >= _ny) return -1;

  // writing it as below gives a huge speed gain (factor two!). Even
  // though answers are identical and the routine here is not the
  // speed-critical step. It's not at all clear why.
 
  int iphi = int( (pfCand->phi() + (twopi_/2.))/dphi_ );
  assert(iphi >= 0 && iphi <= nphi_);
  if (iphi == nphi_) iphi = 0; // just in case of rounding errors

  return iphi;
}

//----------------------------------------------------------------------
// retrieve the grid tile index for a given PseudoJet
int
HiFJGridEmptyAreaCalculator::tileIndexEta(const reco::PFCandidate *pfCand)  {
  // directly taking int does not work for values between -1 and 0
  // so use floor instead
  // double iy_double = (p.rap() - _ymin) / _dy;
  // if (iy_double < 0.0) return -1;
  // int iy = int(iy_double);
  // if (iy >= _ny) return -1;

  // writing it as below gives a huge speed gain (factor two!). Even
  // though answers are identical and the routine here is not the
  // speed-critical step. It's not at all clear why.
  int iy = int(floor( (pfCand->eta() - ymin_) / dy_ ));
  if (iy < 0 || iy >= ny_) return -1;
  
  assert (iy < ny_ && iy >= 0);

  return iy;
}

// #ifndef FASTJET_GMBGE_USEFJGRID
//----------------------------------------------------------------------
// protected material
//----------------------------------------------------------------------
// configure the grid
void
HiFJGridEmptyAreaCalculator::setupGridJet(const reco::Jet *jet) {

  // since we've exchanged the arguments of the grid constructor,
  // there's a danger of calls with exchanged ymax,spacing arguments -- 
  // the following check should catch most such situations.
  yminJet_ = jet->eta()-0.6;
  ymaxJet_ = jet->eta()+0.6;
  
  assert(ymaxJet_ - yminJet_ >= gridWidth_);

  // this grid-definition code is becoming repetitive -- it should
  // probably be moved somewhere central...
  double nyDouble = (ymaxJet_ - yminJet_) / gridWidth_;
  nyJet_ = int(nyDouble+0.5);
  dyJet_ = (ymaxJet_ - yminJet_) / nyJet_;
  
  assert(nyJet_ >= 1);

  ntotalJet_ = nphi_ * nyJet_;
  //_scalar_pt.resize(_ntotal);
}


//----------------------------------------------------------------------
// retrieve the grid tile index for a given PseudoJet
int
HiFJGridEmptyAreaCalculator::tileIndexEtaJet(const reco::PFCandidate *pfCand) {
  // directly taking int does not work for values between -1 and 0
  // so use floor instead
  // double iy_double = (p.rap() - _ymin) / _dy;
  // if (iy_double < 0.0) return -1;
  // int iy = int(iy_double);
  // if (iy >= _ny) return -1;

  // writing it as below gives a huge speed gain (factor two!). Even
  // though answers are identical and the routine here is not the
  // speed-critical step. It's not at all clear why.
  int iyjet = int(floor( (pfCand->eta() - yminJet_) / dy_ ));
  if (iyjet < 0 || iyjet >= nyJet_) return -1;
  
  assert (iyjet < nyJet_ && iyjet >= 0);

  return iyjet;
}

int 
HiFJGridEmptyAreaCalculator::numJetGridCells( std::vector<std::pair<int, int> >& indices )
{
  int ngrid = 0;
  //sort phi eta grid indices in phi
  std::sort(indices.begin(),indices.end());
  int lowestJPhi = indices[0].first;
  int lowestJEta = indices[0].second;
  int highestJEta = lowestJEta;
    
  //for each fixed phi value calculate the number of grids in eta
  for(unsigned int iconst = 1; iconst < indices.size(); iconst++){
     int jphi = indices[iconst].first;
     int jeta = indices[iconst].second;
     if (jphi == lowestJPhi){
	  if (jeta < lowestJEta) lowestJEta = jeta;
	  if (jeta > highestJEta) highestJEta = jeta;
     }else{
	  lowestJPhi = jphi;
	  ngrid += highestJEta - lowestJEta + 1;
      lowestJEta = jeta;
      highestJEta = jeta;
    }
  }
  ngrid += highestJEta - lowestJEta + 1;
  return ngrid;
}

void
HiFJGridEmptyAreaCalculator::beginStream(edm::StreamID)
{
}

void
HiFJGridEmptyAreaCalculator::endStream() {
}

void HiFJGridEmptyAreaCalculator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetSource",edm::InputTag("kt4PFJets"));
  desc.add<edm::InputTag>("CentralityBinSrc",edm::InputTag("centralityBin"));
  desc.add<edm::InputTag>("mapEtaEdges",edm::InputTag("mapEtaEdges"));
  desc.add<edm::InputTag>("mapToRho",edm::InputTag("mapToRho"));
  desc.add<edm::InputTag>("mapToRhoM",edm::InputTag("mapToRhoM"));
  desc.add<edm::InputTag>("pfCandSource",edm::InputTag("particleFlow"));
  desc.add<double>("gridWidth",0.05);
  desc.add<double>("bandWidth",0.2);
  desc.add<bool>("doCentrality", true);
  desc.add<int>("hiBinCut",100);
  desc.add<bool>("keepGridInfo",false);
  descriptions.add("hiFJGridEmptyAreaCalculator",desc);
}

DEFINE_FWK_MODULE(HiFJGridEmptyAreaCalculator);

