#include "RecoMuon/MuonIdentification/interface/MuonHOAcceptance.h"

#include <iostream>
#include <algorithm>
#include <cmath>

//#include "TTree.h"
#include "TMultiGraph.h"
#include "TGraph.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"

std::vector<uint32_t> MuonHOAcceptance::deadIds;
std::vector<MuonHOAcceptance::deadRegion> MuonHOAcceptance::deadRegions;
std::vector<uint32_t> MuonHOAcceptance::SiPMIds;
std::vector<MuonHOAcceptance::deadRegion> MuonHOAcceptance::SiPMRegions;
bool MuonHOAcceptance::inited = false;
int const MuonHOAcceptance::etaBounds = 5;
double const MuonHOAcceptance::etaMin[etaBounds] = 
  //{-1.262, -0.861, -0.307, 0.341, 0.885};
  {-1.2544, -0.8542, -0.3017, 0.3425, 0.8796};
double const MuonHOAcceptance::etaMax[etaBounds] = 
  //{-0.885, -0.341,  0.307, 0.861, 1.262};
  {-0.8796, -0.3425,  0.3017, 0.8542, 1.2544};
double const MuonHOAcceptance::twopi = 2.*3.14159265358979323846;
int const MuonHOAcceptance::phiSectors = 12;
double const MuonHOAcceptance::phiMinR0[phiSectors] = {-0.16172,
						        0.3618786,
						        0.8854773,
						        1.409076116,
						        1.932674892,
						        2.456273667,
						        2.979872443,
						        3.503471219,
						        4.027069994,
						        4.55066877,
						        5.074267545,
						        5.597866321 };
double const MuonHOAcceptance::phiMaxR0[phiSectors] = { 0.317395374,
							0.84099415,
							1.364592925,
							1.888191701,
							2.411790477,
							2.935389252,
							3.458988028,
							3.982586803,
							4.506185579,
							5.029784355,
							5.55338313,
							6.076981906 };
double const MuonHOAcceptance::phiMinR12[phiSectors] = {-0.166264081,
							 0.357334694,
							 0.88093347,
							 1.404532245,
							 1.928131021,
							 2.451729797,
							 2.975328572,
							 3.498927348,
							 4.022526123,
							 4.546124899,
							 5.069723674,
							 5.59332245 };
double const MuonHOAcceptance::phiMaxR12[phiSectors] = { 0.34398862,
							 0.867587396,
							 1.391186172,
							 1.914784947,
							 2.438383723,
							 2.961982498,
							 3.485581274,
							 4.00918005,
							 4.532778825,
							 5.056377601,
							 5.579976376,
							 6.103575152 };

bool MuonHOAcceptance::isChannelDead(uint32_t id) {
  if (!inited) return false;
  std::vector<uint32_t>::const_iterator found = 
    std::find(deadIds.begin(), deadIds.end(), id);
  if ((found != deadIds.end()) && (*found == id)) return true;
  else return false;
}

bool MuonHOAcceptance::isChannelSiPM(uint32_t id) {
  if (!inited) return false;
  std::vector<uint32_t>::const_iterator found =
    std::find(SiPMIds.begin(), SiPMIds.end(), id);
  if ((found != SiPMIds.end()) && (*found == id)) return true;
  else return false;
}

bool MuonHOAcceptance::inGeomAccept(double eta, double phi, 
				    double delta_eta, double delta_phi)
{
  for (int ieta = 0; ieta<etaBounds; ++ieta) {
    if ( (eta > etaMin[ieta]+delta_eta) &&
	 (eta < etaMax[ieta]-delta_eta) ) {
      for (int iphi = 0; iphi<phiSectors; ++iphi) {
	double const * mins =  ((ieta == 2) ? phiMinR0 : phiMinR12);
	double const * maxes = ((ieta == 2) ? phiMaxR0 : phiMaxR12);
	while (phi < mins[0]) 
	  phi += twopi;
	while (phi > mins[0]+twopi)
	  phi -= twopi;
	if ( ( phi > mins[iphi] + delta_phi ) &&
	     ( phi < maxes[iphi] - delta_phi ) ) {
	  return true;
	}
      }
      return false;
    }
  }
  return false;
}

bool MuonHOAcceptance::inNotDeadGeom(double eta, double phi, 
				     double delta_eta, double delta_phi) {
  if (!inited) return true;
  int ieta = int(eta/0.087) + ((eta>0) ? 1 : -1);
  double const * mins = ((std::abs(ieta) > 4) ? phiMinR12 : phiMinR0);
  while (phi < mins[0])
    phi += twopi;
  while (phi > mins[0]+twopi)
    phi -= twopi;
  std::vector<deadRegion>::const_iterator region;
  for (region = deadRegions.begin(); region != deadRegions.end(); ++region) {
    if ( (phi < region->phiMax + delta_phi) && 
	 (phi > region->phiMin - delta_phi) &&
	 (eta < region->etaMax + delta_eta) &&
	 (eta > region->etaMin - delta_eta) )
      return false;
  }
  return true;
}

bool MuonHOAcceptance::inSiPMGeom(double eta, double phi, 
				  double delta_eta, double delta_phi) {
  if (!inited) return false;
  int ieta = int(eta/0.087) + ((eta>0) ? 1 : -1);
  double const * mins = ((std::abs(ieta) > 4) ? phiMinR12 : phiMinR0);
  while (phi < mins[0])
    phi += twopi;
  while (phi > mins[0]+twopi)
    phi -= twopi;
  std::vector<deadRegion>::const_iterator region;
  for (region = SiPMRegions.begin(); region != SiPMRegions.end(); ++region) {
    if ( (phi < region->phiMax - delta_phi) && 
	 (phi > region->phiMin + delta_phi) &&
	 (eta < region->etaMax - delta_eta) && 
	 (eta > region->etaMin + delta_eta) ) {
      return true;
    }
  }
  return false;
}

void MuonHOAcceptance::initIds(edm::EventSetup const& eSetup) {
  deadIds.clear();

  edm::ESHandle<HcalChannelQuality> p;
  eSetup.get<HcalChannelQualityRcd>().get("withTopo",p);
  const HcalChannelQuality *myqual = p.product();

  edm::ESHandle<HcalSeverityLevelComputer> mycomputer;
  eSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
  const HcalSeverityLevelComputer *mySeverity = mycomputer.product();

  // TTree * deads = new TTree("deads", "deads");
  // deads->ReadFile("HOdeadnessChannels.txt", "ieta/I:iphi/I:deadness/D");
  int ieta, iphi;
  // double deadness;
  // deads->SetBranchAddress("ieta", &ieta);
  // deads->SetBranchAddress("iphi", &iphi);
  // deads->SetBranchAddress("deadness", &deadness);
  // deads->Print();
  //std::cout << "ieta\tiphi\n";
  for (ieta=-15; ieta <= 15; ieta++) {
    if (ieta != 0) {
      for (iphi = 1; iphi <= 72; iphi++) {
	// for (int i=0; i<deads->GetEntries(); ++i) {
	//   deads->GetEntry(i);
	//   if (deadness > 0.4) {
	HcalDetId did(HcalOuter,ieta,iphi,4);
	const HcalChannelStatus *mystatus = myqual->getValues(did.rawId());
	if (mySeverity->dropChannel(mystatus->getValue())) {
	  deadIds.push_back(did.rawId());
	  // std::cout << did.ieta() << '\t' << did.iphi() << '\n';
	}
	//HO +1 RBX 10
	if ( (ieta>=5) && (ieta<=10) && (iphi >= 47) && (iphi <= 58) ) {
	  SiPMIds.push_back(did.rawId());
	}
	//HO +2 RBX 12
	if ( (ieta>=11) && (ieta<=15) && (iphi >= 59) && (iphi <= 70) ) {
	  SiPMIds.push_back(did.rawId());
	}
      }
    }
  }
  std::sort(deadIds.begin(), deadIds.end());
  std::sort(SiPMIds.begin(), SiPMIds.end());
  // std::cout << "SiPMIds: " << SiPMIds.size() << '\n';
  // delete deads;
  buildDeadAreas();
  buildSiPMAreas();
  inited = true;
}

void MuonHOAcceptance::buildDeadAreas() {
  std::vector<uint32_t>::iterator did;
  std::list<deadIdRegion> didregions;
  for (did = deadIds.begin(); did != deadIds.end(); ++did) {
    HcalDetId tmpId(*did);
    didregions.push_back( deadIdRegion( tmpId.ieta(), tmpId.ieta(), 
					tmpId.iphi(), tmpId.iphi() ) );
  }
  // std::cout << "dead regions: " << didregions.size() << '\n';

  mergeRegionLists(didregions);
  convertRegions(didregions, deadRegions);
}

void MuonHOAcceptance::buildSiPMAreas() {
  std::vector<uint32_t>::iterator sid;
  std::list<deadIdRegion> idregions;

  for (sid = SiPMIds.begin(); sid != SiPMIds.end(); ++sid) {
    HcalDetId tmpId(*sid);
    idregions.push_back( deadIdRegion( tmpId.ieta(), tmpId.ieta(),
				       tmpId.iphi(), tmpId.iphi() ) );
  }

  mergeRegionLists(idregions);
  convertRegions(idregions,SiPMRegions);
}

void MuonHOAcceptance::mergeRegionLists (std::list<deadIdRegion>& didregions) {
  std::list<deadIdRegion>::iterator curr;
  std::list<deadIdRegion> list2;
  unsigned int startSize;
  do {
    startSize = didregions.size();

    // std::cout << "regions: " << startSize << '\n';
    //merge in phi
    curr = didregions.begin();
    while (curr != didregions.end()) {
      deadIdRegion merger(*curr);
      curr = didregions.erase(curr);
      while (curr != didregions.end()) {
	if ( (merger.sameEta(*curr)) && (merger.adjacentPhi(*curr)) ) {
	  merger.merge(*curr);
	  curr = didregions.erase(curr);
	} else ++curr;
      }
      list2.push_back(merger);
      curr = didregions.begin();
    }

    //merge in eta
    curr = list2.begin();
    while (curr != list2.end()) {
      deadIdRegion merger(*curr);
      curr = list2.erase(curr);
      while (curr != list2.end()) {
	if ( (merger.samePhi(*curr)) && (merger.adjacentEta(*curr)) ) {
	  merger.merge(*curr);
	  curr = list2.erase(curr);
	} else ++curr;
      }
      didregions.push_back(merger);
      curr = list2.begin();
    }
  } while (startSize > didregions.size());
}

void MuonHOAcceptance::convertRegions(std::list<deadIdRegion> const& idregions,
				      std::vector<deadRegion>& regions) {
  double e1, e2;
  double eMin,eMax,pMin,pMax;
  static double const etaStep = 0.087;
  static double const phiStep = twopi/72.;
  double const offset = 2.;
  double const * mins;
  double const * maxes;
  std::list<deadIdRegion>::const_iterator curr;
  double zero;
  for (curr = idregions.begin(); curr != idregions.end(); ++curr) {
    // std::cout << "region boundaries: ieta,iphi\n"
    // 	      << "              min: " << curr->etaMin << ',' << curr->phiMin << '\n'
    // 	      << "              max: " << curr->etaMax << ',' << curr->phiMax << '\n';
    if (curr->etaMin == -4) {
      eMin = etaMax[1];
    } else if (curr->etaMin == 5) {
      eMin = etaMin[3];
    } else {
      e1 = (std::abs(curr->etaMin)-1)*etaStep*
	(-(curr->etaMin<0) + (curr->etaMin>0));
      e2 = std::abs(curr->etaMin)*etaStep*
	(-(curr->etaMin<0) + (curr->etaMin>0));
      eMin = std::min(e1,e2);
    }
    if (curr->etaMax == 4) {
      eMax = etaMin[3];
    } else if (curr->etaMax == -5) {
      eMax = etaMax[1];
    } else {
      e1 = (std::abs(curr->etaMax)-1)*etaStep*
	(-(curr->etaMax<0) + (curr->etaMax>0));
      e2 = std::abs(curr->etaMax)*etaStep*
	(-(curr->etaMax<0) + (curr->etaMax>0));
      eMax = std::max(e1,e2);
    }
    mins = ((std::abs(curr->etaMin)>4) ? phiMinR12 : phiMinR0);
    maxes = ((std::abs(curr->etaMin)>4) ? phiMaxR12 : phiMaxR0);
    zero = (mins[0] + maxes[phiSectors-1] - twopi)/2.;
    pMin = (curr->phiMin-1)*phiStep + zero + phiStep*offset;
    pMax = curr->phiMax*phiStep + zero + phiStep*offset;
    while (pMax < mins[0])
      pMax += twopi;
    while (pMax > mins[0]+twopi) 
      pMax -= twopi;
    while (pMin < mins[0])
      pMin += twopi;
    while (pMin > mins[0]+twopi) 
      pMin -= twopi;

    regions.push_back( deadRegion(eMin, eMax, pMin, pMax) );
    // std::cout << "                 : eta,phi\n"
    // 	      << "              min: " << tmp.etaMin << ',' << tmp.phiMin << '\n'
    // 	      << "              max: " << tmp.etaMax << ',' << tmp.phiMax << '\n';
  }
}

TMultiGraph * MuonHOAcceptance::graphRegions(std::vector<deadRegion> const& regions) {
  TMultiGraph * bounds = new TMultiGraph("bounds", "bounds");
  std::vector<deadRegion>::const_iterator region;
  TGraph * gr;
  for (region = regions.begin(); region != regions.end(); ++region) {
    // std::cout << "region eta range:(" << region->etaMin << ',' << region->etaMax
    // 	      << ") phi range: (" << region->phiMin << ',' << region->phiMax << ")\n";
    double pMin = region->phiMin;
    while (pMin > twopi/2.) pMin -= twopi;
    double pMax = region->phiMax;
    while (pMax > twopi/2.) pMax -= twopi;

    if (pMin < pMax) {
      gr = new TGraph(5);
      gr->SetLineColor(kRed);
      gr->SetPoint(0, region->etaMin, pMin);
      gr->SetPoint(1, region->etaMin, pMax);
      gr->SetPoint(2, region->etaMax, pMax);
      gr->SetPoint(3, region->etaMax, pMin);
      gr->SetPoint(4, region->etaMin, pMin);
      bounds->Add(gr, "l");
    } else {
      gr = new TGraph(5);
      gr->SetLineColor(kRed);
      gr->SetPoint(0, region->etaMin, pMin);
      gr->SetPoint(1, region->etaMin, twopi/2.);
      gr->SetPoint(2, region->etaMax, twopi/2.);
      gr->SetPoint(3, region->etaMax, pMin);
      gr->SetPoint(4, region->etaMin, pMin);
      bounds->Add(gr, "l");
      gr = new TGraph(5);
      gr->SetLineColor(kRed);
      gr->SetPoint(0, region->etaMin, -twopi/2.);
      gr->SetPoint(1, region->etaMin, pMax);
      gr->SetPoint(2, region->etaMax, pMax);
      gr->SetPoint(3, region->etaMax, -twopi/2.);
      gr->SetPoint(4, region->etaMin, -twopi/2.);
      bounds->Add(gr, "l");
    }
  }
  return bounds;
}

void MuonHOAcceptance::deadIdRegion::merge (deadIdRegion const& other) {
  etaMin = std::min(etaMin, other.etaMin);
  etaMax = std::max(etaMax, other.etaMax);
  phiMin = std::min(phiMin, other.phiMin);
  phiMax = std::max(phiMax, other.phiMax);
}
