///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "FastSimulation/ShowerDevelopment/interface/FastHFShowerLibrary.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Randomize.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

// STL headers 
#include <vector>
#include <iostream>

//#define DebugLog

FastHFShowerLibrary::FastHFShowerLibrary(edm::ParameterSet const & p) : fibre(0),
                                                                      hf(0),
							  	emBranch(0),
								hadBranch(0),
                                                          numberingScheme(0), 
                                                         numberingFromDDD(0), 
								      npe(0),
                                                                    photo(0) {
  
  edm::ParameterSet m_HS   = p.getParameter<edm::ParameterSet>("HFShowerLibrary");
  edm::FileInPath fp       = m_HS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName    = fp.fullPath();
  std::string emName       = m_HS.getParameter<std::string>("TreeEMID");
  std::string hadName      = m_HS.getParameter<std::string>("TreeHadID");
  std::string branchEvInfo = m_HS.getUntrackedParameter<std::string>("BranchEvt","HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo");
  std::string branchPre    = m_HS.getUntrackedParameter<std::string>("BranchPre","HFShowerPhotons_hfshowerlib_");
  std::string branchPost   = m_HS.getUntrackedParameter<std::string>("BranchPost","_R.obj");

  probMax                  = m_HS.getParameter<double>("ProbMax");
  backProb                 = m_HS.getParameter<double>("BackProbability");  
  verbose                  = m_HS.getUntrackedParameter<bool>("Verbosity",false);
  applyFidCut              = m_HS.getParameter<bool>("ApplyFiducialCut");
  cFibre                   = c_light*(m_HS.getParameter<double>("CFibre"));

  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree = pTreeName.c_str();
  hf                = TFile::Open(nTree);

  if (!hf->IsOpen()) { 
    edm::LogError("FastCalorimetry") << "HFShowerLibrary: opening " << nTree 
			             << " failed";
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: opening " << nTree 
			            << " successfully"; 
  }

  newForm = (branchEvInfo == "");
  TTree* event(0);
  if (newForm) event = (TTree *) hf ->Get("HFSimHits");
  else         event = (TTree *) hf ->Get("Events");
  if (event) {
    TBranch *evtInfo(0);
    if (!newForm) {
      std::string info = branchEvInfo + branchPost;
      evtInfo          = event->GetBranch(info.c_str());
    }
    if (evtInfo || newForm) {
      loadEventInfo(evtInfo);
    } else {
      edm::LogError("FastCalorimetry") << "HFShowerLibrary: HFShowerLibrayEventInfo"
				       << " Branch does not exist in Event";
      throw cms::Exception("Unknown", "HFShowerLibrary")
	<< "Event information absent\n";
    }
  } else {
    edm::LogError("FastCalorimetry") << "HFShowerLibrary: Events Tree does not "
			             << "exist";
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Events tree absent\n";
  }
  
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Library " << libVers 
			          << " ListVersion "	<< listVersion 
			          << " Events Total " << totEvents << " and "
			          << evtPerBin << " per bin";

  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Energies (GeV) with " 
			          << nMomBin	<< " bins";

  for (int i=0; i<nMomBin; i++) 
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: pmom[" << i << "] = "
			            << pmom[i]/GeV << " GeV";

  std::string nameBr = branchPre + emName + branchPost;
  emBranch         = event->GetBranch(nameBr.c_str());
  if (verbose) emBranch->Print();
  nameBr           = branchPre + hadName + branchPost;
  hadBranch        = event->GetBranch(nameBr.c_str());
  if (verbose) hadBranch->Print();
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary:Branch " << emName 
			          << " has " << emBranch->GetEntries() 
			          << " entries and Branch " << hadName 
			          << " has " << hadBranch->GetEntries() 
			          << " entries";

  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::No packing information -"
			          << " Assume x, y, z are not in packed form";

  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Maximum probability cut off " 
			          << probMax << "  Back propagation of light prob. "
                                  << backProb ;
}

FastHFShowerLibrary::~FastHFShowerLibrary() {
  if (hf)               hf->Close();
  if (fibre)            delete fibre;
  if (photo)            delete photo;
  if (numberingFromDDD) delete numberingFromDDD;
  if (numberingScheme)  delete numberingScheme; 
}

void const FastHFShowerLibrary::initHFShowerLibrary(const edm::EventSetup& iSetup) {

  edm::LogInfo("FastCalorimetry") << "initHFShowerLibrary::initialization"; 

  edm::ESTransientHandle<DDCompactView> cpv;
  iSetup.get<IdealGeometryRecord>().get(cpv);

  std::string name = "HcalHits";
  std::string attribute = "ReadOutName";
  std::string value     = name;
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(*cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();
  if (dodet) {
    DDsvalues_type sv(fv.mergedSpecifics());
    // Radius (minimum and maximum)
    int nR     = -1;
    std::vector<double> rTable = getDDDArray("rTable",sv,nR);
    rMin = rTable[0];
    rMax = rTable[nR-1];
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: rMIN " << rMin/cm 
			            << " cm and rMax " << rMax/cm;
    // Delta phi
    int nEta   = -1;
    std::vector<double> etaTable = getDDDArray("etaTable",sv,nEta);
    int nPhi   = nEta + nR - 2;
    std::vector<double> phibin   = getDDDArray("phibin",sv,nPhi);
    dphi       = phibin[nEta-1];
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: (Half) Phi Width of wedge " 
			            << dphi/deg;

    // Special Geometry parameters
    int ngpar = 7;
    gpar      = getDDDArray("gparHF",sv,ngpar);
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: " << ngpar << " gpar (cm)";

    for (int ig=0; ig<ngpar; ig++) 
      edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: gpar[" << ig << "] = "
			              << gpar[ig]/cm << " cm";
  } else {
    edm::LogError("FastCalorimetry") << "HFShowerLibrary: cannot get filtered "
			             << " view for " << attribute << " matching "
			             << name;
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "cannot match " << attribute << " to " << name <<"\n";
  }

  initRun();  
  fibre = new FastHFFibre(name, *cpv, cFibre);
  photo = new HFShowerPhotonCollection;
  numberingFromDDD = new HcalNumberingFromDDD(name, *cpv);
  numberingScheme  = new HcalNumberingScheme();
}

void FastHFShowerLibrary::initRun() {

  geantinoPDG = 0; gammaPDG = 22;
  emPDG   = 11; epPDG    = -11; nuePDG   = 12; anuePDG   = -12;
  numuPDG = 14; anumuPDG = -14; nutauPDG = 16; anutauPDG = -16;
  pi0PDG = 111; etaPDG   = 221;

#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Particle codes for e- = " 
			   << emPDG << ", e+ = " << epPDG << ", gamma = " 
			   << gammaPDG << ", pi0 = " << pi0PDG << ", eta = " 
			   << etaPDG << ", geantino = " << geantinoPDG 
			   << "\n        nu_e = " << nuePDG << ", nu_mu = " 
			   << numuPDG << ", nu_tau = " << nutauPDG 
			   << ", anti_nu_e = " << anuePDG << ", anti_nu_mu = " 
			   << anumuPDG << ", anti_nu_tau = " << anutauPDG;
#endif
}

void FastHFShowerLibrary::recoHFShowerLibrary(const FSimTrack& myTrack) {

#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: recoHFShowerLibrary ";
#endif 

  if(!myTrack.onVFcal()) {
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: we should not be here ";
#endif
  }

  hitMap.clear();
  double eGen  = 1000.*myTrack.vfcalEntrance().e();                // energy in [MeV]
  double delZv = (myTrack.vfcalEntrance().vertex().Z()>0.0) ? 50.0 : -50.0;
  G4ThreeVector vertex( 10.*myTrack.vfcalEntrance().vertex().X(),
                        10.*myTrack.vfcalEntrance().vertex().Y(),
                        10.*myTrack.vfcalEntrance().vertex().Z()+delZv); // in [mm]

  G4ThreeVector direction(myTrack.vfcalEntrance().Vect().X(),
                          myTrack.vfcalEntrance().Vect().Y(),
                          myTrack.vfcalEntrance().Vect().Z());

  bool ok;
  double weight = 1.0;                     // rad. damage 
  int parCode   = myTrack.type();
  std::vector<FastHFShowerLibrary::Hit> hits =
              getHits(vertex, direction, parCode, eGen, ok, weight, false);

  for (unsigned int i=0; i<hits.size(); ++i) {
    G4ThreeVector pos = hits[i].position;
    int depth         = hits[i].depth;
    double time       = hits[i].time;

    if (isItinFidVolume (pos)) {     
      int det = 5;
      int lay = 1;
      uint32_t id = 0;
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, pos, depth, lay);
      id = numberingScheme->getUnitID(tmp);

      CaloHitID current_id(id,time,myTrack.id());
      std::map<CaloHitID,float>::iterator cellitr;
      cellitr = hitMap.find(current_id);
      if(cellitr==hitMap.end()) {
         hitMap.insert(std::pair<CaloHitID,float>(current_id,1.0));
      } else {
         cellitr->second += 1.0;
      }
    }  // end of isItinFidVolume check 

  } // end loop over hits

}

bool FastHFShowerLibrary::isItinFidVolume (G4ThreeVector& hitPoint) {
  bool flag = true;
  if (applyFidCut) {
    int npmt = HFFibreFiducial:: PMTNumber(hitPoint);
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::isItinFidVolume:#PMT= " 
                                    << npmt << " for hit point " << hitPoint;
#endif
    if (npmt <= 0) flag = false;
  }
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::isItinFidVolume: point " 
                                    << hitPoint << " return flag " << flag;
#endif
  return flag;
}

std::vector<FastHFShowerLibrary::Hit> FastHFShowerLibrary::getHits(const G4ThreeVector & hitPoint,
                                  const G4ThreeVector & momDir, int parCode, double pin, 
                                  bool & ok, double weight, bool onlyLong) {

  std::vector<FastHFShowerLibrary::Hit> hit;
  ok = false;
  if (parCode == pi0PDG || parCode == etaPDG || parCode == nuePDG ||
      parCode == numuPDG || parCode == nutauPDG || parCode == anuePDG ||
      parCode == anumuPDG || parCode == anutauPDG || parCode == geantinoPDG) 
    return hit;
  ok = true;

  double tSlice = 0.1*hitPoint.mag()/29.98;
  double pz     = momDir.z(); 
  double zint   = hitPoint.z(); 

// if particle moves from interaction point or "backwards (halo)
  int backward = 0;
  if (pz * zint < 0.) backward = 1;
  
  double sphi   = sin(momDir.phi());
  double cphi   = cos(momDir.phi());
  double ctheta = cos(momDir.theta());
  double stheta = sin(momDir.theta());

#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: getHits " << parCode
			          << " of energy " << pin/GeV << " GeV"
			          << "  dir.orts " << momDir.x() << ", " <<momDir.y() 
			          << ", " << momDir.z() << "  Pos x,y,z = " 
			          << hitPoint.x() << "," << hitPoint.y() << "," 
			          << hitPoint.z() << "," 
			          << " sphi,cphi,stheta,ctheta  = " << sphi 
			          << ","  << cphi << ", " << stheta << "," << ctheta; 
#endif    

  if (parCode == emPDG || parCode == epPDG || parCode == gammaPDG ) {
    if (pin<pmom[nMomBin-1]) {
      interpolate(0, pin);
    } else {
      extrapolate(0, pin);
    }
  } else {
    if (pin<pmom[nMomBin-1]) {
      interpolate(1, pin);
    } else {
      extrapolate(1, pin);
    }
  }
    
  int nHit = 0;
  FastHFShowerLibrary::Hit oneHit;
  for (int i = 0; i < npe; i++) {
    double zv = std::abs(pe[i].z()); // abs local z  

#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Hit " << i << " " << pe[i] << " zv " << zv;
#endif

    if (zv <= gpar[1] && pe[i].lambda() > 0 && 
	(pe[i].z() >= 0 || (zv > gpar[0] && (!onlyLong)))) {
      int depth = 1;
      if (onlyLong) {
      } else if (backward == 0) {    // fully valid only for "front" particles
	if (pe[i].z() < 0) depth = 2;// with "front"-simulated shower lib.
      } else {                       // for "backward" particles - almost equal
	double r = G4UniformRand();  // share between L and S fibers
        if (r > 0.5) depth = 2;
      } 
      

      // Updated coordinate transformation from local
      //  back to global using two Euler angles: phi and theta
      double pex = pe[i].x();
      double pey = pe[i].y();

      double xx = pex*ctheta*cphi - pey*sphi + zv*stheta*cphi; 
      double yy = pex*ctheta*sphi + pey*cphi + zv*stheta*sphi;
      double zz = -pex*stheta + zv*ctheta;

      G4ThreeVector pos  = hitPoint + G4ThreeVector(xx,yy,zz);
      zv = std::abs(pos.z()) - gpar[4] - 0.5*gpar[1];
      G4ThreeVector lpos = G4ThreeVector(pos.x(),pos.y(),zv);

      zv = fibre->zShift(lpos,depth,0);     // distance to PMT !

      double r  = pos.perp();
      double p  = fibre->attLength(pe[i].lambda());
      double fi = pos.phi();
      if (fi < 0) fi += twopi;
      int    isect = int(fi/dphi) + 1;
      isect        = (isect + 1) / 2;
      double dfi   = ((isect*2-1)*dphi - fi);
      if (dfi < 0) dfi = -dfi;
      double dfir  = r * sin(dfi);

#ifdef DebugLog
      edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Position shift " << xx 
   			              << ", " << yy << ", "  << zz << ": " << pos 
			              << " R " << r << " Phi " << fi << " Section " 
			              << isect << " R*Dfi " << dfir << " Dist " << zv;
#endif

      zz           = std::abs(pos.z());
      double r1    = G4UniformRand();
      double r2    = G4UniformRand();
      double r3    = -9999.;
      if (backward)     r3    = G4UniformRand();
      if (!applyFidCut) dfir += gpar[5];

#ifdef DebugLog
      edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: rLimits " << rInside(r)
			       << " attenuation " << r1 <<":" << exp(-p*zv) 
			       << " r2 " << r2 << " r3 " << r3 << " rDfi "  
			       << gpar[5] << " zz " 
			       << zz << " zLim " << gpar[4] << ":" 
			       << gpar[4]+gpar[1] << "\n"
			       << "  rInside(r) :" << rInside(r) 
			       << "  r1 <= exp(-p*zv) :" <<  (r1 <= exp(-p*zv))
			       << "  r2 <= probMax :"    <<  (r2 <= probMax*weight)
			       << "  r3 <= backProb :"   <<  (r3 <= backProb) 
			       << "  dfir > gpar[5] :"   <<  (dfir > gpar[5])
			       << "  zz >= gpar[4] :"    <<  (zz >= gpar[4])
			       << "  zz <= gpar[4]+gpar[1] :" 
			       << (zz <= gpar[4]+gpar[1]);   
#endif

      if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax*weight && 
	  dfir > gpar[5] && zz >= gpar[4] && zz <= gpar[4]+gpar[1] && 
	  r3 <= backProb && (depth != 2 || zz >= gpar[4]+gpar[0])) {

	oneHit.position = pos;
	oneHit.depth    = depth;
	oneHit.time     = (tSlice+(pe[i].t())+(fibre->tShift(lpos,depth,1)));
	hit.push_back(oneHit);
#ifdef DebugLog
	edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Final Hit " << nHit 
				        <<" position " << (hit[nHit].position) 
				        << " Depth " << (hit[nHit].depth) <<" Time " 
				        << tSlice << ":" << pe[i].t() << ":" 
				        << fibre->tShift(lpos,depth,1) << ":" 
				        << (hit[nHit].time);
#endif
	nHit++;
      }
#ifdef DebugLog
      else  LogDebug("FastCalorimetry") << "HFShowerLibrary: REJECTED !!!";
#endif
      if (onlyLong && zz >= gpar[4]+gpar[0] && zz <= gpar[4]+gpar[1]) {
	r1    = G4UniformRand();
	r2    = G4UniformRand();
	if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax && dfir > gpar[5]){
	  oneHit.position = pos;
	  oneHit.depth    = 2;
	  oneHit.time     = (tSlice+(pe[i].t())+(fibre->tShift(lpos,2,1)));
	  hit.push_back(oneHit);
#ifdef DebugLog
	  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Final Hit " << nHit 
				   << " position " << (hit[nHit].position) 
				   << " Depth " << (hit[nHit].depth) <<" Time "
				   << tSlice << ":" << pe[i].t() << ":" 
				   << fibre->tShift(lpos,2,1) << ":" 
				   << (hit[nHit].time);
#endif
	  nHit++;
	}
      }
    }
  }

#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Total Hits " << nHit
			          << " out of " << npe << " PE";
#endif

  if (nHit > npe && !onlyLong)
    edm::LogWarning("FastCalorimetry") << "HFShowerLibrary: Hit buffer " << npe 
				       << " smaller than " << nHit << " Hits";
  return hit;

}

bool FastHFShowerLibrary::rInside(double r) {

  if (r >= rMin && r <= rMax) return true;
  else                        return false;
}

void FastHFShowerLibrary::getRecord(int type, int record) {

  int nrc     = record-1;
  photon.clear();
  photo->clear();
  if (type > 0) {
    if (newForm) {
      hadBranch->SetAddress(&photo);
      hadBranch->GetEntry(nrc+totEvents);
    } else {
      hadBranch->SetAddress(&photon);
      hadBranch->GetEntry(nrc);
    }
  } else {
    if (newForm) {
      emBranch->SetAddress(&photo);
    } else {
      emBranch->SetAddress(&photon);
    }
    emBranch->GetEntry(nrc);
  }

#ifdef DebugLog
  int nPhoton = (newForm) ? photo->size() : photon.size();
  LogDebug("FastCalorimetry") << "HFShowerLibrary::getRecord: Record " << record
                       << " of type " << type << " with " << nPhoton 
                       << " photons";
  for (int j = 0; j < nPhoton; j++) 
    if (newForm) LogDebug("FastCalorimetry") << "Photon " << j << " " << photo->at(j);
    else         LogDebug("FastCalorimetry") << "Photon " << j << " " << photon[j];
#endif
}

void FastHFShowerLibrary::loadEventInfo(TBranch* branch) {

  if (branch) {
    std::vector<HFShowerLibraryEventInfo> eventInfoCollection;
    branch->SetAddress(&eventInfoCollection);
    branch->GetEntry(0);
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::loadEventInfo loads "
                             << " EventInfo Collection of size "
                             << eventInfoCollection.size() << " records";
    totEvents   = eventInfoCollection[0].totalEvents();
    nMomBin     = eventInfoCollection[0].numberOfBins();
    evtPerBin   = eventInfoCollection[0].eventsPerBin();
    libVers     = eventInfoCollection[0].showerLibraryVersion();
    listVersion = eventInfoCollection[0].physListVersion();
    pmom        = eventInfoCollection[0].energyBins();
  } else {
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::loadEventInfo loads "
                             << " EventInfo from hardwired numbers";
    nMomBin     = 16;
    evtPerBin   = 5000;
    totEvents   = nMomBin*evtPerBin;
    libVers     = 1.1;
    listVersion = 3.6;
    pmom        = {2,3,5,7,10,15,20,30,50,75,100,150,250,350,500,1000};
  }
  for (int i=0; i<nMomBin; i++) 
    pmom[i] *= GeV;
}

void FastHFShowerLibrary::interpolate(int type, double pin) {

#ifdef DebugLog
  LogDebug("FastCalorimetry") << "HFShowerLibrary:: Interpolate for Energy " <<pin/GeV
		              << " GeV with " << nMomBin << " momentum bins and " 
		              << evtPerBin << " entries/bin -- total " << totEvents;
#endif

  int irc[2];
  double w = 0.;
  double r = G4UniformRand();

  if (pin<pmom[0]) {
    w = pin/pmom[0];
    irc[1] = int(evtPerBin*r) + 1;
    irc[0] = 0;
  } else {
    for (int j=0; j<nMomBin-1; j++) {
      if (pin >= pmom[j] && pin < pmom[j+1]) {
	w = (pin-pmom[j])/(pmom[j+1]-pmom[j]);
	if (j == nMomBin-2) { 
	  irc[1] = int(evtPerBin*0.5*r);
	} else {
	  irc[1] = int(evtPerBin*r);
	}
	irc[1] += (j+1)*evtPerBin + 1;
	r = G4UniformRand();
	irc[0] = int(evtPerBin*r) + 1 + j*evtPerBin;
	if (irc[0]<0) {
	  edm::LogWarning("FastCalorimetry") << "HFShowerLibrary:: Illegal irc[0] = "
				             << irc[0] << " now set to 0";
	  irc[0] = 0;
	} else if (irc[0] > totEvents) {
	  edm::LogWarning("FastCalorimetry") << "HFShowerLibrary:: Illegal irc[0] = "
				             << irc[0] << " now set to "<< totEvents;
	  irc[0] = totEvents;
	}
      }
    }
  }
  if (irc[1]<1) {
    edm::LogWarning("FastCalorimetry") << "HFShowerLibrary:: Illegal irc[1] = " 
				       << irc[1] << " now set to 1";

    irc[1] = 1;
  } else if (irc[1] > totEvents) {
    edm::LogWarning("FastCalorimetry") << "HFShowerLibrary:: Illegal irc[1] = " 
				       << irc[1] << " now set to "<< totEvents;

    irc[1] = totEvents;
  }

#ifdef DebugLog
  LogDebug("FastCalorimetry") << "HFShowerLibrary:: Select records " << irc[0] 
		              << " and " << irc[1] << " with weights " << 1-w 
		              << " and " << w;
#endif

  pe.clear(); 
  npe       = 0;
  int npold = 0;
  for (int ir=0; ir < 2; ir++) {
    if (irc[ir]>0) {
      getRecord (type, irc[ir]);
      int nPhoton = (newForm) ? photo->size() : photon.size();
      npold      += nPhoton;
      for (int j=0; j<nPhoton; j++) {
	r = G4UniformRand();
	if ((ir==0 && r > w) || (ir > 0 && r < w)) {
	  storePhoton (j);
	}
      }
    }
  }

  if ((npe > npold || (npold == 0 && irc[0] > 0)) && !(npe == 0 && npold == 0)) 
    edm::LogWarning("FastCalorimetry") << "HFShowerLibrary: Interpolation Warning =="
				<< " records " << irc[0] << " and " << irc[1]
				<< " gives a buffer of " << npold 
				<< " photons and fills " << npe << " *****";
#ifdef DebugLog
  else
    LogDebug("FastCalorimetry") << "HFShowerLibrary: Interpolation == records " 
			 << irc[0] << " and " << irc[1] << " gives a "
			 << "buffer of " << npold << " photons and fills "
			 << npe << " PE";
  for (int j=0; j<npe; j++)
    LogDebug("FastCalorimetry") << "Photon " << j << " " << pe[j];
#endif

}

void FastHFShowerLibrary::extrapolate(int type, double pin) {

  int nrec   = int(pin/pmom[nMomBin-1]);
  double w   = (pin - pmom[nMomBin-1]*nrec)/pmom[nMomBin-1];
  nrec++;
#ifdef DebugLog
  LogDebug("FastCalorimetry") << "HFShowerLibrary:: Extrapolate for Energy " << pin 
		       << " GeV with " << nMomBin << " momentum bins and " 
		       << evtPerBin << " entries/bin -- total " << totEvents 
		       << " using " << nrec << " records";
#endif

  std::vector<int> irc(nrec);

  for (int ir=0; ir<nrec; ir++) {
    double r = G4UniformRand();
    irc[ir] = int(evtPerBin*0.5*r) +(nMomBin-1)*evtPerBin + 1;
    if (irc[ir]<1) {
      edm::LogWarning("FastCalorimetry") << "HFShowerLibrary:: Illegal irc[" << ir 
				  << "] = " << irc[ir] << " now set to 1";
      irc[ir] = 1;
    } else if (irc[ir] > totEvents) {
      edm::LogWarning("FastCalorimetry") << "HFShowerLibrary:: Illegal irc[" << ir 
				  << "] = " << irc[ir] << " now set to "
				  << totEvents;
      irc[ir] = totEvents;
#ifdef DebugLog
    } else {
      LogDebug("FastCalorimetry") << "HFShowerLibrary::Extrapolation use irc[" 
			          << ir  << "] = " << irc[ir];
#endif
    }
  }

  pe.clear(); 
  npe       = 0;
  int npold = 0;
  for (int ir=0; ir<nrec; ir++) {
    if (irc[ir]>0) {
      getRecord (type, irc[ir]);
      int nPhoton = (newForm) ? photo->size() : photon.size();
      npold      += nPhoton;
      for (int j=0; j<nPhoton; j++) {
	double r = G4UniformRand();
	if (ir != nrec-1 || r < w) {
	  storePhoton (j);
	}
      }
#ifdef DebugLog
      LogDebug("FastCalorimetry") << "HFShowerLibrary: Record [" << ir << "] = " 
			          << irc[ir] << " npold = " << npold;
#endif
    }
  }
#ifdef DebugLog
  LogDebug("FastCalorimetry") << "HFShowerLibrary:: uses " << npold << " photons";
#endif

  if (npe > npold || npold == 0)
    edm::LogWarning("FastCalorimetry") << "HFShowerLibrary: Extrapolation Warning == "
				<< nrec << " records " << irc[0] << ", " 
				<< irc[1] << ", ... gives a buffer of " <<npold
				<< " photons and fills " << npe 
				<< " *****";
#ifdef DebugLog
  else
    LogDebug("FastCalorimetry") << "HFShowerLibrary: Extrapolation == " << nrec
			 << " records " << irc[0] << ", " << irc[1] 
			 << ", ... gives a buffer of " << npold 
			 << " photons and fills " << npe << " PE";
  for (int j=0; j<npe; j++)
    LogDebug("FastCalorimetry") << "Photon " << j << " " << pe[j];
#endif
}

void FastHFShowerLibrary::storePhoton(int j) {

  if (newForm) pe.push_back(photo->at(j));
  else         pe.push_back(photon[j]);
#ifdef DebugLog
  LogDebug("FastCalorimetry") << "HFShowerLibrary: storePhoton " << j << " npe " 
		              << npe << " " << pe[npe];
#endif

  npe++;
}

std::vector<double> FastHFShowerLibrary::getDDDArray(const std::string & str, 
						 const DDsvalues_type & sv, 
						 int & nmin) {

#ifdef DebugLog
  LogDebug("FastCalorimetry") << "HFShowerLibrary:getDDDArray called for " << str 
		              << " with nMin " << nmin;
#endif

  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("FastCalorimetry") << "HFShowerLibrary:getDDDArray value " << value;
#endif

    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("FastCalorimetry") << "HFShowerLibrary : # of " << str 
				  << " bins " << nval << " < " << nmin 
				  << " ==> illegal";
	throw cms::Exception("Unknown", "HFShowerLibrary")
	  << "nval < nmin for array " << str << "\n";
      }
    } else {
      if (nval < 2) {
	edm::LogError("FastCalorimetry") << "HFShowerLibrary : # of " << str 
				  << " bins " << nval << " < 2 ==> illegal"
				  << " (nmin=" << nmin << ")";
	throw cms::Exception("Unknown", "HFShowerLibrary")
	  << "nval < 2 for array " << str << "\n";
      }
    }
    nmin = nval;

    return fvec;
  } else {
    edm::LogError("FastCalorimetry") << "HFShowerLibrary : cannot get array " << str;

    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "cannot get array " << str << "\n";
  }
}
