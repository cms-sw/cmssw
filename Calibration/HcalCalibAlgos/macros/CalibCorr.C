#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <TROOT.h>
#include <TChain.h>

void unpackDetId(unsigned int detId, int& subdet, int& zside, int& ieta, 
		 int& iphi, int& depth) {
  // The maskings are defined in DataFormats/DetId/interface/DetId.h
  //                      and in DataFormats/HcalDetId/interface/HcalDetId.h
  // The macro does not invoke the classes there and use them
  subdet = ((detId >> 25) & (0x7));
  if ((detId&0x1000000) == 0) {
    ieta   = ((detId >> 7) & 0x3F);
    zside  = (detId&0x2000)?(1):(-1);
    depth  = ((detId >> 14) & 0x1F);
    iphi   = (detId & 0x3F);
  } else {
    ieta   = ((detId >> 10) & 0x1FF);
    zside  = (detId&0x80000)?(1):(-1);
    depth  = ((detId >> 20) & 0xF);
    iphi   = (detId & 0x3FF);
  }
}

unsigned int truncateId(unsigned int detId, int truncateFlag, bool debug=false){
  //Truncate depth information of DetId's 
  unsigned int id(detId);
  if (debug) {
    std::cout << "Truncate 1 " << std::hex << detId << " " << id 
	      << std::dec << " Flag " << truncateFlag << std::endl;
  }
  int subdet, depth, zside, ieta, iphi;
  unpackDetId(detId, subdet, zside, ieta, iphi, depth);
  if (truncateFlag == 1) {
    //Ignore depth index of ieta values of 15 and 16 of HB
    if ((subdet == 1) && (ieta > 14)) depth  = 1;
  } else if (truncateFlag == 2) {
    //Ignore depth index of all ieta values
    depth = 1;
  } else if (truncateFlag == 3) {
    //Ignore depth index for depth > 1 in HE
    if ((subdet == 2) && (depth > 1)) depth = 2;
    else                              depth = 1;
  } else if (truncateFlag == 4) {
    //Ignore depth index for depth > 1 in HB
    if ((subdet == 1) && (depth > 1)) depth = 2;
    else                              depth = 1;
  } else if (truncateFlag == 5) {
    //Ignore depth index for depth > 1 in HB and HE
    if (depth > 1) depth = 2;
  }
  id = (subdet<<25) | (0x1000000) | ((depth&0xF)<<20) | ((zside>0)?(0x80000|(ieta<<10)):(ieta<<10));
  if (debug) {
    std::cout << "Truncate 2: " << subdet << " " << zside*ieta << " " 
	      << depth << " " << std::hex << id << " input " << detId 
	      << std::dec << std::endl;
  }
  return id;
}

double puFactor(int type, int ieta, double pmom, double eHcal, double ediff) {

  double fac(1.0);
  double frac = (type == 1) ? 0.02 : 0.03;
  if (pmom > 0 && ediff >  frac*pmom) {
    double a1(0), a2(0);
    if (type == 1) {
      a1 = -0.35; a2 = -0.65;
      if (std::abs(ieta) == 25) {
	a2 = -0.30;
      } else if (std::abs(ieta) > 25) {
	a1 = -0.45; a2 = -0.10;
      }
    } else {
      a1 = -0.39; a2 = -0.59;
      if (std::abs(ieta) >= 25) {
	a1 = -0.283; a2 = -0.272;
      } else if (std::abs(ieta) > 22) {
	a1 = -0.238; a2 = -0.241;
      }
    }
    fac = (1.0+a1*(eHcal/pmom)*(ediff/pmom)*(1+a2*(ediff/pmom)));
  }
  return fac;
}

double puFactorRho(int type, int ieta, double rho, double eHcal) {
  // type = 1: 2017 Data;  2: 2017 MC; 3: 2018 MC; 4: 2018AB; 5: 2018BC
  double par[30] = {0.0205395,-43.0914,2.67115,0.239674,-0.0228009,0.000476963,
		    0.129097,-105.831,9.58076,0.156392,-0.034671,0.000809736,
		    0.202391,-145.962,12.1489,0.329384,-0.0511365,0.00113219,
		    0.175356,-175.543,14.3414,0.294718,-0.049836,0.00106228,
		    0.134314,-175.809,13.5307,0.395943,-0.0539062,0.00111573};
  double energy(eHcal);
  if (type >= 1 && type <= 5) {
    int    eta = std::abs(ieta);
    int    it  = 6*(type-1);
    double ea  = (eta < 20) ? par[it] : ((((par[it+5]*eta+par[it+4])*eta+par[it+3])*eta+par[it+2])*eta+par[it+1]);
    energy -= (rho*ea);
  }
  return energy;
}

bool fillChain(TChain *chain, const char* inputFileList) {

  std::string fname(inputFileList);
  if (fname.substr(fname.size()-5,5) == ".root") {
    chain->Add(fname.c_str());
  } else {
    ifstream infile(inputFileList);
    if (!infile.is_open()) {
      std::cout << "** ERROR: Can't open '" << inputFileList << "' for input" 
		<< std::endl;
      return false;
    }
    while (1) {
      infile >> fname;
      if (!infile.good()) break;
      chain->Add(fname.c_str());
    }
    infile.close();
  }
  std::cout << "No. of Entries in this tree : " << chain->GetEntries()
	    << std::endl;
  return true;
}

class CalibCorr {
public :
  CalibCorr(const char* infile, bool useDepth, bool debug);
  ~CalibCorr() {}

  float getCorr(int run, unsigned int id);
private:
  void                     readCorr(const char* infile);
  void                     readCorrDepth(const char* infile);
  std::vector<std::string> splitString(const std::string&);
  unsigned int getDetIdHE(int ieta, int iphi, int depth);
  unsigned int getDetId(int subdet, int ieta, int iphi, int depth);
  unsigned int correctDetId(const unsigned int& detId);

  static const     unsigned int nmax_=10;
  bool                          ifDepth_, debug_;
  std::map<unsigned int,float>  corrFac_[nmax_], corrFacN_;
  std::vector<int>              runlow_;
};

class CalibSelectRBX {
public:
  CalibSelectRBX(int rbx, bool debug=false);
  ~CalibSelectRBX() {}

  bool isItRBX(const unsigned int);
  bool isItRBX(const std::vector<unsigned int> *);
  bool isItRBX(const int, const int);
private:
  bool             debug_;
  int              subdet_, zside_;
  std::vector<int> phis_;
};

CalibCorr::CalibCorr(const char* infile, bool ifDepth, bool debug) : 
  ifDepth_(ifDepth), debug_(debug) {
  if (ifDepth_) readCorrDepth(infile);
  else          readCorr(infile);
}

float CalibCorr::getCorr(int run, unsigned int id) {
  float cfac(1.0);
  unsigned idx = correctDetId(id);
  if (ifDepth_) {
    std::map<unsigned int,float>::iterator itr = corrFacN_.find(idx);
    if (itr != corrFacN_.end()) cfac = itr->second;
  } else {
    int ip(-1);
    for (unsigned int k=0; k<runlow_.size(); ++k) {
      unsigned int i = runlow_.size()-k-1;
      if (run >= runlow_[i]) {
	ip = (int)(i); break;
      }
    }
    if (debug_) {
      std::cout << "Run " << run << " Perdiod " << ip << std::endl;
    }
    if (ip >= 0) {
      std::map<unsigned int,float>::iterator itr = corrFac_[ip].find(idx);
      if (itr != corrFac_[ip].end()) cfac = itr->second;
    }
  }
  if (debug_) {
    int  subdet, zside, ieta, iphi, depth;
    unpackDetId(idx, subdet, zside, ieta, iphi, depth);
    std::cout << "ID " << std::hex << id << std::dec << " (Sub " << subdet 
	      << " eta " << zside*ieta << " phi " << iphi << " depth " << depth
	      << ")  Factor " << cfac << std::endl;
  }
  return cfac;
}

void CalibCorr::readCorrDepth(const char* infile) {

  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer [1024];
    unsigned int all(0), good(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0,5) == "depth") {
	continue; //ignore other comments
      } else {
	std::vector<std::string> items = splitString(bufferString);
	if (items.size () != 3) {
	  std::cout << "Ignore  line: " << buffer << " Size " << items.size();
	  for (unsigned int k=0; k<items.size(); ++k)
	    std::cout << " [" << k << "] : " << items[k];
	  std::cout << std::endl;
	} else {
	  ++good;
	  int   ieta  = std::atoi (items[1].c_str());
	  int   depth = std::atoi (items[0].c_str());
	  float corrf = std::atof (items[2].c_str());
	  int   nphi  = (std::abs(ieta) > 20) ? 36 : 72;
	  for (int i=1; i<=nphi; ++i) {
	    int        iphi = (nphi > 36) ? i : (2*i-1);
	    unsigned int id = getDetIdHE(ieta,iphi,depth);
	    corrFacN_[id]   = corrf;
	    if (debug_) {
	      std::cout << "ID " << std::hex << id << std::dec << ":" << id
			<< " (eta " << ieta << " phi " << iphi << " depth " 
			<< depth << ") " << corrFacN_[id] << std::endl;
	    }
	  }
	}
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
	      << std::endl;
  }
}

void CalibCorr::readCorr(const char* infile) {

  std::ifstream fInput(infile);
  unsigned int ncorr(0);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer [1024];
    unsigned int all(0), good(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0,5) == "#IOVs") {
	std::vector<std::string> items = splitString(bufferString.substr(6));
	ncorr = items.size() - 1;
	for (unsigned int n=0; n<ncorr; ++n) {
	  int run  = std::atoi (items[n].c_str());
	  runlow_.push_back(run);
	}
	std::cout << ncorr << ":" << runlow_.size() << " Run ranges" 
		  << std::endl;
	for (unsigned int n=0; n<runlow_.size(); ++n) 
	  std::cout << " [" << n << "] " << runlow_[n];
	std::cout << std::endl;
      } else if (buffer [0] == '#') {
	continue; //ignore other comments
      } else {
	std::vector<std::string> items = splitString(bufferString);
	if (items.size () != ncorr+3) {
	  std::cout << "Ignore  line: " << buffer << std::endl;
	} else {
	  ++good;
	  int   ieta  = std::atoi (items[0].c_str());
	  int   iphi  = std::atoi (items[1].c_str());
	  int   depth = std::atoi (items[2].c_str());
	  unsigned int id = getDetIdHE(ieta,iphi,depth);
	  for (unsigned int n=0; n<ncorr; ++n) {
	    float corrf = std::atof (items[n+3].c_str());
	    if (n<nmax_) corrFac_[n][id] = corrf;
	  }
	  if (debug_) {
	    std::cout << "ID " << std::hex << id << std::dec << ":" << id
		      << " (eta " << ieta << " phi " << iphi << " depth " 
		      << depth << ")";
	    for (unsigned int n=0; n<ncorr; ++n) 
	      std::cout << " " << corrFac_[n][id];
	    std::cout << std::endl;
	  }
	}
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
	      << std::endl;
  }
}

std::vector<std::string> CalibCorr::splitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || i == fLine.size ()) {
      if (!empty) {
	std::string item (fLine, start, i-start);
	result.push_back (item);
	empty = true;
      }
      start = i+1;
    } else {
      if (empty) empty = false;
    }
  }
  return result;
}

unsigned int CalibCorr::getDetIdHE(int ieta, int iphi, int depth) {
  return getDetId(2,ieta,iphi,depth);
}

unsigned int CalibCorr::getDetId(int subdet, int ieta, int iphi, int depth) {
  // All numbers used here are described as masks/offsets in 
  // DataFormats/HcalDetId/interface/HcalDetId.h
  unsigned int id_ = ((4<<28)|((subdet&0x7)<<25));
  id_ |= ((0x1000000) | ((depth&0xF)<<20) |
	  ((ieta>0)?(0x80000|(ieta<<10)):((-ieta)<<10)) |
	  (iphi&0x3FF));
  return id_;
}

unsigned int CalibCorr::correctDetId(const unsigned int & detId) {
  int subdet, ieta, zside, depth, iphi;
  unpackDetId(detId, subdet, zside, ieta, iphi, depth);
  if (subdet == 0) {
    if (ieta > 16)                    subdet = 2;
    else if (ieta == 16 && depth > 2) subdet = 2;
    else                              subdet = 1;
  }
  unsigned int id = getDetId(subdet,ieta*zside,iphi,depth);
  if ((id != detId) && debug_) {
    std::cout << "Correct Id " << std::hex << detId << " to " << id << std::dec
	      << "(Sub " << subdet << " eta " << ieta*zside << " phi " << iphi
	      << " depth " << depth << ")" << std::endl;
  }
  return id;
}

CalibSelectRBX::CalibSelectRBX(int rbx, bool debug) : debug_(debug) {
  zside_    = (rbx > 0) ? 1 : -1;
  subdet_   = (std::abs(rbx)/100)%10;
  if (subdet_ != 1) subdet_ = 2;
  int iphis = std::abs(rbx)%100;
  if (iphis > 0 && iphis <= 18) {
    for (int i=0; i<4; ++i) {
      int iphi = (iphis-2)*4+3+i;
      if (iphi < 1) iphi += 72;
      phis_.push_back(iphi);
    }
  }
  std::cout << "Select RBX " << rbx << " ==> Subdet " << subdet_ << " zside "
	    << zside_ << " with " << phis_.size() << " iphi values:";
  for (unsigned int i=0; i<phis_.size(); ++i) std::cout << " " << phis_[i];
  std::cout << std::endl;
}

bool CalibSelectRBX::isItRBX(const unsigned int detId) {
  bool ok(true);
  if (phis_.size() == 4) {
    int subdet, ieta, zside, depth, iphi;
    unpackDetId(detId, subdet, zside, ieta, iphi, depth);
    ok = ((subdet == subdet_) && (zside == zside_) &&
	  (std::find(phis_.begin(),phis_.end(),iphi) != phis_.end()));
    
    if (debug_) {
      std::cout << "isItRBX:subdet|zside|iphi " << subdet << ":" << zside 
		<< ":" << iphi << " OK " << ok << std::endl;
    }
  }
  return ok;
}

bool CalibSelectRBX::isItRBX(const std::vector<unsigned int> * detId) {
  bool ok(true);
  if (phis_.size() == 4) {
    ok = true;
    for (unsigned int i=0; i < detId->size(); ++i) {
      int subdet, ieta, zside, depth, iphi;
      unpackDetId((*detId)[i], subdet, zside, ieta, iphi, depth);
      ok = ((subdet == subdet_) && (zside == zside_) &&
	    (std::find(phis_.begin(),phis_.end(),iphi) != phis_.end()));
      if (debug_) {
	std::cout << "isItRBX: subdet|zside|iphi " << subdet << ":" << zside 
		  << ":" << iphi << std::endl;
      }
      if (ok) break;
    }
  }
  if (debug_) std::cout << "isItRBX: size " << detId->size() << " OK " << ok 
			<< std::endl;
  return ok;
}

bool CalibSelectRBX::isItRBX(const int ieta, const int iphi) {
  bool ok(true);
  if (phis_.size() == 4) {
    int zside = (ieta > 0) ? 1 : -1;
    int subd1 = (std::abs(ieta) <= 16) ? 1 : 2;
    int subd2 = (std::abs(ieta) >= 16) ? 2 : 1;
    ok        = (((subd1 == subdet_) || (subd2 == subdet_)) && 
		 (zside == zside_) &&
		 (std::find(phis_.begin(),phis_.end(),iphi) != phis_.end()));
  }
  if (debug_) {
    std::cout << "isItRBX: ieta " << ieta << " iphi " << iphi << " OK " << ok 
	      << std::endl;
  }
  return ok;
}
