#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

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

class CalibCorr {
public :
  CalibCorr(const char* infile, bool debug=false);
  ~CalibCorr() {}

  float getCorr(int run, unsigned int id);
private:
  void                     readCorr(const char* infile);
  std::vector<std::string> splitString(const std::string&);
  unsigned int getDetIdHE(int ieta, int iphi, int depth);
  unsigned int getDetId(int subdet, int ieta, int iphi, int depth);
  unsigned int correctDetId(const unsigned int& detId);

  static const     unsigned int nmax_=10;
  bool                          debug_;
  std::map<unsigned int,float>  corrFac_[nmax_];
  std::vector<int>              runlow_;
};

class CalibSelectRBX {
public:
  CalibSelectRBX(int rbx);
  ~CalibSelectRBX() {}

  bool isItRBX(const unsigned int);
  bool isItRBX(const std::vector<unsigned int> *);
  bool isItRBX(const int, const int);
private:
  int              subdet_, zside_;
  std::vector<int> phis_;
};

CalibCorr::CalibCorr(const char* infile, bool debug) : debug_(debug) {
  readCorr(infile);
}

float CalibCorr::getCorr(int run, unsigned int id) {
  float cfac(1.0);
  int ip(-1);
  for (unsigned int k=0; k<runlow_.size(); ++k) {
    unsigned int i = runlow_.size()-k-1;
    if (run >= runlow_[i]) {
      ip = (int)(i); break;
    }
  }
  if (debug_) std::cout << "Run " << run << " Perdiod " << ip << std::endl;
  unsigned idx = correctDetId(id);
  if (ip >= 0) {
    std::map<unsigned int,float>::iterator itr = corrFac_[ip].find(idx);
    if (itr != corrFac_[ip].end()) cfac = itr->second;
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
	std::cout << ncorr << ":" << runlow_.size() << " Run ranges\n";
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
  if ((id != detId) && debug_) 
    std::cout << "Correct Id " << std::hex << detId << " to " << id << std::dec
	      << "(Sub " << subdet << " eta " << ieta*zside << " phi " << iphi
	      << " depth " << depth << ")" << std::endl;
  return id;
}

CalibSelectRBX::CalibSelectRBX(int rbx) {
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
    /*
    std::cout << "isItRBX:subdet|zside|iphi " << subdet << ":" << zside 
		<< ":" << iphi << " OK " << ok << std::endl;
    */
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
      /*
      std::cout << "isItRBX: subdet|zside|iphi " << subdet << ":" << zside 
		<< ":" << iphi << std::endl;
      */
      if (ok) break;
    }
  }
//std::cout << "isItRBX: size " << detId->size() << " OK " << ok << std::endl;
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
  /*
  std::cout << "isItRBX: ieta " << ieta << " iphi " << iphi << " OK " << ok 
	    << std::endl;
  */
  return ok;
}
