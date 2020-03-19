//This class stores the track fit
#ifndef TRACKFITMEMORY_H
#define TRACKFITMEMORY_H

#include "Tracklet.h"
#include "MemoryBase.h"

using namespace std;

class TrackFitMemory:public MemoryBase{

public:

  TrackFitMemory(string name, unsigned int iSector, 
	       double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addTrack(Tracklet* tracklet) {
    tracks_.push_back(tracklet);
  }
  void addStubList(std::vector<std::pair<Stub*,L1TStub*>> stublist) {
    stublists_.push_back(stublist);
  }
  void addStubidsList(std::vector<std::pair<int,int>> stubidslist) {
    stubidslists_.push_back(stubidslist);
  }

  unsigned int nTracks() const {return tracks_.size();}
  unsigned int nStublists() const {return stublists_.size();}
  unsigned int nStubidslists() const {return stubidslists_.size();}

  Tracklet* getTrack(unsigned int i) const {
    return tracks_[i];
  }
  std::vector<std::pair<Stub*,L1TStub*>> getStublist(unsigned int i) const {
    return stublists_[i];
  }
  std::vector<std::pair<int,int>> getStubidslist(unsigned int i) const {
    return stubidslists_[i];
  }

  void clean() {
    //cout << "Cleaning tracks : "<<tracks_.size()<<endl;
    tracks_.clear();
    stublists_.clear();
    stubidslists_.clear();
  }

  bool foundTrack(ofstream& outres, L1SimTrack simtrk){
    bool match=false;
    double phioffset=phimin_;
    for(unsigned int i=0;i<tracks_.size();i++){
      if (tracks_[i]->getTrack()->duplicate()) continue;
      if (tracks_[i]->foundTrack(simtrk,phioffset)) match=true;
      if (tracks_[i]->foundTrack(simtrk,phioffset)) {
	Tracklet* tracklet=tracks_[i];
	
	/*
	
	int psmatches=0;
	
	if (tracklet->match(1)) psmatches++;
	if (tracklet->match(2)) psmatches++;
	if (tracklet->match(3)) psmatches++;

	cout << "psmatches : "<<psmatches<<endl;

	if (!tracklet->match(3)) continue;

	
	
	if (psmatches<3) continue;
	
	cout << "Tracklet: "
	     <<sinh(simtrk.eta())<<" "
	     <<simtrk.vz()<<" "
	     <<tracklet->stubptrs(1).second->r()<<" "
	     <<tracklet->stubptrs(1).second->z()<<" "
	     <<tracklet->zresid(1)<<" "
	     <<tracklet->stubptrs(2).second->r()<<" "
	     <<tracklet->stubptrs(2).second->z()<<" "
	     <<tracklet->zresid(2)<<" "
	     <<tracklet->innerStub()->r()<<" "
	     <<tracklet->innerStub()->z()<<" "
	     <<tracklet->t()<<" "
	     <<tracklet->z0()<<" "
      	     <<tracklet->tfitexact()<<" "
	     <<tracklet->z0fitexact()<<" "
	     <<endl;
	*/   
	  
	int charge = simtrk.trackid()/abs(simtrk.trackid());
	if(abs(simtrk.trackid())<100) charge = -charge; 
	double simphi=simtrk.phi();
	if (simphi<0.0) simphi+=2*M_PI; 
	int irinv=tracklet->irinvfit().value();
	if (irinv==0) irinv=1;
	int layerordisk=-1;
	if (tracklet->isBarrel()) {
	  layerordisk=tracklet->layer();
	} else {
	  layerordisk=tracklet->disk();
	}
	if (writeResEff) {
	  outres << layerordisk
		 <<" "<<tracklet->nMatches()
		 <<" "<<simtrk.pt()*charge
		 <<" "<<simphi
		 <<" "<<simtrk.eta()
		 <<" "<<simtrk.vz()
		 <<"   "
		 <<(0.3*3.8/100.0)/tracklet->rinvfit()
		 <<" "<<tracklet->phi0fit()+phioffset
		 <<" "<<asinh(tracklet->tfit())
		 <<" "<<tracklet->z0fit()
		 <<"   "
		 <<(0.3*3.8/100.0)/tracklet->rinvfitexact()
		 <<" "<<tracklet->phi0fitexact()+phioffset
		 <<" "<<asinh(tracklet->tfitexact())
		 <<" "<<tracklet->z0fitexact()
		 <<"   "
		 <<(0.3*3.8/100.0)/(irinv*krinvpars)
		 <<" "<<tracklet->iphi0fit().value()*kphi0pars+phioffset
		 <<" "<<asinh(tracklet->itfit().value()*ktpars)
		 <<" "<<tracklet->iz0fit().value()*kz
		 <<"   "
		 <<(0.3*3.8/100.0)/(1e-20+tracklet->fpgarinv().value()*krinvpars)
		 <<" "<<tracklet->fpgaphi0().value()*kphi0pars+phioffset
		 <<" "<<asinh(tracklet->fpgat().value()*ktpars)
		 <<" "<<tracklet->fpgaz0().value()*kz
		 <<"               "
		 <<(0.3*3.8/100.0)/(1e-20+tracklet->rinvapprox())
		 <<" "<<tracklet->phi0approx()+phioffset
		 <<" "<<asinh(tracklet->tapprox())
		 <<" "<<tracklet->z0approx()
		 <<endl;
	}
      }
    }
    return match;
  }
  void writeTF(bool first) {

    std::string fname="../data/MemPrints/FitTrack/TrackFit_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    //unsigned long int uu;
    for (unsigned int j=0;j<tracks_.size();j++){
      //uu = (((long int)tracks_[j]->irinvfit().value()&32767)<<44)|
      //(((long int)tracks_[j]->iphi0fit().value()&524287)<<25)|
      //(((long int)tracks_[j]->itfit().value()&16383)<<11)|
      //((long int)tracks_[j]->iz0fit().value()&2047);
      //out_<<"0000000000000000";
      //out_.fill('0');
      //out_.width(16);
      //out_<<std::hex<<uu;
      out_ << "0x";
      if (j<16) out_<<"0";
      out_<<hex<<j<<dec<<" ";
      out_<<tracks_[j]->trackfitstr()<<" "<<hexFormat(tracks_[j]->trackfitstr());
      out_<<"\n";
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }

private:

  double phimin_;
  double phimax_;
  std::vector<Tracklet*> tracks_;
  std::vector<std::vector<std::pair<Stub*,L1TStub*>>> stublists_;
  std::vector<std::vector<std::pair<int,int>>> stubidslists_;

};

#endif
