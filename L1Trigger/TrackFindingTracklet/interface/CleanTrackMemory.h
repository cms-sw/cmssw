//This class stores the track fit
#ifndef CLEANTRACKMEMORY_H
#define CLEANTRACKMEMORY_H

#include "Tracklet.h"
#include "MemoryBase.h"

using namespace std;

class CleanTrackMemory:public MemoryBase{

public:

  CleanTrackMemory(string name, unsigned int iSector, 
	       double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addTrack(Tracklet* tracklet) {
    tracks_.push_back(tracklet);
  }

  unsigned int nTracks() const {return tracks_.size();}

  void clean() {
    //cout << "Cleaning tracks : "<<tracks_.size()<<endl;
    tracks_.clear();
  }

  bool foundTrack(ofstream& outres, L1SimTrack simtrk){
    bool match=false;
    double phioffset=phimin_-(phimax_-phimin_)/6.0;
    for(unsigned int i=0;i<tracks_.size();i++){
      match=match||tracks_[i]->foundTrack(simtrk,phimin_);
      if (tracks_[i]->foundTrack(simtrk,phimin_)) {
	Tracklet* tracklet=tracks_[i];
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
    return match;
  }

  void writeCT(bool first) {

    std::string fname="../data/MemPrints/CleanTrack/CleanTrack_";
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
      out_ <<"0x";
      if (j<16) out_<<"0";
      out_<<hex<<j<<dec<<" ";
      out_<<tracks_[j]->trackfitstr()<<" "<<hexFormat(tracks_[j]->trackfitstr());
      out_<<"\n";
    }
    out_.close();

    // --------------------------------------------------------------
    // print separately ALL cleaned tracks in single file
    if (writeAllCT) { 

      std::string fnameAll="CleanTracksAll.dat";
      if (first && getName()=="CT_L1L2" && iSector_==0) 
	out_.open(fnameAll.c_str());
      else 
	out_.open(fnameAll.c_str(),std::ofstream::app);
      
      if (tracks_.size()>0) 
	out_ << "BX= "<<(bitset<3>)bx_ << " event= " << event_  << " seed= " << getName() << " phisector= " << iSector_+1 << endl;

      for (unsigned int j=0;j<tracks_.size();j++){
	if (j<16) out_<<"0";
	out_<<hex<<j<<dec<<" ";
	out_<<tracks_[j]->trackfitstr();
	out_<<"\n";
      }
      out_.close();

    }
    // --------------------------------------------------------------


    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }

private:

  double phimin_;
  double phimax_;
  std::vector<Tracklet*> tracks_;

};

#endif
