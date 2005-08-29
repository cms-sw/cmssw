#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBTDCUnpacker.h"

// Timing channels
static const int lcTriggerTime     = 1;
static const int lcTTCL1ATime      = 2;
static const int lcBeamCoincidence = 3;
static const int lcLaserFlash      = 4;
static const int lcQIEPhase        = 5;

static const int lcMuon1           = 90;
static const int lcMuon2           = 91;
static const int lcMuon3           = 92;
static const int lcScint1          = 93;
static const int lcScint2          = 94;
static const int lcScint3          = 95;
static const int lcScint4          = 96;


HcalTBTDCUnpacker::HcalTBTDCUnpacker() {
  setupWC();
}

void HcalTBTDCUnpacker::unpack(const raw::FEDRawData& raw,
			       hcaltb::HcalTBEventPosition& pos,
			       hcaltb::HcalTBTiming& timing) const {
  std::vector<Hit> hits;

  unpackHits(raw, hits);

  reconstructWC(hits, pos);
  reconstructTiming(hits, timing);

}

struct ClassicTDCDataFormat {
  unsigned int cdfHeader0,cdfHeader1,cdfHeader2,cdfHeader3;
  unsigned int n_max_hits; // maximum number of hits possible in the block
  unsigned int n_hits;
  unsigned int hits[2];
};

static const double CONVERSION_FACTOR=25.0/32.0;

void HcalTBTDCUnpacker::unpackHits(const raw::FEDRawData& raw,
				   std::vector<Hit>& hits) const {
  const ClassicTDCDataFormat* tdc=(const ClassicTDCDataFormat*)raw.data();

  for (unsigned int i=0; i<tdc->n_hits; i++) {
    Hit h;
    h.time=(tdc->hits[i]&0xFFFFF) * CONVERSION_FACTOR;
    h.channel=(tdc->hits[i]&0x7FC00000)>>22;
    hits.push_back(h);
  }

}

void HcalTBTDCUnpacker::reconstructTiming(const std::vector<Hit>& hits,
					  hcaltb::HcalTBTiming& timing) const {
  std::vector<Hit>::const_iterator j;
  double trigger_time=0;
  double ttc_l1a_time=0;
  double beam_coinc=0;
  double laser_flash=0;
  double qie_phase=0;
  
  std::vector<double> m1hits, m2hits, m3hits, s1hits, s2hits, s3hits, s4hits;

  for (j=hits.begin(); j!=hits.end(); j++) {
    switch (j->channel) {
    case lcTriggerTime:     trigger_time   = j->time;  break;
    case lcTTCL1ATime:      ttc_l1a_time   = j->time;  break;
    case lcBeamCoincidence: beam_coinc     = j->time;  break;
    case lcLaserFlash:      laser_flash    = j->time;  break;
    case lcQIEPhase:        qie_phase      = j->time;  break;
    case lcMuon1:           m1hits.push_back(j->time); break;
    case lcMuon2:           m2hits.push_back(j->time); break;
    case lcMuon3:           m3hits.push_back(j->time); break;
    case lcScint1:          s1hits.push_back(j->time); break;
    case lcScint2:          s2hits.push_back(j->time); break;
    case lcScint3:          s3hits.push_back(j->time); break;
    case lcScint4:          s4hits.push_back(j->time); break;
    default: break;
    }
  }

  timing.setTimes(trigger_time,ttc_l1a_time,beam_coinc,laser_flash,qie_phase);
  timing.setHits (m1hits,m2hits,m3hits,s1hits,s2hits,s3hits,s4hits);

}

const int HcalTBTDCUnpacker::WC_CHANNELIDS[10*3] = { 12, 13, 14,
						     10, 11, 15,
						     22, 23, 24,
						     20, 21, 25,
						     32, 33, 34,
						     30, 31, 35,
						     42, 43, 44,
						     40, 41, 45,
						     52, 53, 54,
						     50, 51, 55 };

static const double TDC_OFFSET_CONSTANT = 12000;
static const double N_SIGMA = 2.5;

void HcalTBTDCUnpacker::setupWC() {

  wc_[0].b0 = -0.0870056; wc_[0].b1 = -0.193263;  // WCA planes
  wc_[1].b0 = -0.0288171; wc_[1].b1 = -0.191231;

  wc_[2].b0 = -0.2214840; wc_[2].b1 = -0.191683;  // WCB planes
  wc_[3].b0 = -1.0847300; wc_[3].b1 = -0.187691;

  wc_[4].b0 = -0.1981440; wc_[4].b1 = -0.192760;  // WCC planes
  wc_[5].b0 =  0.4230690; wc_[5].b1 = -0.192278;

  wc_[6].b0 = -0.6039130; wc_[6].b1 = -0.185674;  // WCD planes
  wc_[7].b0 = -0.4366590; wc_[7].b1 = -0.184992;

  wc_[8].b0 =  1.7016400; wc_[8].b1 = -0.185575;  // WCE planes
  wc_[9].b0 = -0.2324480; wc_[9].b1 = -0.185367;

  wc_[0].mean=225.2; wc_[0].sigma=5.704; 
  wc_[1].mean=223.5; wc_[1].sigma=5.862; 
  wc_[2].mean=227.2; wc_[2].sigma=5.070; 
  wc_[3].mean=235.7; wc_[3].sigma=6.090; 
  wc_[4].mean=243.3; wc_[4].sigma=7.804; 
  wc_[5].mean=230.3; wc_[5].sigma=28.91; 

  wc_[6].mean=225.0; wc_[6].sigma=6.000; 
  wc_[7].mean=225.0; wc_[7].sigma=6.000; 
  wc_[8].mean=225.0; wc_[8].sigma=6.000; 
  wc_[9].mean=225.0; wc_[9].sigma=6.000; 
}

void HcalTBTDCUnpacker::reconstructWC(const std::vector<Hit>& hits, hcaltb::HcalTBEventPosition& ep) const {
  // process all planes, looping over all hits...
  const int MAX_HITS=100;
  float hits1[MAX_HITS], hits2[MAX_HITS], hitsA[MAX_HITS];
  int n1,n2,nA,chan1,chan2,chanA;
  
  std::vector<double> x;

  for (int plane=0; plane<PLANECOUNT; plane++) {
    n1=0; n2=0; nA=0;

    std::vector<double> plane_hits;

    chan1=WC_CHANNELIDS[plane*3];
    chan2=WC_CHANNELIDS[plane*3+1];
    chanA=WC_CHANNELIDS[plane*3+2];

    for (std::vector<Hit>::const_iterator j=hits.begin(); j!=hits.end(); j++) {
      if (j->channel==chan1 && n1<MAX_HITS) {
	hits1[n1]=j->time-TDC_OFFSET_CONSTANT; n1++;
      }
      if (j->channel==chan2 && n2<MAX_HITS) {
	hits2[n2]=j->time-TDC_OFFSET_CONSTANT; n2++;
      }
      if (j->channel==chanA && nA<MAX_HITS) {
	hitsA[nA]=j->time-TDC_OFFSET_CONSTANT; nA++;
      }
    }
    
    // anode-matched hits
    for (int ii=0; ii<n1; ii++) {
      int jmin=-1, lmin=-1;
      float dsumMin=99999;
      for (int jj=0; jj<n2; jj++) {
	for (int ll=0; ll<nA; ll++) {
	  float dsum=fabs(wc_[plane].mean - hits1[ii] - hits2[jj] + 2.0*hitsA[ll]);
          if(dsum<(N_SIGMA*wc_[plane].sigma) && dsum<dsumMin){
            jmin=jj;
            lmin=ll;
            dsumMin=dsum;
           }
	}	      
      }
      if (jmin>=0) {
	plane_hits.push_back(wc_[plane].b0+wc_[plane].b1 * (hits1[ii]-hits2[jmin]));
	hits1[ii]=-99999;
	hits2[jmin]=-99999;
	hitsA[lmin]=99999;
      }
    }

    // unmatched hits (all pairs get in here)
    for (int ii=0; ii<n1; ii++) {
      if (hits1[ii]<-99990) continue;
      for (int jj=0; jj<n2; jj++) {
	if (hits2[jj]<-99990) continue;
	plane_hits.push_back(wc_[plane].b0+wc_[plane].b1 * (hits1[ii]-hits2[jj]));
      }
    }

    if ((plane%2)==0) x=plane_hits;
    else {
      char chamber='A'+plane/2;
      ep.setChamberHits(chamber,x,plane_hits);
    }
  }
  
}
