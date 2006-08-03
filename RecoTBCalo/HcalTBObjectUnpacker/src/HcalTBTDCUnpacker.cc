#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBTDCUnpacker.h"
#include "FWCore/Utilities/interface/Exception.h"

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
static const int lcBeamHalo1       = 97;
static const int lcBeamHalo2       = 98;
static const int lcBeamHalo3       = 99;
static const int lcBeamHalo4       = 100;
static const int lcTOF1            = 129;
static const int lcTOF2            = 130;

namespace hcaltb {

HcalTBTDCUnpacker::HcalTBTDCUnpacker(bool include_unmatched_hits) :
  includeUnmatchedHits_(include_unmatched_hits) {
//  setupWC(); reads it from configuration file
}
void HcalTBTDCUnpacker::setCalib(const vector<vector<string> >& calibLines_) {
        for(int i=0;i<131;i++)
         {
          tdc_ped[i]=0.;tdc_convers[i]=1.;
         }
        for(unsigned int ii=0;ii<calibLines_.size();ii++)
         {
//   TDC configuration
          if(calibLines_[ii][0]=="TDC")
                {
                if(calibLines_[ii].size()==4)
                  {
                  int channel=atoi(calibLines_[ii][1].c_str());
                  tdc_ped[channel]=atof(calibLines_[ii][2].c_str());
                  tdc_convers[channel]=atof(calibLines_[ii][3].c_str());
        //        printf("Got TDC %i ped %f , conversion %f\n",channel, tdc_ped[channel],tdc_convers[channel]);
                  }
                 else
                  {
               throw cms::Exception("Incomplete configuration") << 
		"Wrong TDC configuration format : expected 3 parameters, got "<<calibLines_[ii].size()-1;
                  }
                } // End of the TDCs

//   Wire chambers calibration
          if(calibLines_[ii][0]=="WC")
                {
                if(calibLines_[ii].size()==6)
                  {
                  int plane=atoi(calibLines_[ii][1].c_str());
                  wc_[plane].b0=atof(calibLines_[ii][2].c_str());
                  wc_[plane].b1=atof(calibLines_[ii][3].c_str());
                  wc_[plane].mean=atof(calibLines_[ii][4].c_str());
                  wc_[plane].sigma=atof(calibLines_[ii][5].c_str());
       //         printf("Got WC plane %i b0 %f, b1 %f, mean %f, sigma %f\n",plane, 
       //		 wc_[plane].b0,wc_[plane].b1,wc_[plane].mean,wc_[plane].sigma);
                  }
                 else
                  {
               throw cms::Exception("Incomplete configuration") << 
		"Wrong Wire Chamber configuration format : expected 5 parameters, got "<<calibLines_[ii].size()-1;
                  }
                } // End of the Wire Chambers

         } // End of the CalibLines
        }

  void HcalTBTDCUnpacker::unpack(const FEDRawData& raw,
			       HcalTBEventPosition& pos,
			       HcalTBTiming& timing) const {
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

  struct CombinedTDCQDCDataFormat {
    unsigned int cdfHeader0,cdfHeader1,cdfHeader2,cdfHeader3;
    unsigned int n_qdc_hits; // Count of QDC channels
    unsigned int n_tdc_hits; // upper/lower TDC counts    
    unsigned short qdc_values[4];
  };

//static const double CONVERSION_FACTOR=25.0/32.0;

void HcalTBTDCUnpacker::unpackHits(const FEDRawData& raw,
				   std::vector<Hit>& hits) const {
  const ClassicTDCDataFormat* tdc=(const ClassicTDCDataFormat*)raw.data();

  if (raw.size()<3*8) {
    throw cms::Exception("Missing Data") << "No data in the TDC block";
  }

  const unsigned int* hitbase=0;
  unsigned int totalhits=0;

  // old TDC (767)
  if (tdc->n_max_hits!=192) {
    const CombinedTDCQDCDataFormat* qdctdc=(const CombinedTDCQDCDataFormat*)raw.data();
    hitbase=(unsigned int*)(qdctdc);
    hitbase+=6; // header
    hitbase+=qdctdc->n_qdc_hits/2; // two unsigned short per unsigned long
    totalhits=qdctdc->n_tdc_hits&0xFFFF; // mask off high bits

    //    for (unsigned int i=0; i<qdctdc->n_qdc_hits; i++)
    //      printf("QADC: %02d %d\n",i,qdctdc->qdc_values[i]&0xFFF);

  } else {
    hitbase=&(tdc->hits[0]);
    totalhits=tdc->n_hits;
  }

  for (unsigned int i=0; i<totalhits; i++) {
    Hit h;    
    h.channel=(hitbase[i]&0x7FC00000)>>22; // hardcode channel assignment
    h.time=(hitbase[i]&0xFFFFF)*tdc_convers[h.channel]; 
    hits.push_back(h);
    //        printf("V767: %d %f\n",h.channel,h.time);
  }

  // new TDC (V775)
  if (tdc->n_max_hits!=192) {
    const CombinedTDCQDCDataFormat* qdctdc=(const CombinedTDCQDCDataFormat*)raw.data();
    hitbase=(unsigned int*)(qdctdc);
    hitbase+=6; // header
    hitbase+=qdctdc->n_qdc_hits/2; // two unsigned short per unsigned long
    hitbase+=(qdctdc->n_tdc_hits&0xFFFF); // same length
    totalhits=(qdctdc->n_tdc_hits&0xFFFF0000)>>16; // mask off high bits    
    
    for (unsigned int i=0; i<totalhits; i++) {
      Hit h;    
      h.channel=129+i;
      h.time=(hitbase[i]&0xFFF)*tdc_convers[h.channel] ;
      hits.push_back(h);
      //      printf("V775: %d %f\n",h.channel,h.time);
    }
  }

}

void HcalTBTDCUnpacker::reconstructTiming(const std::vector<Hit>& hits,
					  HcalTBTiming& timing) const {
  std::vector<Hit>::const_iterator j;
  double trigger_time=0;
  double ttc_l1a_time=0;
  double beam_coinc=0;
  double laser_flash=0;
  double qie_phase=0;
  double TOF1_time=0;
  double TOF2_time=0;
  
  std::vector<double> m1hits, m2hits, m3hits, s1hits, s2hits, s3hits, s4hits,
                      bh1hits, bh2hits, bh3hits, bh4hits;

  for (j=hits.begin(); j!=hits.end(); j++) {
    switch (j->channel) {
    case lcTriggerTime:     trigger_time   = j->time-tdc_ped[lcTriggerTime];  break;
    case lcTTCL1ATime:      ttc_l1a_time   = j->time-tdc_ped[lcTTCL1ATime];  break;
    case lcBeamCoincidence: beam_coinc     = j->time-tdc_ped[lcBeamCoincidence];  break;
    case lcLaserFlash:      laser_flash    = j->time-tdc_ped[lcLaserFlash];  break;
    case lcQIEPhase:        qie_phase      = j->time-tdc_ped[lcQIEPhase];  break;
    case lcMuon1:           m1hits.push_back(j->time-tdc_ped[lcMuon1]); break;
    case lcMuon2:           m2hits.push_back(j->time-tdc_ped[lcMuon2]); break;
    case lcMuon3:           m3hits.push_back(j->time-tdc_ped[lcMuon3]); break;
    case lcScint1:          s1hits.push_back(j->time-tdc_ped[lcScint1]); break;
    case lcScint2:          s2hits.push_back(j->time-tdc_ped[lcScint2]); break;
    case lcScint3:          s3hits.push_back(j->time-tdc_ped[lcScint3]); break;
    case lcScint4:          s4hits.push_back(j->time-tdc_ped[lcScint4]); break;
    case lcTOF1:            TOF1_time   = j->time-tdc_ped[lcTOF1];  break;
    case lcTOF2:            TOF2_time   = j->time-tdc_ped[lcTOF2];  break;
    case lcBeamHalo1:       bh1hits.push_back(j->time-tdc_ped[lcBeamHalo1]); break;
    case lcBeamHalo2:       bh2hits.push_back(j->time-tdc_ped[lcBeamHalo2]); break;
    case lcBeamHalo3:       bh3hits.push_back(j->time-tdc_ped[lcBeamHalo3]); break;
    case lcBeamHalo4:       bh4hits.push_back(j->time-tdc_ped[lcBeamHalo4]); break;
    default: break;
    }
  }

  timing.setTimes(trigger_time,ttc_l1a_time,beam_coinc,laser_flash,qie_phase,TOF1_time,TOF2_time);
  timing.setHits (m1hits,m2hits,m3hits,s1hits,s2hits,s3hits,s4hits,bh1hits,bh2hits,bh3hits,bh4hits);

}

const int HcalTBTDCUnpacker::WC_CHANNELIDS[PLANECOUNT*3] = { 
                                                     12, 13, 14, // WCA LR plane 
						     10, 11, 15, // WCA UD plane
						     22, 23, 24, // WCB LR plane
						     20, 21, 25, // WCB UD plane
						     32, 33, 34, // WCC LR plane
						     30, 31, 35, // WCC UD plane
						     42, 43, 44, // WCD LR plane
						     40, 41, 45, // WCD UD plane
						     52, 53, 54, // WCE LR plane
						     50, 51, 55, // WCE UD plane 
						    101, 102, 103, // WCF LR plane (was WC1)
						    104, 105, 106, // WCF UD plane (was WC1)
						    107, 108, 109, // WCG LR plane (was WC2)
						    110, 111, 112, // WCG UD plane (was WC2)
						    113, 114, 115, // WCH LR plane (was WC3)
						    116, 117, 118, // WCH UD plane (was WC3)

};

static const double TDC_OFFSET_CONSTANT = 12000;
static const double N_SIGMA = 2.5;

/* Obsolated - reads it from the configuration file
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
*/

void HcalTBTDCUnpacker::reconstructWC(const std::vector<Hit>& hits, HcalTBEventPosition& ep) const {
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
	plane_hits.push_back(wc_[plane].b0 + 
			     wc_[plane].b1 * (hits1[ii]-hits2[jmin]));
	hits1[ii]=-99999;
	hits2[jmin]=-99999;
	hitsA[lmin]=99999;
      }
    }

    if (includeUnmatchedHits_)   // unmatched hits (all pairs get in here)
      for (int ii=0; ii<n1; ii++) {
	if (hits1[ii]<-99990) continue;
	for (int jj=0; jj<n2; jj++) {
	  if (hits2[jj]<-99990) continue;
	  plane_hits.push_back(wc_[plane].b0 + 
			       wc_[plane].b1 * (hits1[ii]-hits2[jj]));
	}
      }

    if ((plane%2)==0) x=plane_hits;
    else {
      char chamber='A'+plane/2;
      ep.setChamberHits(chamber,x,plane_hits);
    }
  }
  
}

}
