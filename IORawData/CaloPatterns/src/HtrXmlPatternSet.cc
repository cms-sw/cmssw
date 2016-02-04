#include "HtrXmlPatternSet.h"
#include <math.h>
#include <iostream>

ChannelPattern::ChannelPattern() {
  for (int i=0; i<SAMPLES; i++) {
    fCReal[i]=0;
    fCCoded[i]=0;
    fCQuantized[i]=0;
  }
  m_sample_pos=0;
}

void ChannelPattern::Fill_by_hand(const HcalElectronicsMap *emap,int pattern_number) {

  for (int iSample=0;iSample<SAMPLES;iSample++) {
    fCCoded    [iSample]=0;
    fCQuantized[iSample]=0;
  }

  int dcc9[NUM_CRATES];
  int dcc19[NUM_CRATES];
  for (int jCrate=0;jCrate<NUM_CRATES;jCrate++) {
    dcc9[jCrate]=-1;
    dcc19[jCrate]=-1;
  }

  int iCrate;
  iCrate=4 ; dcc9[iCrate]=700; dcc19[iCrate]=701;
  iCrate=0 ; dcc9[iCrate]=702; dcc19[iCrate]=703;
  iCrate=1 ; dcc9[iCrate]=704; dcc19[iCrate]=705;
  iCrate=5 ; dcc9[iCrate]=706; dcc19[iCrate]=707;
  iCrate=11; dcc9[iCrate]=708; dcc19[iCrate]=709;
  iCrate=15; dcc9[iCrate]=710; dcc19[iCrate]=711;
  iCrate=17; dcc9[iCrate]=712; dcc19[iCrate]=713;
  iCrate=14; dcc9[iCrate]=714; dcc19[iCrate]=715;
  iCrate=10; dcc9[iCrate]=716; dcc19[iCrate]=717;
  iCrate=2 ; dcc9[iCrate]=718; dcc19[iCrate]=719;
  iCrate=9 ; dcc9[iCrate]=720; dcc19[iCrate]=721;
  iCrate=12; dcc9[iCrate]=722; dcc19[iCrate]=723;
  iCrate=3 ; dcc9[iCrate]=724; dcc19[iCrate]=725;
  iCrate=7 ; dcc9[iCrate]=726; dcc19[iCrate]=727;
  iCrate=6 ; dcc9[iCrate]=728; dcc19[iCrate]=729;
  iCrate=13; dcc9[iCrate]=730; dcc19[iCrate]=731;

  int *dcc=0;
  int spigot=-100;
  if (m_slot>=2 && m_slot<=8) {
    spigot=2*m_slot-m_tb-3;
    dcc=dcc9;
  }
  else if (m_slot>=13 && m_slot<=18) {
    spigot=2*m_slot-m_tb-25;
    dcc=dcc19;
  }
  else return;

  int fiber_channel=(m_chan-1)%3;
  int fiber=(m_chan-fiber_channel-1)/3 +1;
  int dcc_num=dcc[m_crate]-700;

  HcalElectronicsId heid(fiber_channel,fiber,spigot,dcc_num);
  heid.setHTR(m_crate,m_slot,m_tb);

  //if (heid.readoutVMECrateId()!=m_crate) std::cout << "crate: " << heid.readoutVMECrateId() << "; " << m_crate << std::endl;
  //if (heid.htrSlot()!=m_slot) std::cout << "slot:  " << heid.htrSlot() << "; " << m_slot << std::endl;
  //if (heid.htrTopBottom()!=m_tb) std::cout << "tb:    " << heid.htrTopBottom() << "; " << m_tb << std::endl;
  //if (heid.htrChanId()!=m_chan) std::cout << "chan:  " << heid.htrChanId() << "; " << m_chan << std::endl;
  //
  //if (heid.readoutVMECrateId()==14 && heid.htrTopBottom()==0 && heid.htrChanId()==3) {
  //  std::cout << heid.rawId() << " " << heid << std::endl;
  //}

  try {
    const HcalDetId hdid=emap->lookup(heid);

    int etaabs=hdid.ietaAbs();
    int phi=hdid.iphi();
    int side=hdid.zside();
    int depth=hdid.depth();
    int subdet=hdid.subdet();

    ///////////////////////////////
    if (pattern_number==1) {
      if (m_crate==2 || m_crate==9 || m_crate==12) return;
      //fill only one channel per half-crate
      if (m_slot!=2 && m_slot!=13) return;
      if (m_tb!=0) return;
      if (m_chan!=5) return;
  
      //put some data here
      fCCoded[31]=17;
    }
    ///////////////////////////////
    if (pattern_number==2) {
      if (depth>1) return;
      if (subdet==4 && etaabs<30) return;
      
      if ((etaabs+2)%4!=0) return;
      int i=(etaabs+2)/4;
      if (i<1 || i>7) return;
      
      if ((phi+3)%4!=0) return;
      int j=(phi+3)/4;
      if (j<1 || j>18) return;
      
      //add one to BX?
      if (side<0) fCCoded[phi+10]=i;
      if (side>0) fCCoded[phi+10]=i+16;
    }
    ///////////////////////////////
    if (pattern_number==3) {
      if (depth>1 || etaabs!=18 || side!=1 || phi!=32) return;
      std::cout << "hello" << std::endl;
      //add one to BX?
      if (side>0) fCCoded[15]=20;
    }
    
  }
  catch (...) {return;}
}

void ChannelPattern::Fill(HtrXmlPatternToolParameters* params,HBHEDigiCollection::const_iterator data) {
  //first fill with pedestal value (currently hard-coded to 0)
  for (int samples=0; samples<params->m_samples_per_event; samples++) {
    int index = m_sample_pos + samples;
    if (index>=SAMPLES) continue;

    fCCoded    [index] = 0;
    fCQuantized[index] = 0;
  }

  //now fill with actual samples
  for (int samples=0; samples<data->size(); samples++) {
    int index = m_sample_pos + params->m_presamples_per_event - data->presamples() + samples;
    if (index<m_sample_pos || 
	index>=(m_sample_pos+params->m_samples_per_event) ||
	index>=SAMPLES) continue;

    fCCoded    [index] = data->sample(samples).adc();
    fCQuantized[index] = data->sample(samples).nominal_fC();
  }
  
  m_sample_pos+=params->m_samples_per_event;
}

void ChannelPattern::Fill(HtrXmlPatternToolParameters* params,HFDigiCollection::const_iterator data) {
  //first fill with pedestal value (currently hard-coded to 0)
  for (int samples=0; samples<params->m_samples_per_event; samples++) {
    int index = m_sample_pos + samples;
    if (index>=SAMPLES) continue;

    fCCoded    [index] = 0;
    fCQuantized[index] = 0;
  }

  //now fill with actual samples
  for (int samples=0; samples<data->size(); samples++) {
    int index = m_sample_pos + params->m_presamples_per_event - data->presamples() + samples;
    if (index<m_sample_pos || 
	index>=(m_sample_pos+params->m_samples_per_event) ||
	index>=SAMPLES) continue;

    fCCoded    [index] = data->sample(samples).adc();
    fCQuantized[index] = data->sample(samples).nominal_fC();
  }
  
  m_sample_pos+=params->m_samples_per_event;
}

void ChannelPattern::Fill(HtrXmlPatternToolParameters* params,HODigiCollection::const_iterator data) {
  //first fill with pedestal value (currently hard-coded to 0)
  for (int samples=0; samples<params->m_samples_per_event; samples++) {
    int index = m_sample_pos + samples;
    if (index>=SAMPLES) continue;

    fCCoded    [index] = 0;
    fCQuantized[index] = 0;
  }

  //now fill with actual samples
  for (int samples=0; samples<data->size(); samples++) {
    int index = m_sample_pos + params->m_presamples_per_event - data->presamples() + samples;
    if (index<m_sample_pos || 
	index>=(m_sample_pos+params->m_samples_per_event) ||
	index>=SAMPLES) continue;

    fCCoded    [index] = data->sample(samples).adc();
    fCQuantized[index] = data->sample(samples).nominal_fC();
  }
  
  m_sample_pos+=params->m_samples_per_event;
}

HalfHtrData::HalfHtrData(int crate, int slot, int tb) {
  for (int i=0; i<24; i++)
    m_patterns[i].setLoc(crate,slot,tb,i+1);
  m_crate=crate;
  m_slot=slot;
  m_tb=tb;
  //these are set later with map data
  m_dcc=0;
  m_spigot=0;
}

CrateData::CrateData(int crate, int slotsActive[ChannelPattern::NUM_SLOTS]) {
  for (int slot=0; slot<ChannelPattern::NUM_SLOTS; slot++) {
    for (int tb=0;tb<2;tb++) {
      if (slotsActive[slot]) m_slotsDoubled[slot][tb] = new HalfHtrData(crate,slot,tb);
      else                   m_slotsDoubled[slot][tb] = 0;
    }
  }
}

CrateData::~CrateData() {
  for (int slot=0; slot<ChannelPattern::NUM_SLOTS; slot++) {
    for (int tb=0;tb<2;tb++) {
      if (m_slotsDoubled[slot][tb]) delete m_slotsDoubled[slot][tb];
    }
  }
}

HalfHtrData* CrateData::getHalfHtrData(int slot, int tb) {
  if ( slot>=0 && slot<ChannelPattern::NUM_SLOTS && (tb==0 || tb==1) ) return m_slotsDoubled[slot][tb];
  else return 0;
}

HtrXmlPatternSet::HtrXmlPatternSet(int cratesActive[ChannelPattern::NUM_CRATES], int slotsActive[ChannelPattern::NUM_SLOTS]) {
  for (int crate=0; crate<ChannelPattern::NUM_CRATES; crate++) {
    if (cratesActive[crate]) m_crates[crate] = new CrateData(crate,slotsActive);
    else                     m_crates[crate] = 0;
  }
}

HtrXmlPatternSet::~HtrXmlPatternSet() {
  for (int crate=0; crate<ChannelPattern::NUM_CRATES; crate++) {
    if (m_crates[crate]) delete m_crates[crate];
  }
}

CrateData* HtrXmlPatternSet::getCrate(int crate) {
  if (crate>=0 && crate<ChannelPattern::NUM_CRATES) return m_crates[crate];
  else return 0;
}
