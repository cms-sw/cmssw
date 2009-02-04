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
