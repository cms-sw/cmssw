#include "HtrXmlPatternTool.h"
#include "HtrXmlPatternToolParameters.h"
#include "HtrXmlPatternSet.h"
#include "HtrXmlPatternWriter.h"
#include "TFile.h"
#include "TH1.h"
#include <iostream>
#include <fstream>
#include "boost/filesystem/operations.hpp"

HtrXmlPatternTool::HtrXmlPatternTool(HtrXmlPatternToolParameters* params) {
  m_params = params;

  //slotsOn array index corresponds to physical number (e.g., slotsOn[1] for slot 1)
  int slotsOn [ChannelPattern::NUM_SLOTS];
  int cratesOn[ChannelPattern::NUM_CRATES];

  //turn off all slots
  for (int i=0; i<ChannelPattern::NUM_SLOTS; i++) slotsOn[i]=0;
  //turn on slots 2,3,4,5,6,7
  for (int i=2; i<8; i++)   slotsOn[i]=1;
  //turn on slots 13,14,15,16,17,18
  for (int i=13; i<19; i++) slotsOn[i]=1;

  //turn on all crates
  for (int i=0; i<ChannelPattern::NUM_CRATES; i++) cratesOn[i]=1;
  //turn off two unused crates
  cratesOn[ 8] = 0;
  cratesOn[16] = 0;

  m_patternSet=new HtrXmlPatternSet(cratesOn,slotsOn);
  m_xmlWriter.setTagName(m_params->m_file_tag);

}

HtrXmlPatternTool::~HtrXmlPatternTool() {
  delete m_patternSet;
}

void HtrXmlPatternTool::Fill(const HcalElectronicsId HEID,HBHEDigiCollection::const_iterator data) {
  CrateData* cd=m_patternSet->getCrate(HEID.readoutVMECrateId());
  HalfHtrData* hd=0;
  ChannelPattern* cp=0;

  if (cd) {
    hd=cd->getHalfHtrData(HEID.htrSlot(),HEID.htrTopBottom());
    if (hd) {
      hd->setSpigot(HEID.spigot());
      hd->setDCC(HEID.dccid());

      cp=hd->getPattern(HEID.htrChanId());
      if (cp) cp->Fill(m_params,data);
      else if(m_params->m_show_errors) std::cerr << "Bad (crate,slot,channel): (" 
						 << HEID.readoutVMECrateId() << "," 
						 << HEID.htrSlot()           << "," 
						 << HEID.htrChanId()         << ")" << std::endl;
    }
    else if(m_params->m_show_errors) std::cerr << "Bad (crate,slot): (" 
					       << HEID.readoutVMECrateId() << "," 
					       << HEID.htrSlot()           << ")" << std::endl;
  }
  else if(m_params->m_show_errors) std::cerr << "Bad (crate): (" << HEID.readoutVMECrateId() << ")" << std::endl;
}

void HtrXmlPatternTool::Fill(const HcalElectronicsId HEID,HFDigiCollection::const_iterator data) {
  CrateData* cd=m_patternSet->getCrate(HEID.readoutVMECrateId());
  HalfHtrData* hd=0;
  ChannelPattern* cp=0;

  if (cd) {
    hd=cd->getHalfHtrData(HEID.htrSlot(),HEID.htrTopBottom());
    if (hd) {
      hd->setSpigot(HEID.spigot());
      hd->setDCC(HEID.dccid());

      cp=hd->getPattern(HEID.htrChanId());
      if (cp) cp->Fill(m_params,data);
      else if(m_params->m_show_errors) std::cerr << "Bad (crate,slot,channel): (" 
						 << HEID.readoutVMECrateId() << "," 
						 << HEID.htrSlot()           << "," 
						 << HEID.htrChanId()         << ")" << std::endl;
    }
    else if(m_params->m_show_errors) std::cerr << "Bad (crate,slot): (" 
					       << HEID.readoutVMECrateId() << "," 
					       << HEID.htrSlot()           << ")" << std::endl;
  }
  else if(m_params->m_show_errors) std::cerr << "Bad (crate): (" << HEID.readoutVMECrateId() << ")" << std::endl;
}

void HtrXmlPatternTool::Fill(const HcalElectronicsId HEID,HODigiCollection::const_iterator data) {
  CrateData* cd=m_patternSet->getCrate(HEID.readoutVMECrateId());
  HalfHtrData* hd=0;
  ChannelPattern* cp=0;

  if (cd) {
    hd=cd->getHalfHtrData(HEID.htrSlot(),HEID.htrTopBottom());
    if (hd) {
      hd->setSpigot(HEID.spigot());
      hd->setDCC(HEID.dccid());

      cp=hd->getPattern(HEID.htrChanId());
      if (cp) cp->Fill(m_params,data);
      else if(m_params->m_show_errors) std::cerr << "Bad (crate,slot,channel): (" 
						 << HEID.readoutVMECrateId() << "," 
						 << HEID.htrSlot()           << "," 
						 << HEID.htrChanId()         << ")" << std::endl;
    }
    else if(m_params->m_show_errors) std::cerr << "Bad (crate,slot): (" 
					       << HEID.readoutVMECrateId() << "," 
					       << HEID.htrSlot()           << ")" << std::endl;
  }
  else if(m_params->m_show_errors) std::cerr << "Bad (crate): (" << HEID.readoutVMECrateId() << ")" << std::endl;
}

void HtrXmlPatternTool::prepareDirs() {
  boost::filesystem::create_directory(m_params->m_output_directory);
}

void HtrXmlPatternTool::writeXML() {
  std::cout << "Writing XML..." << std::endl;
  bool singleMode=false;
  std::ofstream* of=0;

  if (m_params->m_single_XML_file) {
    singleMode=true;
    std::string name=m_params->m_output_directory+(m_params->m_file_tag)+"-all.xml";
    of=new std::ofstream(name.c_str(),std::ios_base::out|std::ios_base::trunc);
    if (!of->good()) {
      std::cerr << "XML output file " << name << " is bad." << std::endl;
      return;
    }
    (*of) << "<?xml version='1.0' encoding='UTF-8'?>" << std::endl;
    (*of) << "<CFGBrickSet name='" << m_params->m_file_tag << "'>" << std::endl;
  }
  for (int crate=0; crate<ChannelPattern::NUM_CRATES; crate++) {
    CrateData* cd=m_patternSet->getCrate(crate);
    if (cd==0) continue;
    for (int slot=0; slot<ChannelPattern::NUM_SLOTS; slot++) {
      for (int tb=0; tb<=1; tb++) {
	HalfHtrData* hd=cd->getHalfHtrData(slot,tb);
	if (hd==0) continue;
       
	for (int fiber=1; fiber<=8; fiber++) {

	  if (!singleMode) {
	    char fname[1024];
	    snprintf(fname,1024,"%s/%s_%d_%d_%d_%d.xml",m_params->m_output_directory.c_str(),
		     m_params->m_file_tag.c_str(),
		     hd->getCrate(),hd->getSlot(),hd->getTB(),fiber);
	    of=new std::ofstream(fname,std::ios_base::out|std::ios_base::trunc);
	    if (!of->good()) {
	      std::cerr << "XML output file " << fname << " is bad." << std::endl;
	      return;
	    }
	    (*of) << "<?xml version='1.0' encoding='UTF-8'?>" << std::endl;
	  }

	  m_xmlWriter.writePattern(hd,fiber,*of,(singleMode)?(1):(0));

	  if (!singleMode) {
	    of->close();
	    delete of; of=0;
	  }
	}
      }
    }
  }
  if (singleMode) {
    (*of) << "</CFGBrickSet>" << std::endl;
    of->close();
    delete of;
    of=0;
  }
}

void HtrXmlPatternTool::createHists() {
  std::cout << "Writing root file..." << std::endl;
  std::string name=m_params->m_output_directory+"/"+(m_params->m_file_tag)+".root";
  TFile of(name.c_str(),"RECREATE");
  if (of.IsZombie()) {
    std::cerr << "root output file " << name << " is bad." << std::endl;
    return;
  }

  //TDirectory* dp=new TDirectory("perfect","perfect");
  //TDirectory* dq=new TDirectory("quantized","quantized");
  TDirectory* da=new TDirectory("adc","adc");

  //dp->Write(); delete dp;
  //dq->Write(); delete dq;
  da->Write(); delete da;

  for (int crate=0; crate<ChannelPattern::NUM_CRATES; crate++) {
    CrateData* cd=m_patternSet->getCrate(crate);
    if (cd==0) continue;
    for (int slot=0; slot<ChannelPattern::NUM_SLOTS; slot++) {
      for (int tb=0; tb<=1; tb++) {
	HalfHtrData* hd=cd->getHalfHtrData(slot,tb);
	if (hd==0) continue;
	for (int chan=1; chan<=24; chan++) {
	  ChannelPattern* cp=hd->getPattern(chan);
	  char hname[128];
	  sprintf(hname,"Exact fC Cr%d,%d%s-%d",crate,slot,
		  ((tb==1)?("t"):("b")),chan);
	  TH1* hp=new TH1F(hname,hname,ChannelPattern::SAMPLES,-0.5,ChannelPattern::SAMPLES-0.5);
	  hp->SetDirectory(0);
	  sprintf(hname,"Quantized fC Cr%d,%d%s-%d",crate,slot,
		  ((tb==1)?("t"):("b")),chan);
	  TH1* hq=new TH1F(hname,hname,ChannelPattern::SAMPLES,-0.5,ChannelPattern::SAMPLES-0.5);
	  hp->SetDirectory(0);
	  sprintf(hname,"Encoded fC Cr%d,%d%s-%d",crate,slot,
		  ((tb==1)?("t"):("b")),chan);
	  TH1* ha=new TH1F(hname,hname,ChannelPattern::SAMPLES,-0.5,ChannelPattern::SAMPLES-0.5);
	  ha->SetDirectory(0);
	  for (int i=0; i<ChannelPattern::SAMPLES; i++) {
	    hp->Fill(i*1.0,(*cp)[i]);
	    hq->Fill(i*1.0,cp->getQuantized(i));
	    ha->Fill(i*1.0,cp->getCoded(i));
	  }
	  //of.cd("perfect");   hp->Write();
	  //of.cd("quantized"); hq->Write();
	  of.cd("adc");       ha->Write();
	  delete hp;
	  delete hq;
	  delete ha;
	}
      }
    }
  }
  of.Close();
}
