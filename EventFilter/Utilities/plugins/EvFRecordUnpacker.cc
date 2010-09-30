#include "EvFRecordUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/FEDInterface/interface/FED1023.h"
#include "DataFormats/Provenance/interface/Timestamp.h" 

#include <iostream>

namespace evf{

  EvFRecordUnpacker::EvFRecordUnpacker( const edm::ParameterSet& pset)
    : label_(pset.getParameter<edm::InputTag>("inputTag"))
    , node_usage_(0)
    , l1_rb_delay_(0)
  {
    node_usage_ = new TH1F("nodes","node usage",3000,0.,3000.);
    l1_rb_delay_ = new TH1F("l1-rb","l1-rb delay in ms",1000,0.,1000.);
    corre_ = new TH2F("corre","Correlation",3000,0.,3000.,1000,0.,1000.);  
    f_ = new TFile("histos.root","RECREATE");
  }
  EvFRecordUnpacker::~EvFRecordUnpacker()
  {
    f_->cd();
    node_usage_->Write();
    l1_rb_delay_->Write();
    corre_->Write();
    f_->Write();
    f_->Close();
    delete f_;
    delete corre_;
    if(node_usage_ !=0) delete node_usage_;
    if(l1_rb_delay_!=0) delete l1_rb_delay_;
  }

  void EvFRecordUnpacker::analyze(const edm::Event & e, const edm::EventSetup& c)
  {
    edm::Timestamp ts = e.eventAuxiliary().time();

    using namespace fedinterface;
    edm::Handle<FEDRawDataCollection> rawdata;
    e.getByLabel(label_,rawdata);
    unsigned int id = fedinterface::EVFFED_ID;
    const FEDRawData& data = rawdata->FEDData(id);
    size_t size=data.size();
    unsigned int rbident = *((unsigned int*)(data.data()+EVFFED_RBIDENT_OFFSET));
    uint64_t rbtime = *((uint64_t*)(data.data()+EVFFED_RBWCTIM_OFFSET));
    unsigned int s = (unsigned int)((rbtime >> 32)-(ts.value() >> 32));
    unsigned int us = (unsigned int)((rbtime & 0xffffffff) - (ts.value() & 0xffffffff));
    float deltams= s*1000.+((float)us)/1000.;

    unsigned int nodeid = (rbident >> EVFFED_RBPCIDE_SHIFT) & EVFFED_RBPCIDE_MASK;
    unsigned int rackid = (nodeid&0xfff00)>>8;
    unsigned int position_in_rack = nodeid&0x000ff;
    std::cout << std::hex << rbident << " node id " << nodeid << " rack " << rackid << " position " << position_in_rack << std::dec << std::endl;
    unsigned int row = ((rackid & 0xf00) >> 8)-10;
    unsigned int rackno = (rackid & 0x0ff)-0x11;
    std::cout << "row " << row << " rackno " << rackno << std::endl;
    std::cout << "BIN " <<  row*9*30+rackno*30+position_in_rack << std::endl;
    position_in_rack = position_in_rack%16 + position_in_rack/16*10;

    float x = row*9*30+rackno*30+position_in_rack;
    std::cout << " X " << x << std::endl;
    node_usage_->Fill(x,1.);
    l1_rb_delay_->Fill(deltams,1.);
    corre_->Fill(x,deltams,1.);
  }
}
