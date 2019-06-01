#include "DataFormats/L1Trigger/interface/L1DataEmulResult.h"

l1t::L1DataEmulResult::L1DataEmulResult()
    : event_match_(true),
      collname_(""),
      pt_mismatch_(0),
      etaphi_mismatch_(0),
      n_mismatch_(0),
      n_dataonly_(0),
      n_emulonly_(0),
      add1_(0),
      add2_(0) {}

l1t::L1DataEmulResult::L1DataEmulResult(bool event_match, std::string collname)
    : event_match_(event_match),
      collname_(collname),
      pt_mismatch_(0),
      etaphi_mismatch_(0),
      n_mismatch_(0),
      n_dataonly_(0),
      n_emulonly_(0),
      add1_(0),
      add2_(0) {}

l1t::L1DataEmulResult::L1DataEmulResult(bool event_match,
                                        int pt_mismatch,
                                        int etaphi_mismatch,
                                        int n_mismatch,
                                        int n_dataonly,
                                        int n_emulonly,
                                        int add1,
                                        int add2,
                                        std::string collname)
    : event_match_(event_match),
      collname_(collname),
      pt_mismatch_(pt_mismatch),
      etaphi_mismatch_(etaphi_mismatch),
      n_mismatch_(n_mismatch),
      n_dataonly_(n_dataonly),
      n_emulonly_(n_emulonly),
      add1_(add1),
      add2_(add2) {}

l1t::L1DataEmulResult::~L1DataEmulResult() {}

bool l1t::L1DataEmulResult::Event_match() { return event_match_; }

std::string l1t::L1DataEmulResult::Collname() { return collname_; }

int l1t::L1DataEmulResult::PT_mismatch() { return pt_mismatch_; }

int l1t::L1DataEmulResult::ETAPHI_mismatch() { return etaphi_mismatch_; }

int l1t::L1DataEmulResult::N_mismatch() { return n_mismatch_; }

int l1t::L1DataEmulResult::N_dataonly() { return n_dataonly_; }

int l1t::L1DataEmulResult::N_emulonly() { return n_emulonly_; }

int l1t::L1DataEmulResult::Add1() { return add1_; }

int l1t::L1DataEmulResult::Add2() { return add2_; }
