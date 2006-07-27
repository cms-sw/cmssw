#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"

#include <cmath>

SummaryGenerator::SummaryGenerator() :


  //initialise private data members

  map_(),
  max_val_(0),
  max_val_err_(0),
  min_val_(0),
  min_val_err_(0)

{;}

//------------------------------------------------------------------------------

SummaryGenerator::~SummaryGenerator() 
{;}

//------------------------------------------------------------------------------

void SummaryGenerator::update(unsigned int key, float comm_val, float comm_val_error) {

  //find range for histograms
  if ((comm_val > max_val_)  || map_.empty()) max_val_ = comm_val;
  if ((comm_val_error > max_val_err_)  || map_.empty()) max_val_err_ = comm_val_error;
  if ((comm_val < min_val_) || map_.empty()) {min_val_ = comm_val;}
  if ((comm_val_error < min_val_err_) || map_.empty()) {min_val_err_ = comm_val_error;}

  //fill map
  map_[key].first = comm_val;
  map_[key].second = comm_val_error;
}


