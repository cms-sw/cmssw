
#include "DQM/SiStripCommissioningSummary/interface/SiStripSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//------------------------------------------------------------------------------

SiStripSummary::SiStripSummary(sistrip::View view) :

  summary_(0),
  histogram_(0),
  generator_(0),
  view_(view)

{
  summary_ = new TH1F();
  histogram_ = new TH1F();
  
  if (view == sistrip::CONTROL) {
    generator_ = new SiStripControlSummaryGenerator();
  }
  else if (view == sistrip::READOUT) {
    generator_ = new SiStripReadoutSummaryGenerator();
  }
  else {edm::LogError("DQM") << "[SiStripSummary::SiStripSummary]" 
			     << " Unexpected view! Not " 
			     << sistrip::controlView_ 
			     << " or " 
			     << sistrip::readoutView_;}
  
}

//------------------------------------------------------------------------------

SiStripSummary::~SiStripSummary() {

  if (summary_) delete summary_;
  if (histogram_) delete histogram_;
}

//------------------------------------------------------------------------------

void SiStripSummary::update(unsigned int key, float value, float error) {

  generator_->update(key,value,error);
}

//------------------------------------------------------------------------------

void SiStripSummary::histogram(const string& dir, const string& option) {

  generator_->summary(summary_,histogram_,dir,option);
  format();
 }

//------------------------------------------------------------------------------

void SiStripSummary::setName(const char* name_and_title) {

  stringstream ss;
  ss << name_and_title 
     << sistrip::sep_
     << SiStripHistoNamingScheme::view(view_);

  summary_->SetName(ss.str().c_str());
  histogram_->SetName(name_and_title);

}

//------------------------------------------------------------------------------

void SiStripSummary::setTitle(const char* name_and_title) {

  stringstream ss;
  ss << name_and_title 
     << sistrip::sep_
     << SiStripHistoNamingScheme::view(view_);

  summary_->SetTitle(ss.str().c_str());
  histogram_->SetTitle(name_and_title);
}
