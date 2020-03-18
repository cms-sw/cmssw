#include "../interface/imath.h"

bool var_base::calculate(int debug_level)
{
  bool ok1 = true;
  bool ok2 = true;
  bool ok3 = true;

  if(p1_) ok1 = p1_->calculate(debug_level);
  if(p2_) ok2 = p2_->calculate(debug_level);
  if(p3_) ok3 = p3_->calculate(debug_level);

  long int ival_prev = ival_;
  local_calculate();

  bool all_ok = ok1 && ok2 && ok3 && debug_level;

  if(fval_ > maxval_) maxval_ = fval_;
  if(fval_ < minval_) minval_ = fval_;
#ifdef IMATH_ROOT
  if(use_root){
    if(h_==0){
      h_file_->cd();
      std::string hname = "h_"+name_;
      h_ = (TH2F*) h_file_->Get(hname.c_str());
      if(h_ == 0){
        h_precision_ = 0.5*h_nbins_*K_;
        std::string st = name_+";fval;fval-ival*K";
        h_ = new TH2F(hname.c_str(),name_.c_str(),
                      h_nbins_,-get_range(), get_range(),
                      h_nbins_,-h_precision_, h_precision_);
        if(debug_level==3) std::cout<<" booking histogram "<<hname<<"\n";
      }
    }
    if(ival_ != ival_prev || op_=="def" || op_=="const") h_->Fill(fval_, K_*ival_-fval_);
  }
#endif

  bool todump = false;
  int nmax = sizeof(long int)*8;
  int ns = nmax - nbits_;
  long int itest = ival_;
  itest = itest<<ns;
  itest = itest>>ns;
  if(itest!=ival_){
    if(debug_level == 3 || (ival_!=ival_prev && all_ok)){
      std::cout<<"imath: truncated value mismatch!! "<<ival_<<" != "<<itest<<"\n";
      todump = true;
    }
    all_ok = false;
  }

  val_ = ival_ * K_;
  float ftest = val_ ;
  float tolerance = 0.1 * fabs(fval_);
  if(tolerance < 2 * K_) tolerance = 2 * K_;
  if(fabs(ftest-fval_)> tolerance){
    if( debug_level == 3 || (ival_!=ival_prev &&(all_ok && (op_!="inv" ||debug_level>=2 )))){
      std::cout<<"imath: **GROSS** value mismatch!! "<<fval_<<" != "<<ftest<<"\n";
      if(op_=="inv") std::cout<<p1_->dump()<<"\n-----------------------------------\n";
      todump = true;
    }
    all_ok = false;
  }

  if(todump)
    std::cout<<dump();

  return all_ok;
}

void var_flag::calculate_step(){
  int max_step = 0;
  for (const auto &cut : cuts_){
    if (!cut->get_cut_var()) continue;
    if (cut->get_cut_var()->get_latency()+cut->get_cut_var()->get_step() > max_step)
      max_step = cut->get_cut_var()->get_latency()+cut->get_cut_var()->get_step();
  }
  step_ = max_step;
}

//
//  local calculations
//

void var_adjustK::local_calculate()
{
  fval_ = p1_->get_fval();
  ival_ = p1_->get_ival();
  if(lr_>0)
    ival_ = ival_ >> lr_;
  else if(lr_<0)
    ival_ = ival_ <<(-lr_);
}

void var_adjustKR::local_calculate()
{
  fval_ = p1_->get_fval();
  ival_ = p1_->get_ival();
  if(lr_>0)
    ival_ = ((ival_ >> (lr_-1))+1)>>1; //rounding
  else if(lr_<0)
    ival_ = ival_ <<(-lr_);
}

void var_add::local_calculate()
{
  fval_ = p1_->get_fval() + p2_->get_fval();
  long int i1 = p1_->get_ival();
  long int i2 = p2_->get_ival();
  if(shift1>0) i1 = i1 << shift1;
  if(shift2>0) i2 = i2 << shift2;
  ival_ = i1 + i2;
  if(ps_>0) ival_ = ival_ >> ps_;
}

void var_subtract::local_calculate()
{
  fval_ = p1_->get_fval() - p2_->get_fval();
  long int i1 = p1_->get_ival();
  long int i2 = p2_->get_ival();
  if(shift1>0) i1 = i1 << shift1;
  if(shift2>0) i2 = i2 << shift2;
  ival_ = i1 - i2;
  if(ps_>0) ival_ = ival_ >> ps_;
}

void var_nounits::local_calculate()
{
  fval_ = p1_->get_fval();
  ival_ = (p1_->get_ival() * cI_)>>ps_;
}

void var_timesC::local_calculate()
{
  fval_ = p1_->get_fval() * cF_;
  ival_ = (p1_->get_ival() * cI_)>>ps_;
}

void var_neg::local_calculate()
{
  fval_ = -p1_->get_fval();
  ival_ = -p1_->get_ival();
}

void var_shift::local_calculate()
{
  fval_ = p1_->get_fval() * pow(2,-shift_);
  ival_ = p1_->get_ival();
  if(shift_>0) ival_ = ival_>>shift_;
  if(shift_<0) ival_ = ival_<<(-shift_);
}

void var_shiftround::local_calculate()
{
  fval_ = p1_->get_fval() * pow(2,-shift_);
  ival_ = p1_->get_ival();
  if(shift_>0) ival_ = ((ival_>>(shift_-1))+1)>>1;
  if(shift_<0) ival_ = ival_<<(-shift_);
}

void var_mult::local_calculate()
{
  fval_ = p1_->get_fval() * p2_->get_fval();
  ival_ = (p1_->get_ival() * p2_->get_ival())>>ps_;
}

void var_DSP_postadd::local_calculate()
{
  fval_ = p1_->get_fval() * p2_->get_fval() + p3_->get_fval();
  ival_ = p3_->get_ival();
  if(shift3_>0) ival_ = ival_<<shift3_;
  if(shift3_<0) ival_ = ival_>>(-shift3_);
  ival_ += p1_->get_ival() * p2_->get_ival();
  ival_ = ival_>>ps_;
}

void var_inv::local_calculate()
{
  fval_ = 1./(offset_ + p1_->get_fval());
  ival_ = LUT[ival_to_addr(p1_->get_ival())];
}
