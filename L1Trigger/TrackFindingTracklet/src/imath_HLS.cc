#include "../interface/imath.h"

void var_inv::writeLUT(std:: ofstream& fs, HLS) const
{
  for(int i=0; i<Nelements_-1; ++i)
      fs<<"0x"<<std::hex<<(LUT[i]&((1<<nbits_)-1))<<std::dec<<",\n";
  fs<<"0x"<<std::hex<<(LUT[Nelements_-1]&((1<<nbits_)-1))<<std::dec<<"\n";
}

void var_base::print_truncation(std::string &t, const std::string &o1, const int ps, HLS) const
{
  if (ps > 0) {
    t += "const ap_int<"+itos(nbits_+ps)+"> "+name_+"_tmp = "+o1+";\n";
    t += "const ap_int<"+itos(nbits_)+"> "+name_+" = "+name_+"_tmp >> "+itos(ps)+";\n";
  }
  else
    t += "const ap_int<"+itos(nbits_)+"> "+name_+" = "+o1+";\n";
}

//
// print functions
//

void var_cut::print(std::map<const var_base *,std::set<std::string> > &cut_strings, const int step, HLS, const std::map<const var_base *,std::set<std::string> > * const previous_cut_strings) const {
  assert(step>-1);
  std::string name = cut_var_->get_name();

  const int lower_cut = lower_cut_/cut_var_->get_K();
  const int upper_cut = upper_cut_/cut_var_->get_K();
  if (!previous_cut_strings || (previous_cut_strings && !previous_cut_strings->count(cut_var_))){
    if (!cut_strings.count(cut_var_)) cut_strings[cut_var_];
    cut_strings.at(cut_var_).insert(name+" > "+itos(lower_cut)+" && "+name+" < "+itos(upper_cut));
  }
}

void var_base::print_cuts(std::map<const var_base *,std::set<std::string> > &cut_strings, const int step, HLS, const std::map<const var_base *,std::set<std::string> > * const previous_cut_strings) const {
  if (p1_) p1_->print_cuts(cut_strings,step,hls,previous_cut_strings);
  if (p2_) p2_->print_cuts(cut_strings,step,hls,previous_cut_strings);
  if (p3_) p3_->print_cuts(cut_strings,step,hls,previous_cut_strings);

  std::string name = name_;

  for (const auto &cut : cuts_){
    const var_cut * const cast_cut = (var_cut *) cut;
    const int lower_cut = cast_cut->get_lower_cut()/K_;
    const int upper_cut = cast_cut->get_upper_cut()/K_;
    if (!previous_cut_strings || (previous_cut_strings && !previous_cut_strings->count(this))){
      if (!cut_strings.count(this)) cut_strings[this];
      cut_strings.at(this).insert(name+" > "+itos(lower_cut)+" && "+name+" < "+itos(upper_cut));
    }
  }
}

void var_adjustK::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);

  std::string shift = "";
  if(lr_>0)
    shift = " >> " + itos(lr_);
  else if(lr_<0)
    shift = " << " + itos(-lr_);

  std::string n1 = p1_->get_name();

  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n";
  fs<<"const ap_int<"<<nbits_<<"> "<<name_<<" = "<<n1<<shift<<";\n";

}

void var_adjustKR::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);

  std::string n1 = p1_->get_name();

  std::string  o1 = n1;
  if(lr_==1) o1 = "("+o1+"+1)>>1";
  if(lr_>1)  o1 = "( ("+o1 + ">>" + itos(lr_-1)+")+1)>>1";
  if(lr_<0)  o1 = "ap_int<" + itos(nbits_) + ">(" + o1 + ")<<" + itos(-lr_);

  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n";
  fs<<"const ap_int<"<<nbits_<<"> "<<name_<<" = "<<o1<<";\n";
}

void var_def::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);

  fs<<"// units "<<get_kstring()<<"\t"<<K_<<"\n";
  fs<<"const ap_int<"<<nbits_<<"> "<<name_<<" = "<<name_<<"_wire;\n";
}

void var_param::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);

  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n";
  fs<<"static const ap_int<"<<nbits_<<"> "<<name_<<" = "<<ival_<<";\n";
}

void var_add::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(p2_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string o1 = p1_->get_name();
  if(shift1>0) {
    o1 = "ap_int<"+itos(nbits_+ps_)+">("+o1+")";
    o1 += "<<"+itos(shift1);
    o1 = "("+o1+")";
  }

  std::string o2 = p2_->get_name();
  if(shift2>0) {
    o2 = "ap_int<"+itos(nbits_+ps_)+">("+o2+")";
    o2 += "<<"+itos(shift2);
    o2 = "("+o2+")";
  }

  o1 = o1 + " + " + o2;

  std::string t = "";
  print_truncation(t, o1, ps_, hls);
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t;
}

void var_subtract::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(p2_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string o1 = p1_->get_name();
  if(shift1>0) {
    o1 = "ap_int<"+itos(nbits_+ps_)+">("+o1+")";
    o1 += "<<"+itos(shift1);
    o1 = "("+o1+")";
  }

  std::string o2 = p2_->get_name();
  if(shift2>0) {
    o2 = "ap_int<"+itos(nbits_+ps_)+">("+o2+")";
    o2 += "<<"+itos(shift2);
    o2 = "("+o2+")";
  }

  o1 = o1 + " - " + o2;

  std::string t = "";
  print_truncation(t, o1, ps_, hls);
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t;
}

void var_nounits::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string n1 = p1_->get_name();
  std::string o1 = "(" + n1 + " * " + itos(cI_) + ")";

  std::string t = "";
  print_truncation(t, o1, ps_, hls);
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t;
}

void var_timesC::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string n1 = p1_->get_name();
  std::string o1 = "(" + n1 + " * " + itos(cI_) + ")";

  std::string t = "";
  print_truncation(t, o1, ps_, hls);
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t;
}

void var_neg::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string n1 = p1_->get_name();

  std::string t = "const ap_int<"+itos(nbits_)+"> "+name_+" = -"+n1+";\n";
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t<<";\n";
}

void var_shiftround::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string n1 = p1_->get_name();
  std::string  o1 = n1;
  if(shift_==1) o1 = "("+o1+"+1)>>1";
  if(shift_>1)  o1 = "( ("+o1 + ">>" + itos(shift_-1)+")+1)>>1";
  if(shift_<0)  o1 = "ap_int<" + itos(nbits_) + ">(" + o1 + ")<<" + itos(-shift_);

  std::string t = "const ap_int<"+itos(nbits_)+"> "+name_+" = "+o1+";\n";
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t<<";\n";
}

void var_shift::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  std::string n1 = p1_->get_name();
  std::string o1  = n1;
  if(shift_>0) o1 = o1 + ">>" + itos(shift_);
  if(shift_<0) o1 = "ap_int<" + itos(nbits_) + ">(" + o1 + ")<<" + itos(-shift_);

  std::string t = "const ap_int<"+itos(nbits_)+"> "+name_+" = "+o1+";\n";
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t<<";\n";
}

void var_mult::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);
  assert(p1_);
  std::string n1 = p1_->get_name();
  assert(p2_);
  std::string n2 = p2_->get_name();
  std::string o1 =  n1 + " * " + n2;

  std::string t = "";
  print_truncation(t, o1, ps_, hls);
  fs<<"// "<<nbits_ <<" bits \t "<<get_kstring()<<"\t"<<K_<<"\n"<<t;
}

void var_inv::print(std::ofstream& fs, HLS, int l1, int l2, int l3)
{
  assert(p1_);
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);

  fs<<"static const ap_int<"<<itos(nbits_)<<"> LUT_"<<name_<<"["<<Nelements_<<"] = {\n";
  fs<<"#include \"LUT_"<<name_<<".h\"\n";
  fs<<"};\n";

  std::string n1 = p1_->get_name();
  //first calculate address
  std::string t1 = "addr_" + name_;
  std::string t = "const ap_uint<"+itos(nbaddr_)+"> "+t1+" = ";
  if(shift_>0)
    t = t + "(" + n1 + ">>"+itos(shift_)+") & "+itos(mask_);
  else
    t = t + n1 + " & "+itos(mask_);
  fs<<t<<"; // address for the LUT\n";

  t = "const ap_int<"+itos(nbits_)+"> "+name_+" = LUT_"+name_+"[addr_"+name_+"];\n";
  fs<<t;
}

void var_flag::print(std::ofstream& fs, HLS, int l1, int l2, int l3){
  assert(l1==0);
  assert(l2==0);
  assert(l3==0);

  fs<<"const ap_int<1> "<<name_<<" = (";
  std::map<const var_base *,std::set<std::string> > cut_strings0, cut_strings1;
  for (const auto &cut : cuts_){
    if (cut->get_op() != "cut") continue;
    const var_cut * const cast_cut = (var_cut *) cut;
    cast_cut->print(cut_strings0,step_,hls);
  }
  for (const auto &cut : cuts_){
    if (cut->get_op() != "cut")
      cut->print_cuts(cut_strings1,step_,hls,&cut_strings0);
    else{
      if (cut->get_cut_var()->get_p1()) cut->get_cut_var()->get_p1()->print_cuts(cut_strings1,step_,hls,&cut_strings1);
      if (cut->get_cut_var()->get_p2()) cut->get_cut_var()->get_p2()->print_cuts(cut_strings1,step_,hls,&cut_strings1);
      if (cut->get_cut_var()->get_p3()) cut->get_cut_var()->get_p3()->print_cuts(cut_strings1,step_,hls,&cut_strings1);
    }
  }

  std::string separator = "";
  for (const auto &cut_var : cut_strings0){
    separator += "((";
    for (const auto &cut_string : cut_var.second){
      fs<<separator<<cut_string;
      separator = ") || (";
    }
    separator = ")) && ";
  }
  for (const auto &cut_var : cut_strings1){
    separator += "((";
    for (const auto &cut_string : cut_var.second){
      fs<<separator<<cut_string;
      separator = ") || (";
    }
    separator = ")) && ";
  }

  fs<<")));";
}

void var_base::print_step(int step, std::ofstream& fs, HLS){
  if(!readytoprint_) return;
  if(step > step_) return;
  if(p1_) p1_->print_step(step, fs, hls);
  if(p2_) p2_->print_step(step, fs, hls);
  if(p3_) p3_->print_step(step, fs, hls);
  if(step==step_){
    print(fs, hls, 0, 0, 0);
    readytoprint_ = false;
  }
}

void var_base::print_all(std::ofstream& fs, HLS)
{
  for(int i=0; i<=step_; ++i){
    fs<<"//\n// STEP "<<i<<"\n\n";
    print_step(i,fs, hls);
  }
}

void var_base::Design_print(std::vector<var_base*> v, std::ofstream& fs, HLS)
{

  //header of the module

  //inputs
  std::vector<var_base*> vd;
  vd.clear();
  int imax = v.size();
  for(int i=0; i<imax; ++i)
    (v[i])->get_inputs(&vd);

  //print header
  fs<<"#include \"ap_int.h\"\n\n";
  fs<<"void XXX (\n";

  imax = vd.size();
  for(int i=0; i<imax; ++i)
    fs<<"  const ap_int<"<<(vd[i])->get_nbits()<<"> "<<(vd[i])->get_name()<<"_wire,\n";
  fs<<"\n";

  imax = v.size()-1;
  for(int i=0; i<imax; ++i)
    fs<<"  ap_int<"<<(v[i])->get_nbits()<<"> * const "<<(v[i])->get_name()<<"_wire,\n";
  if(imax>=0)
    fs<<"  ap_int<"<<(v[imax])->get_nbits()<<"> * const "<<(v[imax])->get_name()<<"_wire\n";
  fs<<")\n{\n";
  fs<<"#pragma HLS pipeline II=1\n";
  fs<<"#pragma HLS latency max=25\n";

  //body of the module
  imax = v.size();
  for(int i=0; i<imax; ++i){
    fs<<"\n//\n";
    fs<<"// calculating "<<(v[i])->get_name()<<"\n";
    fs<<"//\n";
    (v[i])->print_all(fs, hls);
  }
  fs<<"\n";

  //trailer
  fs<<"\n";
  fs<<"\n//\n";
  fs<<"// wiring the outputs \n";
  fs<<"//\n";
  for(int i=0; i<imax; ++i){
    std::string n = v[i]->get_name()+"_wire";
    fs<<"*"<<n<<" = "<<(v[i])->get_name()<<";\n";
  }

  fs<<"}\n";

}
