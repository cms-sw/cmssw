//
// Integer representation of floating point arithmetic suitable for FPGA designs
//
// Author: Yuri Gershtein
// Date:   March 2018
//

#include "../interface/imath.h"

std::string var_base::itos(int i)
{
    std::ostringstream os;
    os << i;
    return os.str();
}

std::string var_base::get_kstring()
{

  char s[1024];
  std::string t="";
  std::map<std::string,int>::iterator it;
  for(it = Kmap_.begin(); it != Kmap_.end(); ++it){
    sprintf(s,"^(%i)",it->second);
    std::string t0(s);
    t = t + it->first + t0;
  }

  return t;
}

void var_base::analyze()
{
  if(!readytoanalyze_) return;

  double u = maxval_;
  if(u < -minval_)
      u = -minval_;

  int iu = log2(get_range()/u);
  if(iu>1){
    printf("analyzing %s: range %g is much larger then %g. suggest cutting by a factor of 2^%i\n",name_.c_str(),get_range(),u,iu);
  }
#ifdef IMATH_ROOT
  if(h_){
    double eff = h_->Integral()/h_->GetEntries();
    if(eff<0.99) {
      printf("analyzing %s: range is too small, contains %f\n",name_.c_str(),eff);
      h_->Print();
    }
    h_file_->cd();
    TCanvas *c = new TCanvas();
    c->cd();
    h_->Draw("colz");
    h_->Write();
  }
  else{
    if(use_root) printf("analyzing %s: no histogram!\n",name_.c_str());
  }
#endif

  if(p1_) p1_->analyze();
  if(p2_) p2_->analyze();

  readytoanalyze_ = false;
}

std::string var_base::dump()
{
  char s[1024];
  std::string u = get_kstring();
  sprintf(s,"Name = %s \t Op = %s \t nbits = %i \n       ival = %li \t fval = %g \t K = %g Range = %f\n       units = %s\n",
          name_.c_str(), op_.c_str(), nbits_, ival_, fval_, K_, get_range(), u.c_str());
  std::string t(s);
  return t;
}

void var_base::dump_cout()
{
  char s[2048];
  std::string u = get_kstring();
  sprintf(s,"Name = %s \t Op = %s \t nbits = %i \n       ival = %li \t fval = %g \t K = %g Range = %f\n       units = %s\n       step = %i, latency = %i\n", name_.c_str(), op_.c_str(), nbits_, ival_, fval_, K_, get_range(), u.c_str(), step_, latency_);
  std::string t(s);
  std::cout<<t;
  if(p1_) p1_->dump_cout();
  if(p2_) p2_->dump_cout();
}

void var_adjustK::adjust(double Knew, double epsilon, bool do_assert, int nbits)
{
  //WARNING!!!
  //THIS METHID CAN BE USED ONLY FOR THE FINAL ANSWER
  //THE CHANGE IN CONSTANT CAN NOT BE PROPAGATED UP THE CALCULATION TREE

    K_     = p1_->get_K();
    Kmap_  = p1_->get_Kmap();
    double r = Knew / K_;

    lr_ = (r>1)? log2(r)+epsilon : log2(r);
    K_ = K_ * pow(2,lr_);
    if(do_assert) assert(fabs(Knew/K_ - 1)<epsilon);

    if(nbits>0)
      nbits_ = nbits;
    else
      nbits_ = p1_->get_nbits()-lr_;

    Kmap_["2"] = Kmap_["2"] + lr_;

}

void var_inv::initLUT(double offset)
{
  offset_ = offset;
  double offsetI = round_int(offset_ / p1_->get_K());
  for(int i=0; i<Nelements_; ++i){
    int i1 = addr_to_ival(i);
    LUT[i] = gen_inv(offsetI+i1);
  }
}

void var_base::makeready()
{
  pipe_counter_ = 0;
  pipe_delays_.clear();
  readytoprint_   = true;
  readytoanalyze_ = true;
  usedasinput_    = false;
  if(p1_) p1_->makeready();
  if(p2_) p2_->makeready();
  if(p3_) p3_->makeready();
}

bool var_base::has_delay(int i)
{
  //dumb sequential search
  for(unsigned int j=0; j<pipe_delays_.size(); ++j)
    if(pipe_delays_[j] == i) return true;
  return false;
}

std::string var_base::pipe_delay(var_base *v, int nbits, int delay)
{
  //have we been delayed by this much already?
  if(v->has_delay(delay)) return "";
  v->add_delay(delay);
  std::string name = v->get_name();
  std::string name_delayed = name+"_delay"+itos(delay);
  std::string out = "wire signed ["+itos(nbits-1)+":0] "+name_delayed+";\n";
  out = out + pipe_delay_wire(v, name_delayed, nbits, delay);
  return out;
}
std::string var_base::pipe_delays(const int step)
{
  std::string out = "";
  if (p1_) out += p1_->pipe_delays (step);
  if (p2_) out += p2_->pipe_delays (step);
  if (p3_) out += p3_->pipe_delays (step);

  int l = step - latency_ - step_;
  return (out + pipe_delay(this,get_nbits(),l));
}
std::string var_base::pipe_delay_wire(var_base *v, std::string name_delayed, int nbits, int delay)
{
  std::string name = v->get_name();
  std::string name_pipe    = name+"_pipe"+itos(v->pipe_counter());
  v->pipe_increment();
  std::string out = "pipe_delay #(.STAGES("+itos(delay)+"), .WIDTH("+itos(nbits)+")) "
    + name_pipe + "(.clk(clk), .val_in("+name+"), .val_out("+name_delayed+"));\n";
  return out;
}

void var_base::get_inputs(std::vector<var_base*> *vd)
{
  if(op_ == "def" && !usedasinput_){
    usedasinput_ = true;
    vd->push_back(this);
  }
  else{
    if(p1_) p1_->get_inputs(vd);
    if(p2_) p2_->get_inputs(vd);
    if(p3_) p3_->get_inputs(vd);
  }
}

#ifdef IMATH_ROOT
TTree* var_base::AddToTree(var_base* v, char *s)
{
  if(h_file_==0){
    h_file_ = new TFile("imath.root","RECREATE");
    printf("recreating file imath.root\n");
  }
  h_file_->cd();
  TTree *tt = (TTree*) h_file_->Get("tt");
  if(tt==0){
    tt = new TTree("tt","");
    printf("creating TTree tt\n");
  }
  std::string si = v->get_name()+"_i";
  std::string sf = v->get_name()+"_f";
  std::string sv = v->get_name();
  if(s!=0){
    std::string prefix(s);
    si = prefix + si;
    sf = prefix + sf;
    sv = prefix + sv;
  }
  if(!tt->GetBranchStatus(si.c_str())){
    tt->Branch(si.c_str(),(Long64_t*) &(v->ival_));
    tt->Branch(sf.c_str(),&(v->fval_));
    tt->Branch(sv.c_str(),&(v->val_));
  }

  if(v->p1_) AddToTree(v->p1_, s);
  if(v->p2_) AddToTree(v->p2_, s);
  if(v->p3_) AddToTree(v->p3_, s);

  return tt;
}
TTree* var_base::AddToTree(double* v, char *s)
{
  if(h_file_==0){
    h_file_ = new TFile("imath.root","RECREATE");
    printf("recreating file imath.root\n");
  }
  h_file_->cd();
  TTree *tt = (TTree*) h_file_->Get("tt");
  if(tt==0){
    tt = new TTree("tt","");
    printf("creating TTree tt\n");
  }
  tt->Branch(s,v);
  return tt;
}
TTree* var_base::AddToTree(int* v, char *s)
{
  if(h_file_==0){
    h_file_ = new TFile("imath.root","RECREATE");
    printf("recreating file imath.root\n");
  }
  h_file_->cd();
  TTree *tt = (TTree*) h_file_->Get("tt");
  if(tt==0){
    tt = new TTree("tt","");
    printf("creating TTree tt\n");
  }
  tt->Branch(s,v);
  return tt;
}
void var_base::FillTree()
{
  if(h_file_==0) return;
  h_file_->cd();
  TTree *tt = (TTree*) h_file_->Get("tt");
  if(tt==0) return;
  tt->Fill();
}
void var_base::WriteTree()
{
  if(h_file_==0) return;
  h_file_->cd();
  TTree *tt = (TTree*) h_file_->Get("tt");
  if(tt==0) return;
  tt->Write();
}

#endif

void var_cut::local_passes(std::map<const var_base *,std::vector<bool> > &passes, const std::map<const var_base *,std::vector<bool> > * const previous_passes) const {
  const int lower_cut = lower_cut_/cut_var_->get_K();
  const int upper_cut = upper_cut_/cut_var_->get_K();
  if (!previous_passes || (previous_passes && !previous_passes->count(cut_var_))){
    if (!passes.count(cut_var_)) passes[cut_var_];
    passes.at(cut_var_).push_back(cut_var_->get_ival() > lower_cut && cut_var_->get_ival() < upper_cut);
  }
}

bool var_base::local_passes() const {
  bool passes = false;
  for (const auto &cut : cuts_){
    const var_cut * const cast_cut = (var_cut *) cut;
    const int lower_cut = cast_cut->get_lower_cut()/K_;
    const int upper_cut = cast_cut->get_upper_cut()/K_;
    passes = passes || (ival_ > lower_cut && ival_ < upper_cut);
    printCutInfo_ && std::cout << "  " << name_ << " " << ((ival_ > lower_cut && ival_ < upper_cut) ? "PASSES" : "FAILS") << " (required: " << lower_cut*K_ << " < " << ival_*K_ << " < " << upper_cut*K_ << ")" << std::endl;
  }
  return passes;
}

void var_base::passes(std::map<const var_base *,std::vector<bool> > &passes, const std::map<const var_base *,std::vector<bool> > * const previous_passes) const {
  if (p1_) p1_->passes(passes,previous_passes);
  if (p2_) p2_->passes(passes,previous_passes);
  if (p3_) p3_->passes(passes,previous_passes);

  for (const auto &cut : cuts_){
    const var_cut * const cast_cut = (var_cut *) cut;
    const int lower_cut = cast_cut->get_lower_cut()/K_;
    const int upper_cut = cast_cut->get_upper_cut()/K_;
    if (!previous_passes || (previous_passes && !previous_passes->count(this))){
      if (!passes.count(this)) passes[this];
      passes.at(this).push_back(ival_ > lower_cut && ival_ < upper_cut);
      printCutInfo_ && std::cout << "  " << name_ << " " << ((ival_ > lower_cut && ival_ < upper_cut) ? "PASSES" : "FAILS") << " (required: " << lower_cut*K_ << " < " << ival_*K_ << " < " << upper_cut*K_ << ")" << std::endl;
    }
  }
}

void var_base::add_cut(var_cut *cut, const bool call_set_cut_var){
  cuts_.push_back(cut);
  if (call_set_cut_var) cut->set_cut_var(this,false);
}

void var_cut::set_cut_var(var_base *cut_var, const bool call_add_cut){
  cut_var_ = cut_var;
  if (call_add_cut) cut_var->add_cut(this,false);
  if (parent_flag_) parent_flag_->calculate_step();
}

void var_flag::add_cut(var_base *cut, const bool call_set_parent_flag){
  cuts_.push_back(cut);
  if (cut->get_op() == "cut" && call_set_parent_flag){
    var_cut * const cast_cut = (var_cut *) cut;
    cast_cut->set_parent_flag(this,false);
  }
  calculate_step();
}

void var_cut::set_parent_flag(var_flag *parent_flag, const bool call_add_cut){
  parent_flag_ = parent_flag;
  if (call_add_cut) parent_flag->add_cut(this,false);
}

var_base * var_base::get_cut_var() {
  if (op_ == "cut")
    return cut_var_;
  else
    return this;
}

bool var_flag::passes(){
  printCutInfo_ && std::cout << "Checking if " << name_ << " passes..." << std::endl;
  std::map<const var_base *,std::vector<bool> > passes0, passes1;
  for (const auto &cut : cuts_){
    if (cut->get_op() != "cut") continue;
    const var_cut * const cast_cut = (var_cut *) cut;
    cast_cut->local_passes(passes0);
  }
  for (const auto &cut : cuts_){
    if (cut->get_op() != "cut")
      cut->passes(passes1,&passes0);
    else{
      if (cut->get_cut_var()->get_p1()) cut->get_cut_var()->get_p1()->passes(passes1,&passes0);
      if (cut->get_cut_var()->get_p2()) cut->get_cut_var()->get_p2()->passes(passes1,&passes0);
      if (cut->get_cut_var()->get_p3()) cut->get_cut_var()->get_p3()->passes(passes1,&passes0);
    }
  }

  bool passes = true;
  for (const auto &cut_var : passes0){
    bool local_passes = false;
    for (const auto &pass : cut_var.second)
      local_passes = local_passes || pass;
    passes = passes && local_passes;
  }
  for (const auto &cut_var : passes1){
    bool local_passes = false;
    for (const auto &pass : cut_var.second)
      local_passes = local_passes || pass;
    passes = passes && local_passes;
  }
  printCutInfo_ && std::cout << name_ << " " << (passes ? "PASSES" : "FAILS") << std::endl;

  return passes;
}
