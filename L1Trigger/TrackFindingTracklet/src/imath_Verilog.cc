#include "../interface/imath.h"

void VarInv::writeLUT(std::ofstream& fs, Verilog) const {
  for (int i = 0; i < Nelements_; ++i)
    fs << std::hex << (LUT[i] & ((1 << nbits_) - 1)) << std::dec << "\n";
}

void VarBase::print_truncation(std::string& t, const std::string& o1, const int ps, Verilog) const {
  if (ps > 0) {
    t += "wire signed [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
    t += "reg signed  [" + itos(nbits_ + ps - 1) + ":0]" + name_ + "_tmp;\n";
    t += "always @(posedge clk) " + name_ + "_tmp <= " + o1 + ";\n";
    t += "assign " + name_ + " = " + name_ + "_tmp[" + itos(nbits_ + ps - 1) + ":" + itos(ps) + "];\n";
  } else {
    t += "reg signed  [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
    t += "always @(posedge clk) " + name_ + " <= " + o1 + ";\n";
  }
}

//
// print functions
//

void VarCut::print(std::map<const VarBase*, std::set<std::string> >& cut_strings,
                   const int step,
                   Verilog,
                   const std::map<const VarBase*, std::set<std::string> >* const previous_cut_strings) const {
  int l = step - cut_var_->latency() - cut_var_->step();
  std::string name = cut_var_->name();
  if (l > 0)
    name += "_delay" + itos(l);

  const int lower_cut = lower_cut_ / cut_var_->K();
  const int upper_cut = upper_cut_ / cut_var_->K();
  if (!previous_cut_strings || (previous_cut_strings && !previous_cut_strings->count(cut_var_))) {
    if (!cut_strings.count(cut_var_))
      cut_strings[cut_var_];
    cut_strings.at(cut_var_).insert(name + " > " + itos(lower_cut) + " && " + name + " < " + itos(upper_cut));
  }
}

void VarBase::print_cuts(std::map<const VarBase*, std::set<std::string> >& cut_strings,
                         const int step,
                         Verilog,
                         const std::map<const VarBase*, std::set<std::string> >* const previous_cut_strings) const {
  if (p1_)
    p1_->print_cuts(cut_strings, step, verilog, previous_cut_strings);
  if (p2_)
    p2_->print_cuts(cut_strings, step, verilog, previous_cut_strings);
  if (p3_)
    p3_->print_cuts(cut_strings, step, verilog, previous_cut_strings);

  int l = step - latency_ - step_;
  std::string name = name_;
  if (l > 0)
    name += "_delay" + itos(l);

  for (const auto& cut : cuts_) {
    const VarCut* const cast_cut = (VarCut*)cut;
    const int lower_cut = cast_cut->lower_cut() / K_;
    const int upper_cut = cast_cut->upper_cut() / K_;
    if (!previous_cut_strings || (previous_cut_strings && !previous_cut_strings->count(this))) {
      if (!cut_strings.count(this))
        cut_strings[this];
      cut_strings.at(this).insert(name + " > " + itos(lower_cut) + " && " + name + " < " + itos(upper_cut));
    }
  }
}

void VarAdjustK::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);

  std::string shift = "";
  if (lr_ > 0)
    shift = " >>> " + itos(lr_);
  else if (lr_ < 0)
    shift = " << " + itos(-lr_);

  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);

  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n";
  std::string t = "wire signed [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
  t += "assign " + name_ + " = " + n1 + shift;
  fs << t << "; \n";
}

void VarAdjustKR::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);

  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);

  std::string o1 = n1;
  if (lr_ == 1)
    o1 = "(" + o1 + "+1)>>>1";
  if (lr_ > 1)
    o1 = "( (" + o1 + ">>>" + itos(lr_ - 1) + ")+1)>>>1";
  if (lr_ < 0)
    o1 = o1 + "<<" + itos(-lr_);

  std::string t = "reg signed [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
  t += "always @(posedge clk) " + name_ + " <= " + o1;
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t << ";\n";
}

void VarDef::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(l1 == 0);
  assert(l2 == 0);
  assert(l3 == 0);

  std::string t = "reg signed  [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
  t = t + "always @(posedge clk) " + name_ + " <= " + name_ + "_wire;\n";
  fs << "// units " << kstring() << "\t" << K_ << "\n" << t;
}

void VarParam::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(l1 == 0);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string t = "parameter " + name_ + " = ";
  if (ival_ < 0)
    t = t + "- " + itos(nbits_) + "\'sd" + itos(-ival_);
  else
    t = t + itos(nbits_) + "\'sd" + itos(ival_);
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t << ";\n";
}

void VarAdd::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(p2_);
  assert(l3 == 0);
  std::string o1 = p1_->name();
  if (l1 > 0)
    o1 += "_delay" + itos(l1);
  if (shift1 > 0) {
    o1 += "<<" + itos(shift1);
    o1 = "(" + o1 + ")";
  }

  std::string o2 = p2_->name();
  if (l2 > 0)
    o2 += "_delay" + itos(l2);
  if (shift2 > 0) {
    o2 += "<<" + itos(shift2);
    o2 = "(" + o2 + ")";
  }

  o1 = o1 + " + " + o2;

  std::string t = "";
  print_truncation(t, o1, ps_, verilog);
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t;
}

void VarSubtract::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(p2_);
  assert(l3 == 0);
  std::string o1 = p1_->name();
  if (l1 > 0)
    o1 += "_delay" + itos(l1);
  if (shift1 > 0) {
    o1 += "<<" + itos(shift1);
    o1 = "(" + o1 + ")";
  }

  std::string o2 = p2_->name();
  if (l2 > 0)
    o2 += "_delay" + itos(l2);
  if (shift2 > 0) {
    o2 += "<<" + itos(shift2);
    o2 = "(" + o2 + ")";
  }

  o1 = o1 + " - " + o2;

  std::string t = "";
  print_truncation(t, o1, ps_, verilog);
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t;
}

void VarNounits::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  std::string o1 = "(" + n1 + " * " + itos(cI_) + ")";

  std::string t = "";
  print_truncation(t, o1, ps_, verilog);
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t;
}

void VarTimesC::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  std::string o1 = "(" + n1 + " * " + itos(cI_) + ")";

  std::string t = "";
  print_truncation(t, o1, ps_, verilog);
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t;
}

void VarNeg::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);

  std::string t = "reg signed  [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
  t += "always @(posedge clk) " + name_ + " <= - " + n1;
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t << ";\n";
}

void VarShiftround::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  std::string o1 = n1;
  if (shift_ == 1)
    o1 = "(" + o1 + "+1)>>>1";
  if (shift_ > 1)
    o1 = "( (" + o1 + ">>>" + itos(shift_ - 1) + ")+1)>>>1";
  if (shift_ < 0)
    o1 = o1 + "<<" + itos(-shift_);

  std::string t = "reg signed [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
  t += "always @(posedge clk) " + name_ + " <= " + o1;
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t << ";\n";
}

void VarShift::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  std::string o1 = n1;
  if (shift_ > 0)
    o1 = o1 + ">>>" + itos(shift_);
  if (shift_ < 0)
    o1 = o1 + "<<" + itos(-shift_);

  std::string t = "wire signed [" + itos(nbits_ - 1) + ":0]" + name_ + ";\n";
  t += "assign " + name_ + " = " + o1;
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t << ";\n";
}

void VarMult::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(l3 == 0);
  assert(p1_);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  assert(p2_);
  std::string n2 = p2_->name();
  if (l2 > 0)
    n2 = n2 + "_delay" + itos(l2);
  std::string o1 = n1 + " * " + n2;

  std::string t = "";
  print_truncation(t, o1, ps_, verilog);
  fs << "// " << nbits_ << " bits \t " << kstring() << "\t" << K_ << "\n" << t;
}

void VarInv::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(l2 == 0);
  assert(l3 == 0);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  //first calculate address
  std::string t1 = "addr_" + name_;
  std::string t = "wire [" + itos(nbaddr_ - 1) + ":0] " + t1 + ";\n";
  t = t + "assign " + t1 + " = ";
  if (shift_ > 0)
    t = t + "(" + n1 + ">>>" + itos(shift_) + ") & " + itos(mask_);
  else
    t = t + n1 + " & " + itos(mask_);
  fs << t << "; // address for the LUT\n";

  t = "wire signed [" + itos(nbits_ - 1) + ":0] " + name_ + ";\n";
  fs << t;

  std::string t2 = "LUT_" + name_;

  fs << "Memory #( \n";
  fs << "         .RAM_WIDTH(" << nbits_ << "),            // Specify RAM data width \n";
  fs << "         .RAM_DEPTH(" << Nelements_ << "),                     // Specify RAM depth (number of entries) \n";
  fs << "         .RAM_PERFORMANCE(\"HIGH_PERFORMANCE\"), // \"HIGH_PERFORMANCE\" = 2 clks latency \n";
  fs << "         .INIT_FILE() \n";
  fs << "       ) " << t2 << " ( \n";
  fs << "         .addra(" << itos(nbaddr_) << "\'b0),    // Write address bus, width determined from RAM_DEPTH  \n";
  fs << "         .addrb(" << t1 << " ),                   // Read address bus, width determined from RAM_DEPTH  \n";
  fs << "         .dina(" << itos(nbits_) << "\'b0),      // RAM input data, width determined from RAM_WIDTH   \n";
  fs << "         .clka(clk),      // Write clock \n";
  fs << "         .clkb(clk),      // Read clock  \n";
  fs << "         .wea(1\'b0),        // Write enable  \n";
  fs << "         .enb(1\'b1),        // Read Enable, for additional power savings, disable when not in use  \n";
  fs << "         .rstb(reset),      // Output reset (does not affect memory contents)                      \n";
  fs << "         .regceb(1\'b1),  // Output register enable                                                \n";
  fs << "         .doutb(" << name_ << ")     // RAM output data,                                                \n";
  fs << "     ); \n";
}

void VarDSPPostadd::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(p1_);
  assert(p2_);
  assert(p3_);
  std::string n1 = p1_->name();
  if (l1 > 0)
    n1 = n1 + "_delay" + itos(l1);
  std::string n2 = p2_->name();
  if (l2 > 0)
    n2 = n2 + "_delay" + itos(l2);
  std::string n3 = p3_->name();
  if (l3 > 0)
    n3 = n3 + "_delay" + itos(l3);

  if (shift3_ > 0)
    n3 = n3 + "<<" + itos(shift3_);
  if (shift3_ < 0)
    n3 = n3 + ">>>" + itos(-shift3_);

  std::string n4 = "";
  if (ps_ > 0)
    n4 = ">>>" + itos(ps_);

  fs << name_ + " = DSP_postadd(" + n1 + ", " + n2 + ", " + n3 + ")" + n4 + ";";
}

void VarFlag::print(std::ofstream& fs, Verilog, int l1, int l2, int l3) {
  assert(l1 == 0);
  assert(l2 == 0);
  assert(l3 == 0);

  fs << "wire " << name_ << ";" << std::endl;
  fs << "assign " << name_ << " = (";
  std::map<const VarBase*, std::set<std::string> > cut_strings0, cut_strings1;
  for (const auto& cut : cuts_) {
    if (cut->op() != "cut")
      continue;
    const VarCut* const cast_cut = (VarCut*)cut;
    cast_cut->print(cut_strings0, step_, verilog);
  }
  for (const auto& cut : cuts_) {
    if (cut->op() != "cut")
      cut->print_cuts(cut_strings1, step_, verilog, &cut_strings0);
    else {
      if (cut->cut_var()->p1())
        cut->cut_var()->p1()->print_cuts(cut_strings1, step_, verilog, &cut_strings1);
      if (cut->cut_var()->p2())
        cut->cut_var()->p2()->print_cuts(cut_strings1, step_, verilog, &cut_strings1);
      if (cut->cut_var()->p3())
        cut->cut_var()->p3()->print_cuts(cut_strings1, step_, verilog, &cut_strings1);
    }
  }

  std::string separator = "";
  for (const auto& cut_var : cut_strings0) {
    separator += "((";
    for (const auto& cut_string : cut_var.second) {
      fs << separator << cut_string;
      separator = ") || (";
    }
    separator = ")) && ";
  }
  for (const auto& cut_var : cut_strings1) {
    separator += "((";
    for (const auto& cut_string : cut_var.second) {
      fs << separator << cut_string;
      separator = ") || (";
    }
    separator = ")) && ";
  }

  fs << ")));";
}

void VarBase::print_step(int step, std::ofstream& fs, Verilog) {
  if (!readytoprint_)
    return;
  if (step > step_)
    return;
  int l1 = 0;
  int l2 = 0;
  int l3 = 0;
  if (p1_) {
    p1_->print_step(step, fs, verilog);
    l1 = step_ - p1_->latency() - p1_->step();
  }
  if (p2_) {
    p2_->print_step(step, fs, verilog);
    l2 = step_ - p2_->latency() - p2_->step();
  }
  if (p3_) {
    p3_->print_step(step, fs, verilog);
    l3 = step_ - p3_->latency() - p3_->step();
  }
  if (step == step_) {
    if (l1 < 0 || l2 < 0 || l3 < 0 || (l1 > 0 && l2 > 0 && l3 > 0)) {
      char slog[100];
      sprintf(slog, "%s::print_step(%i): something wrong with latencies! %i %i %i\n", name_.c_str(), step, l1, l2, l3);
      edm::LogVerbatim("Tracklet") << slog;
      dump_msg();
      assert(0);
    }
    if (l1 > 0) {
      if (p1_->op() != "const")
        fs << pipe_delay(p1_, p1_->nbits(), l1);
      else
        l1 = 0;
    }
    if (l2 > 0) {
      if (p2_->op() != "const")
        fs << pipe_delay(p2_, p2_->nbits(), l2);
      else
        l2 = 0;
    }
    if (l3 > 0) {
      if (p3_->op() != "const")
        fs << pipe_delay(p3_, p3_->nbits(), l3);
      else
        l3 = 0;
    }

    if (op_ == "flag") {
      for (const auto& cut : cuts_)
        fs << cut->cut_var()->pipe_delays(step_);
    }

    print(fs, verilog, l1, l2, l3);
    readytoprint_ = false;
  }
}

void VarBase::print_all(std::ofstream& fs, Verilog) {
  for (int i = 0; i <= step_; ++i) {
    fs << "//\n// STEP " << i << "\n\n";
    print_step(i, fs, verilog);
  }
}

void VarBase::design_print(std::vector<VarBase*> v, std::ofstream& fs, Verilog) {
  //step at which all the outputs should be valid
  int maxstep = 0;

  //header of the module

  //inputs
  std::vector<VarBase*> vd;
  vd.clear();
  int imax = v.size();
  for (int i = 0; i < imax; ++i) {
    (v[i])->inputs(&vd);
    int step = v[i]->step() + v[i]->latency();
    if (step > maxstep)
      maxstep = step;
  }

  //print header
  fs << "module \n";
  fs << "(\n";
  fs << "   input clk,\n";
  fs << "   input reset,\n\n";

  imax = vd.size();
  for (int i = 0; i < imax; ++i)
    fs << "   input [" << (vd[i])->nbits() - 1 << ":0] " << (vd[i])->name() << "_wire,\n";
  fs << "\n";

  imax = v.size() - 1;
  for (int i = 0; i < imax; ++i)
    if (v[i]->nbits() > 1)
      fs << "   output [" << (v[i])->nbits() - 1 << ":0] " << (v[i])->name() << "_wire,\n";
    else
      fs << "   output " << (v[i])->name() << "_wire,\n";
  if (imax >= 0) {
    if (v[imax]->nbits() > 1)
      fs << "   output [" << (v[imax])->nbits() - 1 << ":0] " << (v[imax])->name() << "_wire\n";
    else
      fs << "   output " << (v[imax])->name() << "_wire\n";
  }
  fs << ");\n\n";

  //body of the module
  imax = v.size();
  for (int i = 0; i < imax; ++i) {
    fs << "\n//\n";
    fs << "// calculating " << (v[i])->name() << "\n";
    fs << "//\n";
    (v[i])->print_all(fs, verilog);
  }
  fs << "\n";

  //trailer
  fs << "\n";
  fs << "\n//\n";
  fs << "// wiring the outputs \n";
  fs << "// latency = " << maxstep << "\n";
  fs << "//\n";
  for (int i = 0; i < imax; ++i) {
    std::string n = v[i]->name() + "_wire";
    int delay = maxstep - v[i]->step() - v[i]->latency();
    if (delay == 0)
      fs << "assign " << n << " = " << (v[i])->name() << ";\n";
    else
      fs << pipe_delay_wire(v[i], n, v[i]->nbits(), delay);
  }

  fs << "endmodule\n";
}
