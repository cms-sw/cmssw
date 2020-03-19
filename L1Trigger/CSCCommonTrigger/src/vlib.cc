// Alexander Madorsky, University of Florida/Physics.

#include <L1Trigger/CSCCommonTrigger/interface/vlib.h>

globcontrol glc;

char const* obnames[] = {"none ", "reg ", "wire ", "input ", "output ", "inout ", "num ", "temp "};

#define dbgmsg(a) std::cerr << a << " Set breakpoint at " << __FILE__ << ":" << __LINE__ << "\n";

// Signal class --------------------------------------------------------------------------

void Signal::create() {
#ifdef VGEN
  name = "";
  orname = "";
  obname = "";
  lb = "(";
  rb = ")";
#endif

  outhost = host = outreg = ca1 = ca2 = nullptr;
  inited = printable = r = rc = l = pedge = nedge = change = alwaysn = mode = 0;
  h = 8 * Sizeofrval - 1;
  mask = (rval)(-1);
  hostl = -1;
}

Signal::Signal() { create(); }

Signal::Signal(int bits, rval value) {
  create();
#ifdef VGEN
  ostringstream sval;
  sval << dec << bits << "'d" << value;
  Signal::init(bits - 1, 0, sval.str().c_str());
#else
  Signal::init(bits - 1, 0, "");
#endif
  r = rc = value & mask;
}

Signal::Signal(const char* sval) {
  create();
  mode = mnum;
  std::string val = sval;
  int bits;
  unsigned int i;
  char radix;
  rval value = 0;
  int dig;

  sscanf(val.c_str(), "%d'%c", &bits, &radix);
  switch (radix) {
    case 'h':
    case 'H':
      //		sscanf (val.c_str(), "%d'%c%x", &bits, &radix, &value);
      for (i = 0; val[i] != 'h' && val[i] != 'H'; ++i)
        ;
      for (; i < val.length(); ++i) {
        switch (val[i]) {
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
            dig = val[i] - '0';
            break;
          case 'a':
          case 'b':
          case 'c':
          case 'd':
          case 'e':
          case 'f':
            dig = val[i] - 'a' + 10;
            break;
          case 'A':
          case 'B':
          case 'C':
          case 'D':
          case 'E':
          case 'F':
            dig = val[i] - 'A' + 10;
            break;
          default:
            dig = -1;
            break;
        }
        if (dig >= 0) {
          value = value << 4;
          value = value | dig;
        }
      }

      break;
    case 'd':
    case 'D':
      sscanf(val.c_str(), "%d'%c%d", &bits, &radix, reinterpret_cast<int*>(&value));
      break;
    case 'o':
    case 'O':
      //		sscanf (val.c_str(), "%d'%c%o", &bits, &radix, &value);
      for (i = 0; val[i] != 'o' && val[i] != 'O'; ++i)
        ;
      for (; i < val.length(); ++i) {
        switch (val[i]) {
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
            dig = val[i] - '0';
            break;
          default:
            dig = -1;
            break;
        }
        if (dig >= 0) {
          value = value << 3;
          value = value | dig;
        }
      }

      break;

    case 'b':
    case 'B':

      for (i = 0; val[i] != 'b' && val[i] != 'B'; ++i)
        ;
      for (; i < val.length(); ++i) {
        switch (val[i]) {
          case '0':
            value = value << 1;
            break;
          case '1':
            value = value << 1;
            value = value | 1;
            break;
        }
      }
      break;
  }
  Signal::init(bits - 1, 0, val.c_str());
  rc = value & mask;
  r = rc;
}

Signal::Signal(rval n) {
  create();
  mode = mnum;
#ifdef VGEN
  ostringstream ln;
  ln << dec << n;
  init(8 * Sizeofrval - 1, 0, ln.str().c_str());
#else
  init(8 * Sizeofrval - 1, 0, "");
#endif
  inited = 0;
  rc = n;
  r = n;
}

Signal::Signal(int n) {
  create();
  mode = mnum;
#ifdef VGEN
  ostringstream ln;
  ln << dec << n;
  init(Sizeofrval * 8 - 1, 0, ln.str().c_str());
#else
  init(Sizeofrval * 8 - 1, 0, "");
#endif
  inited = 0;
  r = (rval)n;
  rc = r;
}

Signal::Signal(unsigned int n) {
  create();
  mode = mnum;
#ifdef VGEN
  ostringstream ln;
  ln << dec << n;
  init(Sizeofrval * 8 - 1, 0, ln.str().c_str());
#else
  init(Sizeofrval * 8 - 1, 0, "");
#endif
  inited = 0;
  r = (rval)n;
  rc = r;
}

void Signal::makemask() {
  mask = (1LL << (h - l + 1)) - 1LL;
  if (mask == 0LL)
    mask = ~mask;
}

void Signal::init(int high, int low, const char* rname) {
#ifdef VGEN
  name = rname;
  orname = rname;
  lb = "";
  rb = "";
#endif
  if (!inited) {
    h = high;
    l = low;
    makemask();
    inited = 1;
  }
  source = this;
}

void Signal::init(Signal* shost, int high, int low, const char* rname) {
  host = shost;
#ifdef VGEN
  name = rname;
  orname = rname;
  lb = "";
  rb = "";
#endif
  h = high;
  l = low;
  makemask();
  source = this;
  if (host) {
    rc = (host->getr() >> (l - host->getl())) & mask;
    r = rc;
  }
  change = pedge = nedge = 0;
}

#ifdef VGEN
string& Signal::getcatname() {
  if (lb[0] == '{')
    catname = lb + name + rb;
  else
    catname = name;
  return catname;
}
#endif

// unary operators -------------------------------------

#ifdef VGEN
#define unop(op, cop)               \
  Signal Signal::op() {             \
    Signal t;                       \
    t.name = #cop + lb + name + rb; \
    printable = 0;                  \
    t.r = mask & (cop getval());    \
    return t;                       \
  }
#else
#define unop(op, cop)            \
  Signal Signal::op() {          \
    Signal t;                    \
    t.init(h, l, "");            \
    t.r = mask & (cop getval()); \
    return t;                    \
  }
#endif

unop(operator!, !) unop(operator~, ~)

    Signal* Signal::operator&() {
#ifdef VGEN
  printable = 0;
#endif
  return this;
}

// binary operators ------------------------------------
#ifdef VGEN
#define binop(op, cop)                                                       \
  Signal Signal::op(Signal arg) {                                            \
    Signal t;                                                                \
    t.name = lb + name + rb + " " + #cop + " " + arg.lb + arg.name + arg.rb; \
    printable = arg.printable = 0;                                           \
    t.r = (getval() cop arg.getval());                                       \
    return t;                                                                \
  }
#else
#define binop(op, cop)                          \
  Signal Signal::op(Signal arg) {               \
    Signal t;                                   \
    int ln = h - l;                             \
    int aln = arg.h - arg.l;                    \
    t.init((ln > aln) ? ln : aln, 0, "");       \
    t.r = t.mask & (getval() cop arg.getval()); \
    return t;                                   \
  }
#endif

binop(operator+, +) binop(operator-, -) binop(operator*, *) binop(operator/, /) binop(operator%, %) binop(operator^, ^)
    binop(operator<<, <<) binop(operator>>, >>) binop(operator&, &) binop(operator&&, &&) binop(operator|, |)

        Signal Signal::operator||(Signal arg) {
  Signal t;
#ifdef VGEN
  t.name = lb + name + rb + " || " + arg.lb + arg.name + arg.rb;
  t.orname = orname + " or " + arg.orname;
  printable = arg.printable = 0;
#else
  int ln = h - l;
  int aln = arg.h - arg.l;
  t.init((ln > aln) ? ln : aln, 0, "");
#endif
  t.r = t.mask & (getval() || arg.getval());
  t.change = change || arg.change;
  return t;
}

Signal Signal::operator,(Signal arg) {
  Signal t;
#ifdef VGEN
  t.name = name + ", " + arg.name;
  t.lb = "{";
  t.rb = "}";
  printable = arg.printable = 0;
#else
  t.ca1 = this;
  t.ca2 = arg.source;
  t.init(h - l + arg.h - arg.l + 1, 0, "");
#endif
  t.r = (((getval() << (arg.h - arg.l + 1)) & (~(arg.mask))) | arg.getval()) & t.mask;
  return t;
}

// comparison operators ---------------------------------------------------

#ifdef VGEN
#define compop(op, cop)                                                      \
  Signal Signal::op(Signal arg) {                                            \
    Signal t;                                                                \
    t.name = lb + name + rb + " " + #cop + " " + arg.lb + arg.name + arg.rb; \
    printable = arg.printable = 0;                                           \
    t.r = (getval() cop arg.getval());                                       \
    return t;                                                                \
  }
#else
#define compop(op, cop)                \
  Signal Signal::op(Signal arg) {      \
    Signal t;                          \
    t.init("");                        \
    t.r = (getval() cop arg.getval()); \
    return t;                          \
  }
#endif

compop(operator>, >) compop(operator<, <) compop(operator<=, <=) compop(operator>=, >=) compop(operator==, ==)
    compop(operator!=, !=)

    // reduction operators --------------------------------------------------------------
    Signal ror(Signal arg) {
  Signal t;
  rval tr;
#ifdef VGEN
  t.name = "|" + arg.lb + arg.name + arg.rb;
  arg.printable = 0;
#else
  t.init("");
#endif
  tr = (arg.getval()) & arg.mask;
  t.r = (tr != 0) ? 1 : 0;
  return t;
}

Signal rand(Signal arg) {
  Signal t;
  rval tr;
#ifdef VGEN
  t.name = "&" + arg.lb + arg.name + arg.rb;
  arg.printable = 0;
#else
  t.init("");
#endif
  tr = (arg.getval()) & arg.mask;
  t.r = (tr == arg.mask) ? 1 : 0;
  return t;
}

Signal rxor(Signal arg) {
  Signal t;
  rval tr;
  int i;
#ifdef VGEN
  t.name = "^" + arg.lb + arg.name + arg.rb;
  arg.printable = 0;
#else
  t.init("");
#endif
  tr = (arg.getval()) & arg.mask;
  t.r = 0;
  for (i = 0; i < arg.h - arg.l + 1; ++i) {
    t.r = ((tr & 1) != 0) ? !t.r : t.r;
    tr = tr >> 1;
  }
  t.r = t.r & 1;
  return t;
}

// blocking assignment operator -----------------------------------------------------

rval Signal::getval() { return (getalwaysn() == glc.getalwaysn()) ? rc : r; }

Signal Signal::operator=(Signal other) {
#ifndef VGEN
#ifdef _VDEBUG
  switch (mode) {
    case moutput:
      if (glc.getalwaysn() == -1 && outreg != NULL)
        dbgmsg("Assigning output-reg outside always block.");
      if (glc.getalwaysn() != -1 && outreg == NULL)
        dbgmsg("Assigning non-reg output inside always block.");
      break;

    case minout:
      if (glc.getalwaysn() != -1)
        dbgmsg("Assigning inout inside always block.");
      break;

    case minput:
      dbgmsg("Assigning to input is not allowed.");
      return other;

    case mreg:
      if (glc.getalwaysn() == -1)
        dbgmsg("Assigning reg outside always block.");
      break;

    case mwire:
      if (glc.getalwaysn() != -1)
        dbgmsg("Assigning wire inside always block.");
      break;
  }
#endif
#endif
  return asgn(other);
}

Signal Signal::set(Signal other) {
#ifdef VGEN
  glc.setprintassign(0);
#endif
  return asgn(other);
}

Signal Signal::asgn(Signal other) {
  Signal t;

#ifdef VGEN
  t.name = lb + name + rb + " = " + other.getcatname();
  if (glc.printassign())
    std::cout << glc.getmargin() << t.name << ";\n" << flush;
#endif
  rval hr, portionr, portionmask, otr;
  int shn;

  otr = other.getval() & mask;

  if (otr != r) {
    setalwaysn(glc.getalwaysn());
    glc.setchange(1);
  }

  rc = otr;
  if (host)  //  is this a portion of the other register?
  {
    hr = host->rc;
    if (hostl > 0)
      shn = (hostl - host->getl());
    else
      shn = (l - host->getl());
    portionr = rc << shn;
    portionmask = mask << shn;
    host->set((hr & (~portionmask)) | portionr);
  }

  if (ca1 != nullptr && ca2 != nullptr)  // is this a concatenation of the two registers?
  {
    ca2->set(other);
    ca1->set(other >> (ca2->h - ca2->l + 1));
  }
  if (outhost != nullptr && outhost != this && (mode == moutput || mode == minout)) {
    outhost->set(other);
  }
  if (outreg) {
    outreg->set(other);
  }

  t.h = h;
  t.l = l;
  t.mask = mask;
  t.pedge = pedge;
  t.nedge = nedge;
  t.change = change;
  t.r = rc;
#ifdef VGEN
  glc.setprintassign(1);
#endif
  return t;
}

// bit selection operator -----------------------------------------------

Signal Signal::operator()(Signal hn, Signal ln) {
  Signal t;
#ifdef VGEN
  string bname;
  if (hn.getname().compare(ln.getname()) != 0)
    bname = name + "[" + hn.getname() + ":" + ln.getname() + "]";
  else
    bname = name + "[" + hn.getname() + "]";
  hn.printable = ln.printable = 0;
  t.init(this, 0, 0, bname.c_str());
#else
  t.init(this, (unsigned)hn.getval(), (unsigned)ln.getval(), "");
#endif
  t.rc = (getval() >> (t.l - l)) & t.mask;
  t.r = t.rc;
  return t;
}

Signal Signal::operator()(Signal n) { return (*this)(n, n); }

// insertion operator ---------------------------------------------------
std::ostream& operator<<(std::ostream& stream, Signal s) {
  stream << s.getr();
  s.setprintable(0);
  return stream;
}

// input -----------------------------------------------------------------------------
void Signal::input(int high, int low, const char* rname) {
#ifdef VGEN
  if (lb == "{")
    glc.AddIO(lb + name + rb);
  else
    glc.AddIO(name);
#else
#ifdef _VDEBUG
  if (h - l != high - low &&
      inited)  // inited is analysed in case the passed parameter is a temp var. Actually, every operator should return temp with the proper mask,h,l inherited from operands
  {
    dbgmsg("Different port length for input argument: declared [" << high << ":" << low << "], passed: [" << h << ":"
                                                                  << l << "]. ");
  }
#endif
#endif
  Signal::init(high, low, rname);
  if (high >= low) {
    h = high;
    l = low;
  } else {
    h = low;
    l = high;
  }
  mode = minput;
#ifdef VGEN
  obname = obnames[mode];
  ostringstream ln;
  glc.AddParameter(name);
  if (h == l)
    ln << obname << name << ";\n";
  else
    ln << obname << "[" << dec << h << ":" << l << "] " << name << ";\n";
  glc.AddDeclarator(ln.str());
  printable = 0;
#else
  outreg = glc.getparent()->AddOutReg((Signal)(*this));
  r = r & outreg->getmask();
  if (outreg->getr() != r) {
    change = 1;
    glc.getparent()->setchange(1);
    if (r == 1 && outreg->getr() == 0)
      pedge = 1;
    else
      pedge = 0;
    if (r == 0 && outreg->getr() == 1)
      nedge = 1;
    else
      nedge = 0;
  } else
    change = pedge = nedge = 0;
  outreg->setr(r);
  outreg->setrc(r);
  outhost = host = nullptr;
#endif
}

// clock input --------------------------------------------------------------
void Signal::clock(const char* rname) {
#ifdef VGEN
  if (lb == "{")
    glc.AddIO(lb + name + rb);
  else
    glc.AddIO(name);
#else
#ifdef _VDEBUG
  if (h - l != 0 &&
      inited)  // inited is analysed in case the passed parameter is a temp var. Actually, every operator should return temp with the proper mask,h,l inherited from operands
  {
    dbgmsg("Different port length for clock argument: declared [" << 0 << ":" << 0 << "], passed: [" << h << ":" << l
                                                                  << "]. ");
  }
#endif
#endif
  Signal::init(0, 0, rname);
  h = l = 0;
  mode = minput;
#ifdef VGEN
  obname = obnames[mode];
  ostringstream ln;
  glc.AddParameter(name);
  if (h == l)
    ln << obname << name << ";\n";
  else
    ln << obname << "[" << dec << h << ":" << l << "] " << name << ";\n";
  glc.AddDeclarator(ln.str());
  printable = 0;
#else
  outreg = glc.getparent()->AddOutReg((Signal)(*this));
  if (rc != r) {
    change = 1;
    glc.getparent()->setchange(1);
    if (rc == 1 && r == 0)
      pedge = 1;
    else
      pedge = 0;
    if (rc == 0 && r == 1)
      nedge = 1;
    else
      nedge = 0;
  } else
    change = pedge = nedge = 0;
  outreg->setr(r);
  outreg->setrc(r);
  outhost = host = nullptr;
#endif
}

// output -----------------------------------------------------------------------------

void Signal::output(int high, int low, const char* rname) {
#ifdef VGEN
  if (lb == "{")
    glc.AddIO(lb + name + rb);
  else
    glc.AddIO(name);
#else
#ifdef _VDEBUG
  if (h - l != high - low) {
    dbgmsg("Different port length for output argument: declared [" << high << ":" << low << "], passed: [" << h << ":"
                                                                   << l << "]. ");
  }
  if (mode == mreg) {
    dbgmsg("Using reg as output.");
  }
#endif
#endif
  hostl = l;
  Signal::init(high, low, rname);
  if (high >= low) {
    h = high;
    l = low;
  } else {
    h = low;
    l = high;
  }
  mode = moutput;
#ifdef VGEN
  obname = obnames[mode];
  ostringstream ln;
  glc.AddParameter(name);
  if (h == l)
    ln << obname << name << ";\n";
  else
    ln << obname << "[" << dec << h << ":" << l << "] " << name << ";\n";
  glc.AddDeclarator(ln.str());
  printable = 0;
#endif
}

void Signal::output(int high, int low, const char* rname, module* parent) {
  output(high, low, rname);
#ifdef VGEN
  ostringstream ln;
  if (h == l)
    ln << "reg    " << name << ";\n";
  else
    ln << "reg    [" << dec << h << ":" << l << "] " << name << ";\n";
  glc.AddDeclarator(ln.str());
#else
  outreg = parent->AddOutReg((Signal)(*this));
  setr(outreg->getr());
  setchange(outreg->getchange());
  setposedge(outreg->getposedge());
  setnegedge(outreg->getnegedge());
#endif
}

void Signal::output(const char* rname, module* parent) { output(0, 0, rname, parent); }

// inout -----------------------------------------------------------------------------

void Signal::inout(int high, int low, const char* rname) {
#ifdef VGEN
  if (lb == "{")
    glc.AddIO(lb + name + rb);
  else
    glc.AddIO(name);
#else
#ifdef _VDEBUG
  if (h - l != high - low) {
    dbgmsg("Different port length for inout argument: declared [" << high << ":" << low << "], passed: [" << h << ":"
                                                                  << l << "]. ");
  }
#endif
#endif
  hostl = l;
  Signal::init(high, low, rname);
  if (high >= low) {
    h = high;
    l = low;
  } else {
    h = low;
    l = high;
  }
  mode = minout;
#ifdef VGEN
  obname = obnames[mode];
  ostringstream ln;
  glc.AddParameter(name);
  if (h == l)
    ln << obname << name << ";\n";
  else
    ln << obname << "[" << dec << h << ":" << l << "] " << name << ";\n";
  glc.AddDeclarator(ln.str());
  printable = 0;
#endif
}

// reg -------------------------------------------------------------------------------

void Signal::reg(int high, int low, const char* rname) {
  initreg(high, low, rname);
#ifdef VGEN
  if (h == l)
    cout << glc.getmargin() << obname << name << ";\n" << flush;
  else
    std::cout << glc.getmargin() << obname << "[" << dec << h << ":" << l << "] " << name << ";\n" << flush;
#endif
}

void Signal::initreg(int high, int low, const char* rname) {
  Signal::init(high, low, rname);
  mode = mreg;
#ifdef VGEN
  obname = obnames[mode];
#endif
  change = (r != rc);
  if (change)
    glc.getparent()->setchange(1);
  pedge = (rc == 1 && r == 0);
  nedge = (rc == 0 && r == 1);
  r = rc;
}

// parameter class -------------------------------------------------------------------------------

parameter::parameter(const char* rname, Signal arg) : Signal() {
#ifdef VGEN
  obname = "parameter ";
#endif
  init(Sizeofrval * 8 - 1, 0, rname);
  operator=(arg);
}

void parameter::init(int h, int l, const char* rname) {
  Signal::init(h, l, rname);
  change = pedge = nedge = 0;

#ifdef VGEN
  cout << glc.getmargin() << obname << name << flush;
#endif
}

void parameter::operator=(Signal arg) {
  r = arg.getr();
#ifdef VGEN
  cout << " = " << arg.getname() << ";\n" << flush;
#endif
}

// wire  -------------------------------------------------------------------------------

void Signal::wire(int high, int low, const char* rname) {
  Signal::init(high, low, rname);
  mode = mwire;
  outhost = this;
  change = (r != rc);
  if (change && glc.getparent())
    glc.getparent()->setchange(1);
  pedge = (rc == 1 && r == 0);
  nedge = (rc == 0 && r == 1);
  r = rc;
#ifdef VGEN
  obname = obnames[mode];
  if (h == l)
    cout << glc.getmargin() << obname << name << ";\n" << flush;
  else
    std::cout << glc.getmargin() << obname << "[" << dec << h << ":" << l << "] " << name << ";\n" << flush;
#endif
}

void Signal::wire(int high, int low, const char* rname, int i) {
#ifdef VGEN
  ostringstream instnamestream;
  instnamestream << rname << dec << i;
  //	init(high, low, instnamestream.str().c_str());
  wire(high, low, instnamestream.str().c_str());
#else
  wire(high, low, rname);
#endif
}

// memory class -----------------------------------------------------------------------------

void memory::reg(int high, int low, int nup, int ndown, const char* rname) {
  int i;
  if (nup > ndown) {
    up = nup;
    down = ndown;
  } else {
    up = ndown;
    down = nup;
  }
  if (r == nullptr) {
    r = new Signal[up - down + 1];
    for (i = 0; i <= up - down; ++i) {
      r[i].initreg(high, low, "");
      r[i].setr(0);
    }
  } else {
    for (i = 0; i <= up - down; ++i) {
      r[i].initreg(high, low, "");
    }
  }
#ifdef VGEN
  name = rname;
  if (high == low)
    std::cout << glc.getmargin() << "reg " << name << " [" << dec << up << ":" << down << "]"
              << ";\n"
              << flush;
  else
    std::cout << glc.getmargin() << "reg "
              << "[" << dec << high << ":" << low << "] " << name << " [" << dec << up << ":" << down << "]"
              << ";\n"
              << flush;
#endif
}

memory::~memory() {
  if (r != nullptr)
    delete[] r;
  r = nullptr;
}

#ifdef VGEN
Signal memory::operator[](Signal i)
#else
Signal& memory::operator[](Signal i)
#endif
{
#ifdef VGEN
  string ln;
  ln = name + "[" + i.getname() + "]";
  r[0].setname(ln);
  r[0].setorname(ln);
  return r[0];
#else
  rval ind = i.getval();
#ifdef _VDEBUG
  if (ind < down || ind > up) {
    dbgmsg("Memory index out of range: index: " << ind << ", range: [" << up << ":" << down << "]. ");
    return r[down];
  } else
#endif
    return r[ind - down];
#endif
}

// module class -------------------------------------------------------------------------

void module::create() {
  for (unsigned int i = 0; i < sizeof(outreg) / sizeof(Signal*); ++i)
    outreg[i] = nullptr;
  outregn = 0;
  runperiod = nullptr;
}

module::module() { create(); }

module::~module() {
  for (unsigned int i = 0; i < sizeof(outreg) / sizeof(Signal*); ++i) {
    if (outreg[i] != nullptr)
      delete outreg[i];
  }
}

void module::init(const char* mname, const char* iname) {
#ifdef VGEN
  name = mname;
  instname = iname;
#endif
}

void module::init(const char* mname, const char* iname, module* fixt) {
#ifdef VGEN
  name = mname;
  instname = iname;
#endif
  tfixt = fixt;
}

void module::init(const char* mname, const char* iname, int index) {
#ifdef VGEN
  name = mname;
  ostringstream instnamestream;
  instnamestream << iname << dec << index;
  instname = instnamestream.str().c_str();
#endif
}

#ifdef VGEN
void module::PrintHeader() {
  char* username = NULL;
  struct tm* newtime;
  time_t aclock;
  time(&aclock);
  newtime = localtime(&aclock);
  username = std::getenv("USER");
  if (username == NULL || strlen(username) < 2)
    username = std::getenv("USERNAME");
  if (username == NULL || strlen(username) < 2)
    username = std::getenv("LOGNAME");
  cout << "// This  Verilog HDL  source  file was  automatically generated" << std::endl;
  cout << "// by C++ model based on VPP library. Modification of this file" << std::endl;
  cout << "// is possible, but if you want to keep it in sync with the C++" << std::endl;
  cout << "// model,  please  modify  the model and re-generate this file." << std::endl;
  cout << "// VPP library web-page: http://www.phys.ufl.edu/~madorsky/vpp/" << std::endl;
  cout << std::endl;
  if (username != NULL)
    cout << "// Author    : " << username << std::endl;
  cout << "// File name : " << name << ".v" << std::endl;
  cout << "// Timestamp : " << asctime(newtime) << std::endl << flush;
}
#endif

void module::vbeginmodule() {
  switchn = 0;
#ifdef VGEN
  string filename = name + ".v";
  cout << glc.getmargin() << name << " " << instname << std::endl << flush;
  cout << glc.getmargin() << "(" << glc.PrintIO(true).c_str() << std::endl << flush;
  cout << glc.getmargin() << ");\n" << flush;
  vfile.open(filename.c_str());
  outbuf = std::cout.rdbuf(vfile.rdbuf());
  OuterIndPos = glc.getpos();
  oldenmarg = glc.getenablemargin();
  glc.enablemargin(1);
  glc.setpos(0);
  PrintHeader();
  cout << glc.getmargin() << "module " << name << std::endl << glc.getmargin() << "(" << flush;
  glc.setfunction(0);
  glc.Print();
  glc.setFileOpen(1);
#endif
}

void module::vendmodule() {
#ifdef VGEN
  glc.Outdent();
  cout << glc.getmargin() << "endmodule\n" << flush;
  glc.setpos(OuterIndPos);
  cout.rdbuf(outbuf);
  vfile.close();
  glc.enablemargin(oldenmarg);
#endif
  outregn = 0;
}

Signal module::posedge(Signal arg) {
  Signal t;
#ifdef VGEN
  string ln = "";
  ln = "posedge " + arg.getname();
  t.init(NULL, 0, 0, ln.c_str());
#else
  t.init(nullptr, 0, 0, "");
#endif
  if (arg.getposedge())
    glc.setce(0);
  t.setchange(arg.getposedge());
  return t;
}

Signal module::negedge(Signal arg) {
  Signal t;
#ifdef VGEN
  string ln = "";
  ln = "negedge " + arg.getname();
  t.init(NULL, 0, 0, ln.c_str());
#else
  t.init(nullptr, 0, 0, "");
#endif
  if (arg.getnegedge())
    glc.setce(0);
  t.setchange(arg.getnegedge());
  return t;
}

Signal* module::AddOutReg(Signal arg) {
  if (outreg[outregn] == nullptr) {
    outreg[outregn] = new Signal;
    outreg[outregn]->setr(0);
  }
  outreg[outregn]->reg(arg.geth(), arg.getl(), "");
  outregn++;
  return outreg[outregn - 1];
}

Signal module::ifelse(Signal condition, Signal iftrue, Signal iffalse) {
#ifdef VGEN
  Signal t;
  string ln;
  ln = "(" + condition.getname() + ") ? " + iftrue.getname() + " : " + iffalse.getname();
  t.setname(ln);
  return t;
#else
  if (condition.getbool())
    return iftrue;
  else
    return iffalse;
#endif
}

// function class -------------------------------------------------------------

void function::makemask(int hpar, int lpar) {
  //int i;
  unsigned int lng = hpar - lpar + 1;

  if (lng < Sizeofrval * 8)
    mask = (1LL << lng) - 1;
  else
    mask = (rval)(-1);
}

void function::init(int high, int low, const char* rname) {
#ifdef VGEN
  name = rname;
  cout << "`include " << '"' << name << ".v" << '"' << "\n" << flush;
#endif
  if (high >= low) {
    h = high;
    l = low;
  } else {
    h = low;
    l = high;
  }
  makemask(h, l);
}

void function::vbeginfunction() {
#ifdef VGEN
  string filename = name + ".v";
  retname = name + "(" + glc.PrintIO(false) + ")";
  vfile.open(filename.c_str());
  outbuf = std::cout.rdbuf(vfile.rdbuf());
  OuterIndPos = glc.getpos();
  oldenmarg = glc.getenablemargin();
  glc.enablemargin(1);
  glc.setpos(0);
  PrintHeader();
  cout << glc.getmargin() << "function [" << dec << h << ":" << l << "] " << name << ";\n" << flush;
  glc.setfunction(1);
  glc.Print();
  glc.setFileOpen(1);
  result.setname(name);
  result.setbrackets("", "");
#endif
  switchn = 0;
  OldChange = glc.getchange();
}

void function::vendfunction() {
  result.sethlmask(h, l, mask);
  result.setr(result.getr() & mask);
  glc.setchange(OldChange);
#ifdef VGEN
  glc.Outdent();
  cout << glc.getmargin() << "endfunction\n" << flush;
  glc.setpos(OuterIndPos);
  cout.rdbuf(outbuf);
  result.setname(retname);
  result.setbrackets("", "");
  vfile.close();
  glc.enablemargin(oldenmarg);
#endif
  outregn = 0;
}

// globcontrol class --------------------------------------------------------------------

globcontrol::globcontrol() {
#ifdef VGEN
  nomargin = 0;
  ndio = 0;
  npar = 0;
  ndecl = 0;
  indpos = 0;
  pa = 1;
  zeromargin = "";
  VFileOpen = 0;
#endif
  alwayscnt = -1;
  alwaysn = 1;
  change = 0;
}

#ifdef VGEN
void globcontrol::Print() {
  int i;

  if (functiondecl == 0) {
    Indent();
    for (i = 0; i < npar; ++i) {
      cout << std::endl;
      cout << getmargin();
      cout << pars[i];
      if (i != npar - 1)
        std::cout << ",";
    }
    Outdent();
    cout << std::endl << getmargin() << ");\n";
  }
  Indent();
  cout << "\n";

  for (i = 0; i < ndecl; ++i) {
    cout << glc.getmargin() << decls[i];
  }
  npar = ndecl = 0;
  cout << "\n" << flush;
}

string& globcontrol::PrintIO(bool col) {
  int i;

  if (col)
    Indent();
  outln = "";
  if (ndio > 0) {
    for (i = 0; i < ndio; ++i) {
      if (col) {
        outln += "\n";
        outln += getmargin();
      }
      outln += dios[i];
      if (i < ndio - 1)
        outln += ",";
    }
  }
  ndio = 0;
  if (col)
    Outdent();
  return outln;
}

void globcontrol::PrepMargin() {
  int i;
  margin = "";
  for (i = 0; i < indpos; ++i) {
    margin += "    ";
  }
}

void globcontrol::AddIO(std::string ln) {
  dios[ndio] = ln;
  ndio++;
}
#endif

void globcontrol::setchange(int i) { change = i; }

char* globcontrol::constant(int bits, char* value) {
  sprintf(constring, "%d%s", bits, value);
  return constring;
}

char* globcontrol::constant(int bits, int value) {
  sprintf(constring, "%d'd%u", bits, (unsigned)value);
  return constring;
}
