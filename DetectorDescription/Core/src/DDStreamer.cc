#include "DetectorDescription/Core/interface/DDStreamer.h"

#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iomanip>

DDStreamer::DDStreamer()
 : o_(0), i_(0)
 {
 }

DDStreamer::DDStreamer(std::ostream & os)
 :  o_(0), i_(0)
{
  if (os) {
    o_ = &os;
  }
  else {
    throw cms::Exception("DDException") << "DDStreamer::DDStreamer(std::ostream&): not valid std::ostream";
  }
}

DDStreamer::DDStreamer(std::istream & is)
 :  o_(0), i_(0)
{
  if (is) {
    i_ = &is;
  }
  else {
    throw cms::Exception("DDException") << "DDStreamer::DDStreamer(std::ostream&): not valid std::ostream";
  }
}

DDStreamer::~DDStreamer() {}
/*  
DDName dd_get_name(std::istream & is)
{
   size_t nm(0), ns(0);
   is >> nm;
   is >> ns;
   return DDName(std::make_pair(nm,ns));
}

void nameout(std::ostream & o, const DDName & n)
{
  o << n.id().first << ' ' << n.id().second;
}
*/


std::string dd_get_delimit(std::istream & is, char d)
{
  std::string nm;
  char i;
  while ((i=is.get()) && i != d) ;
  while ((i=is.get()) && i != d) {
      nm = nm + i; 
   }
  return nm;
}

DDName dd_get_name_string(std::istream & is)
{  
   std::string nm(dd_get_delimit(is,'"'))  ;
   return DDName(nm,
                 dd_get_delimit(is,'"'));
/*
   char i;
   while ((i=is.get()) && i != '"') ;
   while ((i=is.get()) && i != '"') {
      nm = nm + i; 
   }
   std::pair<std::string,std::string> p = DDSplit(nm);
   return DDName(p.first,p.second);
*/   
}

DDName dd_get_name(std::istream & is)
{
  size_t id(0);
  is >> id;
  return DDName(id);
}

void nameout_strings(std::ostream & o, const DDName & n)
{
  o << '"' << n.name() << '"' << ' ' << '"' << n.ns() << '"' ;
}
 
void nameout(std::ostream & o, const DDName & n)   
{
  o << n.id();
}


void DDStreamer::write()
{
   if (o_ && *o_) {
     write(*o_);
   }
   else {
     throw cms::Exception("DDException") << "DDStreamer::write(): bad std::ostream";
   }
}

void DDStreamer::read()
{
   if (i_ && *i_) {
    read(*i_);
   }
   else {
     throw cms::Exception("DDException") << "DDStreamer::read(): bad std::istream";
   }
}

void DDStreamer::write(std::ostream & os)
{
  o_=&os;
  std::streamsize prec(os.precision());
  os << std::setprecision(26) << std::scientific;
  names_write();
  vars_write();  
  
  materials_write();
  solids_write();
  parts_write();
  
  pos_write();
  specs_write();
  
  rots_write();    
  //os << DDI::Singleton<DDName::IdToName>::instance().size() << std::endl;
  //names_write();
  os << resetiosflags((std::ios_base::fmtflags)0);
  os << std::setprecision(prec);
}


void DDStreamer::read(std::istream & is)
{

  i_=&is;
  names_read();
  vars_read();
  
  materials_read();
  solids_read();
  parts_read();
  
  pos_read();
  specs_read();
  rots_read();        
}


void DDStreamer::names_write()
{
  DCOUT('Y', "DDStreamer::names_write()");
  std::ostream & os = *o_;
  DDName::IdToName & ids = DDI::Singleton<DDName::IdToName>::instance();
  
  DDName::IdToName::const_iterator it(ids.begin()), ed(ids.end());
  os << ids.size() << std::endl;
  size_t count(0);
  for (; it != ed; ++it) {
    os << '"' << (*it)->first.first << '"' << ' ' 
       << '"' << (*it)->first.second << '"' << ' ' 
       << count << std::endl;
    ++count;
  }
  
}


void DDStreamer::names_read()
{
  DCOUT('Y', "DDStreamer::names_read()");
  std::istream & is = *i_;
  DDName::IdToName & ids = DDI::Singleton<DDName::IdToName>::instance();
  DDName::Registry & reg = DDI::Singleton<DDName::Registry>::instance();
  
  size_t s;
  is >> s;
  ids.clear();
  //ids.resize(s);
  reg.clear();
  size_t i(0);
  //std::string nm; getline(is,nm);
  for (; i<s; ++i) {
    std::string nm(dd_get_delimit(is,'"'));
    std::string ns(dd_get_delimit(is,'"'));
    size_t id(0);
    is >> id;
    DDName::defineId(std::make_pair(nm,ns),id);
  }
}


template<class T> 
size_t dd_count(const T & /*dummy*/)
{
  size_t result(0);
  typename T::template iterator<T> it(T::begin()), ed(T::end());
  for (; it!=ed; ++it) {
    if (it->isDefined().second) {
      ++result;
    }
  }
  return result;
}


struct double_binary
{
  explicit double_binary(double d) : val_(d) { }
  double_binary() : val_(0) { }
  double val_;  
};

typedef double_binary B;

std::ostream & operator<<(std::ostream & os, double_binary b)
{
  const char * d = (const char *)(&(b.val_));
  //size_t s(sizeof(double)), i(0);
  //os << '(';
  os.write(d,sizeof(double));
  return os;
}


inline std::istream & operator>>(std::istream & is, double_binary & b)
{
  char * d = (char *)(&(b.val_));
  //size_t s(sizeof(double)), i(0);
  is.read(d,sizeof(double));
  return is;
} 


void DDStreamer::materials_write()
{
  DCOUT('Y', "DDStreamer::materials_write()");
  std::ostream & os = *o_;
  DDMaterial::iterator<DDMaterial> it(DDMaterial::begin()), ed(DDMaterial::end());
  size_t no = dd_count(DDMaterial());
  os << no << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;
    const DDMaterial & m = *it;
    os << "--Material: " << m.name() << " @ " ;
    nameout(os,m.name()); 
    DCOUT('y', "write-material=" << m.name());
    os << ' ' << m.z() << ' ' << m.a() << ' ' << m.density() << ' ';
    
    int noc = m.noOfConstituents();
    os << noc;
    int j=0;
    for (; j<noc; ++j) {
      DCOUT('y', "  write-const-material=" << m.constituent(j).first.name());
      os << ' ';
      nameout(os,m.constituent(j).first.name());
      os << ' ' << m.constituent(j).second;
    }
    os << std::endl;
  }
}


void DDStreamer::materials_read()
{
  DCOUT('Y', "DDStreamer::materials_read()");
  std::istream & is = *i_;
  //DDMaterial::clear();
  size_t n=0;
  is >> n;
  size_t i=0;
  for (; i < n; ++i) { // Materials
    is.ignore(1000,'@');
    DDName dn = dd_get_name(is);
    double z(0), a(0), d(0);
    is >> z;
    is >> a;
    is >> d;
    int comp(0);
    is >> comp; // composites
    if (comp) { // composite material
      DDMaterial m(dn,d);
      DCOUT('y', "read-comp-material=" << m.name());
      int j=0;
      for(; j<comp; ++j) {
        DDName cname(dd_get_name(is));
	double fm(0);
	is >> fm;
	DDMaterial constituent(cname);
        DCOUT('y', "  read-composite=" << constituent.name());
	m.addMaterial(constituent,fm);
      }
    }
    else { // elementary material
      DDMaterial m(dn,z,a,d);
      DCOUT('y', "read-elem-material=" << m.name());
    }
  }
}

void dd_stream_booleans(std::ostream& os, DDSolid s, DDSolidShape /*sh*/)
{
  DDBooleanSolid b(s);
  DDRotation temprot = b.rotation();
  if(!temprot.isDefined().second) {
    temprot = DDRotation();
    edm::LogError("DDStreamer") << "DDStreamer::dd_stream_booleans(): solid=" << s.name() << " has no rotation. Using unit-rot." << std::endl;
  }
  nameout(os,temprot.name()); 
  // binary output of the translation std::vector
  os << ' ' << B(b.translation().x()) // << ' '
     << B(b.translation().y()) // << ' '
     << B(b.translation().z()) << ' '; 
  nameout(os,b.solidA().name());
  os << ' ';
  nameout(os,b.solidB().name());	    
}

void dd_stream_reflected(std::ostream & os, DDSolid s)
{
      DDReflectionSolid ref(s);
      nameout(os,ref.unreflected().name());
}

void DDStreamer::solids_write()
{
  DCOUT('Y', "DDStreamer::solids_write()");
  std::ostream & os = *o_;
  DDSolid::iterator<DDSolid> it(DDSolid::begin()), ed(DDSolid::end());
  size_t no = dd_count(DDSolid());
  os << no << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;  
    const DDSolid & s = *it;
    DCOUT('y', "write-solid=" << s << " enum=" << s.shape());
    os << "--Solid: " << s.name() << ' ' << DDSolidShapesName::name(s.shape()) << " @ ";
    nameout(os,s.name()); 
    os << ' ' << s.shape() << ' ';
    switch (s.shape()) {
    case ddunion: case ddsubtraction: case ddintersection:
      dd_stream_booleans(os, s, s.shape());
      break;
    case ddreflected:
      dd_stream_reflected(os, s);
      break;
    default:
      size_t ps = s.parameters().size();
      os << ps;
      const std::vector<double> & p = s.parameters();
      os << ' ';
      os.write((char*)(&(*p.begin())),ps*sizeof(double));
      /*
      std::vector<double>::const_iterator it(p.begin()), ed(p.end());
      for (; it != ed; ++it) {
        os << ' ' << *it;
      }
      */
    }
    os << std::endl;
  }
}



void dd_get_boolean_params(std::istream & is, DDRotation & r, DDTranslation & t, DDSolid & a, DDSolid & b)
{
   DDName n = dd_get_name(is);
   r = DDRotation(n);
   //double x(0), y(0), z(0);
   B x,y,z;
   char cr = is.get();
   if(cr != ' ') 
      throw cms::Exception("DDException") << "DDStreamer::get_boolean_param(): inconsistent sequence! no blank delimiter before trans!";
   is >> x;
   is >> y;
   is >> z;
   t = DDTranslation(x.val_,y.val_,z.val_);
   n = dd_get_name(is);
   a = DDSolid(n);
   n = dd_get_name(is);
   b = DDSolid(n);
   DCOUT('y', "boolean-par: rot=" << r.name() << " t=" << t << " a=" << a.name() << " b=" << b.name());
}

void DDStreamer::solids_read()
{
  DCOUT('Y', "DDStreamer::solids_read()");
  std::istream & is = *i_;
  //DDSolid::clear();
  size_t n=0;
  is >> n;
  size_t i=0;
  for (; i < n; ++i) { // Solids
    is.ignore(1000,'@');
    DDName dn = dd_get_name(is);

    size_t sp(0);
    is >> sp;
    DDSolidShape shape = DDSolidShape(sp);
    
    // boolean solids
    if ( (shape==ddunion) | (shape==ddsubtraction) || (shape==ddintersection) ) {
      DDRotation r;
      DDTranslation t;
      DDSolid a;
      DDSolid b;
      dd_get_boolean_params(is,r,t,a,b);
      switch (shape) {
      case ddunion:
        DDSolidFactory::unionSolid(dn,a,b,t,r);
	break;
      case ddintersection:
        DDSolidFactory::intersection(dn,a,b,t,r);
	break;
      case ddsubtraction:
        DDSolidFactory::subtraction(dn,a,b,t,r);
	break;	
      default:
        throw cms::Exception("DDException") << "DDStreamer::solids_read(): messed up in boolean solid reading!";	
      }
    }
    
    // reflection solids
    else if (shape==ddreflected) {
      DDName ref_nm = dd_get_name(is);
      DDSolidFactory::reflection(dn,ref_nm);
    }
    else if ( (shape==ddbox ) ||
              (shape==ddtrap) ||
              (shape==ddcons) ||
              (shape==ddtubs) ||    
              (shape==ddpolycone_rz) ||
              (shape==ddpolycone_rrz) ||
              (shape==ddpolyhedra_rz) ||	      	          
              (shape==ddpolyhedra_rrz) ||
              (shape==ddpseudotrap) ||
              (shape==ddshapeless) )
    {
      // read in the solid's parameters
      size_t npars(0);
      is >> npars;
      
      std::vector<double> p(npars);
      if(npars) {
        //edm::LogError("DDStreamer") << npars << flush << std::endl;
        char c;
        c = is.get();
        if (c != ' ') {
	   edm::LogError("DDStreamer") << "delimiter: " << c << std::endl;
          throw cms::Exception("DDException") << "DDStreamer::solids_read(): wrong separator in atomic for atomic solids parameters";
	}
        is.read((char*)&(*(p.begin())),npars*sizeof(double));	
        /*
	size_t i(0);
	for(; i< npars; ++i) {
          double d(0);
	  is >> d;
	  p.push_back(d);
        }
        */
      }	
      DDSolid so = DDSolid(dn,shape,p);
      DCOUT('y', "read-solid=" << so);     
    }
    else {
      edm::LogError("DDStreamer") << "wrong solid enum: " << shape << std::endl;
      throw cms::Exception("DDException") << "Error in DDStreamer::solids_read(), wrong shape-enum!";
    }
  }
}

void DDStreamer::parts_write()
{
  DCOUT('Y', "DDStreamer::parts_write()");
  std::ostream & os = *o_;
  DDLogicalPart::iterator<DDLogicalPart> it(DDLogicalPart::begin()), ed(DDLogicalPart::end());
  size_t no = dd_count(DDLogicalPart());
  os << no << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;  
    const DDLogicalPart & lp = *it;
    os << "--Part: " << lp.name() << " @ ";
    nameout(os,lp.name()); 
    os << ' ' << lp.category() << ' ';
    nameout(os,lp.material().name());
    os << ' ';
    nameout(os,lp.solid().name());
    os << std::endl;
  }
}


void DDStreamer::parts_read()
{
  DCOUT('Y', "DDStreamer::parts_read()");
  std::istream & is = *i_;
  //DDLogicalPart::clear();
  size_t n=0;
  is >> n;
  size_t i=0;
  for (; i < n; ++i) { // LogicalParts
    is.ignore(1000,'@');
    DDName dn = dd_get_name(is);
    size_t cat(0);
    is >> cat;
    DDEnums::Category categ = DDEnums::Category(cat);
    DDName mat = dd_get_name(is);
    DDName sol = dd_get_name(is);
    DDLogicalPart lp(dn,mat,sol,categ);
    DCOUT('y', "read-lp=" << lp);
  }
}

void dd_rot_bin_out(std::ostream & os, const DDRotationMatrix & rm)
{
  double v[9]; 
  rm.GetComponents(v,v+9);
  for (int i=0;i<9;i++)
    os        << B(v[i]);
}

void dd_rot_out(std::ostream & os, const DDRotation & r) {
    os << "--Rot: " << r.name() << " @ ";
    nameout(os,r.name());
    os << ' ';
    const DDRotationMatrix & rm = *(r.rotation());
    DCOUT('y', "write-rots=" << r.name());
/*
    os << ' ' << B(rep.xx_) << ' ' << B(rep.xy_) << ' ' << B(rep.xz_) << ' '
              << B(rep.yx_) << ' ' << B(rep.yy_) << ' ' << B(rep.yz_) << ' '
	      << B(rep.zx_) << ' ' << B(rep.zy_) << ' ' << B(rep.zz_) << std::endl; 
*/	      
   dd_rot_bin_out(os,rm);	      
   os << std::endl;
}

void DDStreamer::rots_write()
{ 
  DCOUT('Y', "DDStreamer::rots_write()");
  std::ostream & os = *o_;
  DDRotation::iterator<DDRotation> it(DDRotation::begin()), ed(DDRotation::end());
  size_t no = dd_count(DDRotation());
  os << no << std::endl;
  //DDName ano;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;  
    const DDRotation & r = *it;
    //if (r.name().id() == ano.id()) {
    //  continue;
    //}
    dd_rot_out(os,r);   
  } 
}


void dd_rot_bin_in(std::istream & is, DDRotationMatrix & r)
{
    double v[9];
    B w;
    for (int i=0; i<9;i++) {
      is >> w; v[i]=w.val_;
    }
    r.SetComponents(v,v+9);
}

void DDStreamer::rots_read()
{
  DCOUT('Y', "DDStreamer::rots_read()");
  std::istream & is = *i_;
  //DDRotation::clear();
  size_t n=0;
  is >> n;
  size_t i=0;
  for (; i < n; ++i) { // Rotations
    is.ignore(1000,'@');
    DDName dn = dd_get_name(is);
    char c = is.get();
    if (c != ' ') { 
      throw cms::Exception("DDException") << "DDStreamer::rots_read(): inconsitency! no blank separator found!";
    }
 
    DDRotationMatrix * rm = new DDRotationMatrix();
    dd_rot_bin_in(is,*rm);
    DDRotation ddr = DDRotation(dn,rm);
    DCOUT('y',"read-rots=" << ddr.name());
  }
}

void DDStreamer::pos_write()
{
  DCOUT('Y', "DDStreamer::pos_write()");
  DDCompactView cpv;
  const DDCompactView::graph_type & g = cpv.graph();
  DDCompactView::graph_type::const_iterator it = g.begin_iter();
  DDCompactView::graph_type::const_iterator ed = g.end_iter();
  std::ostream & os = *o_;
  // first the root
  DDLogicalPart rt = DDRootDef::instance().root();
  os << "--Root: @ ";
  nameout(os,rt.name());
  os << std::endl;
  //os << g.edge_size() << std::endl;
  DDCompactView::graph_type::const_iterator iit = g.begin_iter();
  DDCompactView::graph_type::const_iterator eed = g.end_iter();
  size_t count(0);
  for(; iit != eed; ++iit) {
    ++count;
  }
  os << count << std::endl;
  count=0;
  DDName unit_rot_name;
  DDRotationMatrix unit_rot;
  for(; it != ed; ++it) {
     os << "--Pos[" << count << "]: @ "; ++count;
     //const DDLogicalPart & fr = it->from(); 
     nameout(os, it->from().name());
     os << ' ';
     nameout(os, it->to().name());
     os << ' ' << it->edge()->copyno_;
     const DDTranslation & tr = it->edge()->translation();
     //os << ' ' << B(tr.x()) << ' ' << B(tr.y()) << ' ' << B(tr.z());
     os << ' ' << B(tr.x()) << B(tr.y()) << B(tr.z());
     const DDRotation & ro = it->edge()->rot_;
     os << ' ';
     /* if it's an anonymous rotation stemming from an AlgoPosPart
        then it's id must be the one of a unit rotation AND
	it must be defined at this point AND it must not be the
	unit rotation.
	A character identifier is issues to mark the type of the rotation:
	a ... anonymous rotation, followed by the binary numbers of the matrix
	u ... unit-rotation matrix, no values following
	r ... regular defined rotation-matrix (named rotation matrix) followed by name-id
     */
     if (ro.name() == unit_rot_name) {
       if(ro.isDefined().second) {
         if(*(ro.rotation()) != unit_rot) {
	   os << "a ";
	   dd_rot_bin_out(os,*(ro.rotation()));
	 }  
	 else {
	   os << "u ";
	 }
       }
     }
     else {
       os << "r ";
       nameout(os, ro.name());
     }  
     os << std::endl;
  }

}


void DDStreamer::pos_read()
{
  DCOUT('Y', "DDStreamer::pos_read()");
  std::istream & is = *i_;
  is.ignore(1000,'@');
  DDName rtname = dd_get_name(is);
  DDLogicalPart root(rtname);
  DCOUT('y', "root is: " << root.name());
  DDRootDef::instance().set(root);
  size_t n=0;
  is >> n;
  size_t i=0;
  DDCompactView cpv;
  DDCompactView::graph_type & g = const_cast<DDCompactView::graph_type&>(cpv.graph());
  //  DDPositioner pos_(&cpv);
  //LogDebug << "===== GRAPH SIZE = " << g.size() << " ======" << std::endl << std::endl;
  if (g.size()) {
    edm::LogWarning("DDStreamer") << std::endl;
    edm::LogWarning("DDStreamer") << "DDStreamer::pos_read(): The CompactView already contains some position information." << std::endl
         << "                        It may cause an inconsistent geometry representation!" << std::endl << std::endl;
    throw cms::Exception("DDException") << "DDStreamer::pos_read() failed; CompactView has already been populated by another data source";	 
  }
  for (; i < n; ++i) { // Positions
    is.ignore(1000,'@');
    DDName from(dd_get_name(is));
    DDName to(dd_get_name(is));
    std::string cp;
    is >> cp;
    char cr = is.get();
    if (cr != ' ') throw cms::Exception("DDException") << "DDStreamer::pos_read(): inconsistent sequence! no blank delimiter found!";
    //double x,y,z;
    B x,y,z;
    is >> x;
    is >> y;
    is >> z;
    DDTranslation t(x.val_,y.val_,z.val_);
    is.ignore();
    char rottype = is.get();
    DDRotationMatrix * matrix(0);
    //DDName rotname;
    DDRotation rot;
    switch(rottype) {
      case 'a': // anonymous rotation
        is.ignore();
        matrix = new DDRotationMatrix;
	dd_rot_bin_in(is,*matrix);
	rot = DDanonymousRot(matrix);
        break;
      case 'u': // unit rotation
        break;
      case 'r': // regular (named) rotation
        rot = DDRotation(dd_get_name(is));
        break;
      default:
        std::string message = "DDStreamer::pos_read(): could not determine type of rotation\n";
        throw cms::Exception("DDException") << message;
      }	              	               
    //DDName rot(dd_get_name(is));
    cpv.position(DDLogicalPart(to),DDLogicalPart(from),cp,t,rot); 
    DCOUT('y', " pos-read: f=" << from << " to=" << to << " t=" << t << " r=" << rot);
  }
}

/*
void dd_ps_out(std::ostream & os, DDPartSelection* p)
{
   size_t i(0),j(p->size());
   os << ' ' << j << ' ';
   for (; i<j; ++i) {
     nameout(os,(*p)[j].lp_.name());
     os << ' ';
     
   }
}

void DDStreamer::specs_write()
{
  DCOUT('Y', "DDStreamer::parts_write()");
  std::ostream & os = *o_;
  DDLogicalPart::iterator<DDLogicalPart> it(DDLogicalPart::begin()), ed(DDLogicalPart::end());
  size_t no = DDLogicalPart::size();
  os << no << std::endl;
  for (; it != ed; ++it) {
    const DDLogicalPart & lp = *it;
    const std::vector< std::pair<DDPartSelection*,DDsvalues_type> > & sp = lp.attachedSpecifics();
    if ((size_t s = sp.size())) {
      os << "--Specs " << lp.name() << " @ " << s << ' ';
      size_t j=0;
      for (; j<s; ++j) {
        dd_ps_out(os,s[j].first);
	DDsvalues_type* sv = s[j].second;
	os << ' ' << sv->size() << ' ';
	dd_sv_out(os,sv);
      }
    }
}
*/

void DDStreamer::specs_write()
{
  DCOUT('Y', "DDStreamer::specs_write()");
  std::ostream & os = *o_;
  DDSpecifics::iterator<DDSpecifics> it(DDSpecifics::begin()), ed(DDSpecifics::end());
  size_t no = dd_count(DDSpecifics());
  os << no << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;  
    const DDSpecifics & sp = *it;
    os << "--Spec: @ ";
    nameout(os,sp.name());
    os << ' ' << sp.selection().size() << std::endl;
    std::vector<DDPartSelection>::const_iterator sit(sp.selection().begin()), sed(sp.selection().end());
    for (; sit != sed; ++sit) {
      os << *sit << std::endl;
    }
    os << sp.specifics().size() << std::endl;
    DDsvalues_type::const_iterator vit(sp.specifics().begin()), ved(sp.specifics().end());
    for (; vit != ved; ++vit) {
      const DDValue & v = vit->second;
      os << ' ' << '"' << v.name() << '"' << ' ';
      if (v.isEvaluated()) {
        os << 1 << ' ';
      }
      else {
        os << 0 << ' ';
      }
      os << v.size() << ' ';
      if (v.isEvaluated()) {
        size_t s=v.size();
	size_t i=0;
	for (; i<s; ++i) {
	  os << '"' << v[i].first << '"' << ' ' << v[i].second << ' ';
	}
      }
      else {
        size_t s=v.size();
	size_t i=0;
	const std::vector<std::string> & vs = v.strings();
	for (; i<s; ++i) {
	  os << '"' << vs[i] << '"' << ' ';
        }
      }
      os << std::endl;
      
    }
  }  
}

void DDStreamer::specs_read()   
{
  DCOUT('Y', "DDStreamer::specs_read()");
  std::istream & is = *i_;
  //DDSpecifics::clear();
  size_t n=0;
  is >> n;
  size_t i=0;
  for (; i < n; ++i) { // Specifics
    is.ignore(1000,'@');
    DDName sn(dd_get_name(is));
    size_t nps(0);
    is >> nps;
    size_t ii=0;
    std::vector<std::string> ps;
    is.ignore(100,'\n');
    for (; ii < nps; ++ii) {
      std::string s;
      getline(is,s);
      DCOUT('y', "specs-ps=" << s);
      ps.push_back(s);
    }
    is >> nps;
    ii=0;
    DDsvalues_type sv;
    for(; ii<nps; ++ii) {
      std::string name = dd_get_delimit(is,'"');
      bool evl(false);
      is >> evl;
      size_t no(0);
      is >> no;
      std::vector<DDValuePair> valv;
      DDValue result;
      if (evl) {
        size_t iii=0;
	for(; iii<no; ++iii) {
	  std::string strv = dd_get_delimit(is,'"');
	  double dblv(0);
	  is >> dblv;
	  DDValuePair valp(strv,dblv);
	  valv.push_back(valp);
	}
	result = DDValue(name,valv);
	result.setEvalState(true);
      }
      else {
        size_t iii=0;
	for(; iii<no; ++iii) {
	  std::string strv = dd_get_delimit(is,'"');	
	  DDValuePair valp(strv,0);
	  valv.push_back(valp);
	}
	result = DDValue(name,valv);
	result.setEvalState(false);
      }
      sv.push_back(DDsvalues_Content_type(result,result));
    }
    std::sort(sv.begin(),sv.end());
    DDSpecifics sp(sn,ps,sv,false);
    DCOUT('y', " specs-read: " << sp);
  }  
}


void DDStreamer::vars_write()
{
  std::ostream & os = *o_;
  ClhepEvaluator & ev = ExprEvalSingleton::instance();
  ClhepEvaluator * eval = dynamic_cast<ClhepEvaluator*>(&ev);
  if (eval){
    const std::vector<std::string> & vars = eval->variables();
    const std::vector<std::string> & vals = eval->values();
    if (vars.size() != vals.size()) {
      throw cms::Exception("DDException") << "DDStreamer::vars_write(): different size of variable names & values!";
    }
    size_t i(0), s(vars.size());
    os << s << std::endl;
    for (; i<s; ++i) {
      os << '"' << vars[i] << '"' << ' ' 
         << '"' << vals[i] << '"' << std::endl; 
    }  
  }
  else {
    throw cms::Exception("DDException") << "DDStreamer::vars_write(): expression-evaluator is not a ClhepEvaluator-implementation!";
  }
}


void DDStreamer::vars_read()
{
  DCOUT('Y', "DDStreamer::vars_read()");
  std::istream & is = *i_;
  ClhepEvaluator & ev = ExprEvalSingleton::instance();
  ClhepEvaluator * eval = dynamic_cast<ClhepEvaluator*>(&ev);
  if (eval){
    size_t n(0);
    is >> n;
    size_t i(0);
  
    for(; i<n; ++i) {
      std::string name(dd_get_delimit(is,'"'));
      std::string value(dd_get_delimit(is,'"'));
      eval->set(name,value);
    }
  }
  else {
    throw cms::Exception("DDException") << "DDStreamer::vars_write(): expression-evaluator is not a ClhepEvaluator-implementation!";  
  }
  DDConstant::createConstantsFromEvaluator();
}


