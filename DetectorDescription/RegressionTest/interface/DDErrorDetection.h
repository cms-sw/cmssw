#ifndef DD_DDErrorDetection_h
#define DD_DDErrorDetection_h

#include <map>
#include <set>
#include <string>

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include <DetectorDescription/Core/interface/DDCompactView.h>

//=================
#include "DetectorDescription/Base/interface/Singleton.h"
typedef DDI::Singleton<std::map<std::string,std::set<DDLogicalPart> > > lp_err;
typedef DDI::Singleton<std::map<std::string,std::set<DDMaterial> > >    ma_err;
typedef DDI::Singleton<std::map<std::string,std::set<DDSolid> > >       so_err;
typedef DDI::Singleton<std::map<std::string,std::set<DDRotation> > >    ro_err;
typedef DDI::Singleton<std::map<std::string,std::set<DDSpecifics> > >   sp_err;

//==================
//*********************************************************************************************************************************

typedef std::map<std::string, std::set<std::string> > ns_type;
typedef std::map<std::string, std::set<DDName> > ns_nm_type;

//*********************************************************************************************************************************



template<class T> std::ostream & operator<<(std::ostream & o, const std::set<T> & v)
{
  typename std::set<T>::const_iterator it(v.begin()), ed(v.end());
  for(; it != ed; ++it) {
    o << it->ddname() << ' ';
  }
  return o;
}

/*
ostream & operator<<(ostream & o, const std::set<DDLogicalPart> & v)
{
  std::set<DDLogicalPart>::const_iterator it(v.begin()), ed(v.end());
  for(; it != ed; ++it) {
    o << it->ddname().name() << ' ';
  }
  return o;
}
*/
template<class T> std::ostream & operator<<(std::ostream & o, const std::map<std::string, std::set<T> > & m) {
  typedef typename std::map<std::string, std::set<T> >::const_iterator c_it;
  c_it it(m.begin()), ed(m.end());
  for (; it != ed; ++it) {
    o << it->first << ": " << it->second;
    o << std::endl;
  }
  return o;
}

template<class T, class N> std::ostream & operator<<(std::ostream & o, const std::map<N, std::set<T> > & m) {
  typedef typename std::map<N, std::set<T> >::const_iterator c_it;
  c_it it(m.begin()), ed(m.end());
  for (; it != ed; ++it) {
    o << it->first.ddname() << ": " << it->second;
    o << std::endl;
  }
  return o;
}

template<typename T>
bool findNameSpaces(T dummy, ns_type & m)
{
   bool result=true;
   typename T::template iterator<T> it,ed;
   ed.end();
   for (; it != ed; ++it) {
     result = it->isDefined().second;
     if (!result) 
       DDI::Singleton<std::map<std::string,std::set<T> > >::instance()[it->name().ns()].insert(*it);
     m[it->name().ns()].insert(it->name().name());
   }
   return result;
}


template<typename T>
bool findNameSpaces(T dummy, ns_nm_type & m)
{
   bool result=true;
   typename T::template iterator<T> it,ed;
   ed.end();
   for (; it != ed; ++it) {
     result = it->isDefined().second;
     if (!result) 
       DDI::Singleton<std::map<std::string,std::set<T> > >::instance()[it->name().ns()].insert(*it);
     m[it->name().ns()].insert(it->name().name());
   }
   return result;
}


template <class C> const std::map<std::string, std::set<C> > & dd_error_scan(const C &)
{
    typedef std::map<std::string, std::set<C> > error_type;
    static error_type result_;
    typename C::template iterator<C> it;
    typename C::template iterator<C> ed(C::end());
    for (; it != ed; ++it) {
      if (!it->isDefined().second) {
        result_[it->name().ns()].insert(*it);
      }
    }  
    return result_;
}

class DDErrorDetection
{
public:
  DDErrorDetection(const DDCompactView& cpv);    
  ~DDErrorDetection();

  void scan( const DDCompactView& cpv);
  
  void errors();
  
  void warnings();
  
  const std::map<std::string, std::set<DDLogicalPart> > & lp_cpv(const DDCompactView& cpv);
  const std::map<DDMaterial, std::set<DDLogicalPart> > & ma_lp();
  const std::map<DDSolid, std::set<DDLogicalPart> > & so_lp();
  const std::map<DDSolid, std::set<DDSolid> > & so();

  void nix();
  
  const std::vector<std::pair<std::string,DDName> > &  ma();

  void report(const DDCompactView& cpv, std::ostream & o); 

 private:
  DDErrorDetection() { };

};

#endif
