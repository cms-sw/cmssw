#include "DetectorDescription/Core/interface/DDAlgo.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"

std::ostream & operator<<(std::ostream & os, const DDAlgo & algo)
{
 DDBase<DDName,AlgoPos*>::def_type defined(algo.isDefined());
  if (defined.first) {
    os << *(defined.first) << " ";
    if (defined.second) {
      algo.rep().stream(os); 
    }
    else {
      os << "* algorithm not defined * ";  
    }
  }  
  else {
    os << "* algorithm not declared * ";  
  }  
  return os;
  
}


DDAlgo::DDAlgo() : DDBase<DDName,AlgoPos*>()
{ } 


DDAlgo::DDAlgo(const DDName & name) : DDBase<DDName,AlgoPos*>()
{ 
  prep_ = StoreT::instance().create(name);
}


DDAlgo::DDAlgo(const DDName & name, AlgoPos * a) : DDBase<DDName,AlgoPos*>()

{ 
  prep_ = StoreT::instance().create(name,a);
}



void DDAlgo::setParameters(int start, int end, int incr,
                         const parS_type & ps, const parE_type & pe)
{
   rep().setParameters(start, end, incr, ps, pe);
}			 


int DDAlgo::start() const
{
  return rep().start();
}

int DDAlgo::end() const
{
  return rep().end();
}

int DDAlgo::incr() const
{
  return rep().incr();
}

const parS_type & DDAlgo::parS() const
{
  return rep().parS();
}

const parE_type & DDAlgo::parE() const
{
  return rep().parE();
}


DDTranslation DDAlgo::translation()
{
  return rep().translation();
}

  
DDRotationMatrix DDAlgo::rotation()
{
  return rep().rotation();
}


int DDAlgo::copyno() const
{
  return rep().copyno();
}


#include <cstdio>
std::string DDAlgo::label() const
{
  char buffer [50]; 
  /*int n =*/ sprintf(buffer,"%d",copyno());
  return std::string(buffer);
}


void DDAlgo::next()
{
  rep().next();
}


bool DDAlgo::go() const
{
  return rep().go();
}
  
// friend, factory function
DDAlgo DDalgo(const DDName & n, AlgoPos * a)
{
  return DDAlgo(n,a);
}


// void DDAlgo::clear()
// {
//  StoreT::instance().clear();
// }
