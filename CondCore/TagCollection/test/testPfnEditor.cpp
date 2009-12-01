#include "CondCore/TagCollection/interface/PfnEditor.h"

#include<iostream>

int err = 0;

void error(char const * m) {
  ++err;
  std::cerr << m << std::endl;
}

int main() {
  
  std::string det("STRIP");
  std::string cond("CMS_COND_31X_");
  std::string pfn("oracle://cmsarc_lb/CMS_COND_31X_STRIP");
  std::string pre1("oracle://cmsarc_lb/");
  std::string pre2("oracle://cmsprod/");
  
  std::string post("_0911");
  
  {
    cond::PfnEditor ed("","");
    if (ed(pfn)!=pfn) 
      error("error adding null pre and null post");
  }

  {
    cond::PfnEditor ed("",post);
    if (ed(pfn)!=pfn+post) 
      error("error adding null pre and post");
  }

  {
    cond::PfnEditor ed(pre2,"");
    if (ed(pfn)!=pre2+cond+det) 
      error("error changing pre and null post");
  }

  {
    cond::PfnEditor ed(pre2,post);
    if (ed(pfn)!=pre2+cond+det+post) 
      error("error changing pre and post");
  }

  {
    cond::PfnEditor ed(pre1+cond,post);
    if (ed(det)!=pre1+cond+det+post) 
      error("error adding pre and post");
  }


  return err;
}
