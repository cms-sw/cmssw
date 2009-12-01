#include "CondCore/TagCollection/interface/PfnEditor.h"

#include<iostream>

int err = 0;

void error(char * m) {
  ++err;
  std::cerr << m << std::endl;
}

int main() {

  std::string det("STRIP");

  std::string pfn("oracle://cmsarc_lb/CMS_COND_31X_STRIP");
  std::string pre1("oracle://cmsarc_lb/");
  std::string pre2("oracle://cmsprod/");

  std::post("_0911");

  {
    PfnEditor ed("","");
    if (ed.pfn(pfn)!=pfn) 
      error("error adding null pre and null post");
  }

  {
    PfnEditor ed("",post);
    if (ed.pfn(pfn)!=pfn+post) 
      error("error adding null pre and post");
  }

  {
    PfnEditor ed(pre2,"");
    if (ed.pfn(pfn)!=pre2+de) 
      error("error changin pre and null post");
  }

  {
    PfnEditor ed(pre2,post);
    if (ed.pfn(pfn)!=pre2+det+post) 
      error("error changing pre and post");
  }

  {
    PfnEditor ed(pre1,post);
    if (ed.pfn(det)!=pre1+det+post) 
      error("error adding pre and post");
  }


  return err;
}
