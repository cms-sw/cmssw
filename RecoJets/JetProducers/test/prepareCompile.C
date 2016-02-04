{
  char name[4096],stub[1024];
  sprintf(name,"%s/src",getenv("CMSSW_BASE"));
  gROOT->GetInterpreter()->AddIncludePath(name);
  sprintf(name,"%s/src",getenv("CMSSW_RELEASE_BASE"));
  gROOT->GetInterpreter()->AddIncludePath(name);
  sprintf(name,"%s/include",getenv("SEAL"));
  gROOT->GetInterpreter()->AddIncludePath(name);
  sprintf(name,"%s/%s/include",getenv("SEAL"),getenv("SCRAMRT_CMS_SYS"));
  gROOT->GetInterpreter()->AddIncludePath(name);
  //ccla
  sprintf(name,"%s/include",getenv("CLHEP_PARAM_PATH"));

  gROOT->GetInterpreter()->AddIncludePath(name);
// Get the BOOST include path!
  char* ldl=getenv("LD_LIBRARY_PATH");
  char* pos=strstr(ldl,"boost");
  char* pp;
  for (pp=pos;*pp!=':' && pp!=ldl; pp--);
  if (*pp==':') pp++;
  pos=strstr(pp,"lib");
  memset(stub,0,1000); // zero it out
  for (int i=0; pp!=pos; pp++) 
    stub[i++]=*pp;
  sprintf(name,"%sinclude/boost-1_33_1",stub);
  gROOT->GetInterpreter()->AddIncludePath(name);

  sprintf(name,"-L%s/lib/%s",getenv("CMSSW_BASE"),getenv("SCRAMRT_CMS_SYS"));
  gSystem->AddLinkedLibs(name);
  sprintf(name,"-L%s/lib/%s",getenv("CMSSW_RELEASE_BASE"),getenv("SCRAMRT_CMS_SYS"));
  gSystem->AddLinkedLibs(name);
  gSystem->AddLinkedLibs("-lDataFormatsJetReco");
  //ccla 
  gSystem->AddLinkedLibs("-lSimDataFormatsHepMCProduct");
  //gSystem->AddLinkedLibs("-lDataFormatsMETReco");
  //gSystem->AddLinkedLibs("-lDataFormatsMETObjects");
}
