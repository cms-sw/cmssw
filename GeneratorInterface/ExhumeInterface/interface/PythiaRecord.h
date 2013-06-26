////PythiaInterface.h
#ifndef PYTHIA_RECORD_HH
#define PYTHIA_RECORD_HH

#define py1ent py1ent_
extern "C" {
  void py1ent(int&,int&,double&,double&,double&);
}

#define pyexec pyexec_
extern "C" {
  void pyexec();
}

#define pyinre pyinre_
extern "C" {
  void pyinre();
}

#define pyjoin pyjoin_
extern "C" {
  void pyjoin(int&,int[2]);
}

#define pyshow pyshow_
extern "C" {
  void pyshow(int&,int&,double&);
}

#endif

