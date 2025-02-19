#ifndef CASCADE_WRAPPER_H
#define CASCADE_WRAPPER_H

//-- the pyhepc routine used by Pythia to fill the HEPEVT common block
//-- is using double precision and 4000 entries

#include <ctype.h>
#include <cstring>

//-- CASCADE Common Block Declarations

extern "C" {
  extern struct {
    int ke,kp,keb,kph,kgl,kpa,nflav;
    } caluco_;
  }

#define caluco caluco_

extern "C" {
  extern struct {
    int lst[30],ires[2];
  } capar6_;
}

#define capar6 capar6_

extern "C" {
  extern struct {
    double plepin,ppin;
    int nfrag,ilepto,ifps,ihf,inter,isemih,ifinal;
  } cainpu_;
}

#define cainpu cainpu_

extern "C" {
  extern struct {
    int ipst;
  } cashower_;
}

#define cashower cashower_

extern "C" {
  extern struct {
    int ipsipol,ipsiel1,ipsiel2;
    //-- int i23s;  //-- from version 2.2.03 on 
  } jpsi_;
}

#define jpsi jpsi_

extern "C" {
  extern struct {
    int iorder,itimshr,iccfm;
  } casshwr_;
}

#define casshwr casshwr_

extern "C" {
  extern struct {
    int ipro,iruna,iq2,irunaem;
  } capar1_;
}

#define capar1 capar1_

extern "C" {
  extern struct {
    int ihfla,kpsi,kchi;
  } cahflav_;
}

#define cahflav cahflav_

extern "C" {
  extern struct {
    int icolora,irespro,irpa,irpb,irpc,irpd,irpe,irpf,irpg;
  } cascol_;
}

#define cascol cascol_

extern "C" {
  extern struct {
    int iglu;
  } cagluon_;
}

#define cagluon cagluon_

extern "C" {
  extern struct {
    int irspl;
  } casprre_;
}

#define casprre casprre_

extern "C" {
  extern struct {
    double pt2cut[1000];
  } captcut_;
}

#define captcut captcut_

extern "C" {
  extern struct {
    double acc1,acc2;
    int iint,ncb;
  } integr_;
}

#define integr integr_

extern "C" {
  extern struct {
    double scalfa,scalfaf;
  } scalf_;
}

#define scalf scalf_

extern "C" {
  extern struct {
    char pdfpath[512];
  } caspdf_;
}

#define caspdf caspdf_

extern "C" {
  extern struct {
    double avgi,sd;
    int nin,nout;
  } caeffic_;
}

#define caeffic caeffic_

//-- CASCADE routines declarations

extern "C" {
  void casini_();
  void steer_();
  void cascha_();
  void cascade_();
  void caend_(int* mode);
  void event_();
  // void rluxgo_(int* mode1,int* mode2,int* mode3,int* mode4);
  double dcasrn_(int* idummy);
  
}

inline void call_casini() { casini_(); }
inline void call_steer() { steer_(); }
inline void call_cascha() { cascha_(); }
inline void call_cascade() { cascade_(); }
inline void call_caend(int mode) { caend_(&mode); }
inline void call_event() { event_(); }
// inline void call_rluxgo(int mode1, int mode2, int mode3, int mode4) { rluxgo_(&mode1, &mode2, &mode3, &mode4); }
// inline double call_dcasrn() { return(dcasrn_()); }

//-- PYTHIA Common Block Declarations

//-- PYTHIA routines declarations

extern "C" {
  void pytcha_();
  void pyedit_(int* mode);
}

inline void call_pytcha() { pytcha_(); }
inline void call_pyedit(int mode){ pyedit_(&mode); }

//inline void call_pyhepc(int mode) { pyhepc(&mode); }
//inline void call_pylist(int mode) { pylist(&mode); }
//inline void call_pystat(int mode) { pystat(&mode); }
//inline void call_pyevnt() { pyevnt(); }

#endif  // CASCADE_WRAPPER_H

