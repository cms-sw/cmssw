#ifndef DataFormats_SiStripDigi_Histo_H
#define DataFormats_SiStripDigi_Histo_H

#include "TH1F.h"

class Histo {

 public:

  Histo() : hist_() {;}
  Histo(const TH1F& hist) : hist_(hist) {;}
  ~Histo() {;}

 inline const TH1F& get() const { return hist_;}
 inline void set(const TH1F& hist) {hist_ = hist;} 
 inline bool operator<(const Histo& compare) const
    { return get().GetEntries() < compare.get().GetEntries(); }

private:

  TH1F hist_;

};

#endif //DataFormats_SiStripDigi_Histo_H
