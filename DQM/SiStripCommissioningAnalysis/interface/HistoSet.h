#ifndef DQM_SiStripCommissioningAnalysis_HistoSet_H
#define DQM_SiStripCommissioningAnalysis_HistoSet_H

class TH1F;

using namespace std;

/** Simple container class that owns a set of TH1F histograms that are
    commonly used by the commissioning tasks. */
class HistoSet {
  
 public:
  
  HistoSet( TH1F* sum2, TH1F* sum, TH1F* num ) : 
  sumOfSquares_(sum2), sumOfContents_(sum), numOfEntries_(num) {;}
  
  ~HistoSet() {;}
  
  inline const TH1F* const sumOfSquares()  const { return sumOfSquares_; }
  inline const TH1F* const sumOfContents() const { return sumOfContents_; }
  inline const TH1F* const numOfEntries()  const { return numOfEntries_; }
  
 private:

  HistoSet() {;}

  TH1F* sumOfSquares_;
  TH1F* sumOfContents_;
  TH1F* numOfEntries_;

};

#endif // DQM_SiStripCommissioningAnalysis_HistoSet_H

