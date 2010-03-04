#ifndef __Cent_Bin_h__
#define __Cent_Bin_h__

#include <TNamed.h>
#include <TFile.h>
#include <vector>
#include <map>

class CBin : public TObject {
 public:
   CBin(){;}
   ~CBin(){;}

   double bin_edge;
   double n_part_mean;
   double n_part_var;
   double n_coll_mean;
   double n_coll_var;
   double n_hard_mean;
   double n_hard_var;
   double b_mean;
   double b_var;
   ClassDef(CBin,1)
};

class CentralityBins : public TNamed {
   
 public:
   typedef std::map<int, const CentralityBins*> RunMap;

   CentralityBins(){;}
   CentralityBins(const char* name, const char* title, int nbins) : TNamed(name,title) {
      table_.reserve(nbins);
      for(int j = 0; j < nbins; ++j){
	 CBin b;
	 table_.push_back(b); 
      }
   }
      ~CentralityBins() {;}
      int getBin(double value) const;
      int getNbins() const {return table_.size();}
      double lowEdge(double value) const { return lowEdgeOfBin(getBin(value));}
      double lowEdgeOfBin(int bin) const { return table_[bin].bin_edge;}
      double NpartMean(double value) const { return NpartMeanOfBin(getBin(value));}
      double NpartMeanOfBin(int bin) const { return table_[bin].n_part_mean;}
      double NpartSigma(double value) const { return NpartSigmaOfBin(getBin(value));}
      double NpartSigmaOfBin(int bin) const { return table_[bin].n_part_var;}
      double NcollMean(double value) const { return NcollMeanOfBin(getBin(value));}
      double NcollMeanOfBin(int bin) const { return table_[bin].n_coll_mean;}
      double NcollSigma(double value) const { return NcollSigmaOfBin(getBin(value));}
      double NcollSigmaOfBin(int bin) const { return table_[bin].n_coll_var;}
      double NhardMean(double value) const { return NhardMeanOfBin(getBin(value));}
      double NhardMeanOfBin(int bin) const { return table_[bin].n_hard_mean;}
      double NhardSigma(double value) const { return NhardSigmaOfBin(getBin(value));}
      double NhardSigmaOfBin(int bin) const { return table_[bin].n_hard_var;}
      double bMean(double value) const { return bMeanOfBin(getBin(value));}
      double bMeanOfBin(int bin) const { return table_[bin].b_mean;}
      double bSigma(double value) const { return bSigmaOfBin(getBin(value));}
      double bSigmaOfBin(int bin) const { return table_[bin].b_var;}

      // private:
      std::vector<CBin> table_;
      ClassDef(CentralityBins,1)
};

CentralityBins::RunMap getCentralityFromFile(TFile*, const char* tag, int firstRun = 0, int lastRun = 10);





#endif
