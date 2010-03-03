#ifndef __Cent_Bin_h__
#define __Cent_Bin_h__

//#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include <TNamed.h>
#include <vector>


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

   CentralityBins(){;}
   CentralityBins(const char* name, const char* title, int nbins) : TNamed(name,title), nbins_(nbins) {
      table_.reserve(nbins_);
      for(int j = 0; j < nbins_; ++j){
	 CBin b;
	 table_.push_back(b); 
      }
   }
      ~CentralityBins() {;}
      int getBin(double value) const;
      int getNbins() const {return nbins_;}
      int getNbinsMax() const {return table_.size();}
      void setNbins(int nb) const {nbins_ = nb;}
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

      /*      double nPartMean(double value){ return table_.m_table[getBin(value)].n_part_mean;}
      double nPartMeanOfBin(int bin){ return table_.m_table[bin].n_part_mean;}
      double nPartSigma(double value){ return table_.m_table[getBin(value)].n_part_var;}
      double nPartSigmaOfBin(int bin){ return table_.m_table[bin].n_part_var;}
      */
      
      // private:
      mutable int nbins_;
      std::vector<CBin> table_;
      //      std::vector<CentralityTable::CBin> table_;

   ClassDef(CentralityBins,2)
};

#endif
