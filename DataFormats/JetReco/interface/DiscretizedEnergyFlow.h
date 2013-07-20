/** \class DiscretizedEnergyFlow
 *
 * \short A grid filled with discretized energy flow 
 *
 * This is a pure storage class with limited functionality.
 * Applications should use fftjet::Grid2d. This object
 * is not sparsified and should be dropped before the event
 * is written out.
 *
 * \author Igor Volobouev, TTU, June 28, 2010
 * \version  $Id: DiscretizedEnergyFlow.h,v 1.1 2011/07/18 17:00:23 igv Exp $
 ************************************************************/

#ifndef DataFormats_FFTJetAlgorithms_DiscretizedEnergyFlow_h
#define DataFormats_FFTJetAlgorithms_DiscretizedEnergyFlow_h

#include <vector>
#include <string>

namespace reco {
  class DiscretizedEnergyFlow
  {
  public:
    inline DiscretizedEnergyFlow() 
        : title_(""), etaMin_(0.0), etaMax_(0.0), phiBin0Edge_(0.0),
          nEtaBins_(0), nPhiBins_(0) {}

    DiscretizedEnergyFlow(const double* data, const char* title,
                          double etaMin, double etaMax,
                          double phiBin0Edge, unsigned nEtaBins,
                          unsigned nPhiBins);

    inline const double* data() const
    {
        if (data_.empty()) return 0;
        else return &data_[0];
    }
    inline const char* title() const {return title_.c_str();}
    inline double etaMin() const {return etaMin_;}
    inline double etaMax() const {return etaMax_;}
    inline double phiBin0Edge() const {return phiBin0Edge_;}
    inline unsigned nEtaBins() const {return nEtaBins_;}
    inline unsigned nPhiBins() const {return nPhiBins_;}
    
  private:
    std::vector<double> data_;
    std::string title_;
    double etaMin_;
    double etaMax_;
    double phiBin0Edge_;
    unsigned nEtaBins_;
    unsigned nPhiBins_;
  };
}

#endif // DataFormats_FFTJetAlgorithms_DiscretizedEnergyFlow_h
