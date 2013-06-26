#ifndef CalibMuon_DTCalibration_DTVDriftSegment_h
#define CalibMuon_DTCalibration_DTVDriftSegment_h

/** \class DTVDriftSegment
 *  Concrete implementation of a DTVDriftBaseAlgo.
 *  Computes vDrift using fit result segment by segment.
 *
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTVDriftBaseAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class DTMtime;
class DTResidualFitter;
class TH1F;
class TFile;

class DTVDriftSegment: public DTVDriftBaseAlgo {
public:
   DTVDriftSegment(edm::ParameterSet const&);
   virtual ~DTVDriftSegment();

   virtual void setES(const edm::EventSetup& setup);
   virtual DTVDriftData compute(const DTSuperLayerId&);
private:
   TH1F* getHisto(const DTSuperLayerId&);
   std::string getHistoName(const DTSuperLayerId&);

   unsigned int nSigmas_;

   const DTMtime* mTimeMap_;
   TFile* rootFile_;
   DTResidualFitter* fitter_;
};
#endif
