#ifndef PhysicsTools_Utilities_interface_LumiReWeighting_h
#define PhysicsTools_Utilities_interface_LumiReWeighting_h


/**
  \class    LumiReWeighting LumiReWeighting.h "PhysicsTools/Utilities/interface/LumiReWeighting.h"
  \brief    Class to provide lumi weighting for analyzers to weight "flat-to-N" MC samples to data

  This class will trivially take two histograms:
  1. The generated "flat-to-N" distributions from a given processing
  2. A histogram generated from the "estimatePileup" macro here:

  https://twiki.cern.ch/twiki/bin/view/CMS/LumiCalc#How_to_use_script_estimatePileup

  \author Salvatore Rappoccio
*/

#include "TH1.h"
#include "TFile.h"
#include <string>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace edm {
  class EventBase;
  class LumiReWeighting {
  public:
    LumiReWeighting( std::string generatedFile,
		     std::string dataFile,
		     std::string GenHistName,
		     std::string DataHistName);
    
    LumiReWeighting( std::vector< float > MC_distr, std::vector< float > Lumi_distr);

    LumiReWeighting ( ) { } ;

    double weight( int npv ) ;

    double weight( const edm::EventBase &e ) ;

    double weightOOT( const edm::EventBase &e ) ;

    void weightOOT_init(); 

  protected:

    std::string generatedFileName_;
    std::string dataFileName_;
    std::string GenHistName_;
    std::string DataHistName_;
    boost::shared_ptr<TFile>     generatedFile_;
    boost::shared_ptr<TFile>     dataFile_;
    boost::shared_ptr<TH1F>      weights_;

    double WeightOOTPU_[25][25];

    int  OldLumiSection_;
    bool Reweight_4_2_2p2_;
    bool FirstWarning_;

  };
}



#endif
